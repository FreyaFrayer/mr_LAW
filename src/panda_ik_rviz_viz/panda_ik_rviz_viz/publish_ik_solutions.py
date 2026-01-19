#!/usr/bin/env python3
import json
import math
import os
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory

from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import RobotState
from moveit_msgs.msg import RobotTrajectory


def _to_duration(seconds: float) -> Duration:
    sec = int(math.floor(seconds))
    nsec = int((seconds - sec) * 1e9)
    msg = Duration()
    msg.sec = sec
    msg.nanosec = nsec
    return msg


class IkSolutionsTrajectoryPublisher(Node):
    """
    Read IK solutions from a JSON file and publish them as a DisplayTrajectory,
    so MoveIt RViz MotionPlanning plugin can visualize them under Planned Path.
    """

    def __init__(self) -> None:
        super().__init__("ik_solutions_trajectory_publisher")

        # Parameters
        self.declare_parameter("json_path", "")
        self.declare_parameter("n", 8)
        self.declare_parameter("dt", 0.4)
        self.declare_parameter("topic", "display_planned_path")
        self.declare_parameter("model_id", "panda")
        self.declare_parameter("publish_period_s", 1.0)  # <=0 means publish once then idle

        json_path = self.get_parameter("json_path").get_parameter_value().string_value
        n = int(self.get_parameter("n").get_parameter_value().integer_value)
        dt = float(self.get_parameter("dt").get_parameter_value().double_value)
        topic = self.get_parameter("topic").get_parameter_value().string_value
        model_id = self.get_parameter("model_id").get_parameter_value().string_value
        publish_period_s = float(self.get_parameter("publish_period_s").get_parameter_value().double_value)

        if not json_path:
            raise RuntimeError("Parameter 'json_path' is empty. Please set it to your panda_ik_solutions.json path.")
        if not os.path.exists(json_path):
            raise RuntimeError(f"JSON file not found: {json_path}")
        if n <= 0:
            raise RuntimeError("Parameter 'n' must be > 0.")
        if dt <= 0.0:
            raise RuntimeError("Parameter 'dt' must be > 0.")
        if not topic:
            raise RuntimeError("Parameter 'topic' must be non-empty.")

        # QoS: make it latched-like so RViz can still get it if it starts late
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(DisplayTrajectory, topic, qos)

        self.display_msg = self._build_display_trajectory(json_path=json_path, n=n, dt=dt, model_id=model_id)

        self.get_logger().info(
            f"Publishing {n} IK solutions as DisplayTrajectory on topic '{topic}'. "
            f"json_path='{json_path}', dt={dt}s"
        )

        # Publish once immediately
        self.pub.publish(self.display_msg)

        # Optionally republish periodically (useful if RViz reconnects / you reset displays)
        self.timer = None
        if publish_period_s and publish_period_s > 0.0:
            self.timer = self.create_timer(publish_period_s, self._on_timer)

    def _on_timer(self) -> None:
        self.pub.publish(self.display_msg)

    def _build_display_trajectory(self, json_path: str, n: int, dt: float, model_id: str) -> DisplayTrajectory:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        solutions = data.get("solutions", [])
        if not isinstance(solutions, list) or len(solutions) == 0:
            raise RuntimeError("JSON has no 'solutions' list or it is empty.")

        n = min(n, len(solutions))

        # Build canonical joint order from the first solution (trim to match positions length)
        first = solutions[0]
        first_names = first.get("joint_names", [])
        first_pos = first.get("joint_positions", [])

        if not isinstance(first_names, list) or not isinstance(first_pos, list):
            raise RuntimeError("Invalid JSON: joint_names/joint_positions must be lists.")

        m = min(len(first_names), len(first_pos))
        if m == 0:
            raise RuntimeError("First solution has empty joint_names/joint_positions.")

        joint_names = list(first_names[:m])

        # Create trajectory points
        points: List[JointTrajectoryPoint] = []
        start_positions: List[float] = []

        for i in range(n):
            sol = solutions[i]
            names = sol.get("joint_names", [])
            pos = sol.get("joint_positions", [])

            if not isinstance(names, list) or not isinstance(pos, list):
                self.get_logger().warn(f"Solution[{i}] has invalid joint_names/joint_positions, skipped.")
                continue

            mm = min(len(names), len(pos))
            name_pos_map: Dict[str, float] = {str(names[j]): float(pos[j]) for j in range(mm)}

            # Ensure all canonical joints exist; if missing, fill 0.0 (and warn)
            positions_ordered: List[float] = []
            for jn in joint_names:
                if jn not in name_pos_map:
                    self.get_logger().warn(f"Solution[{i}] missing joint '{jn}', using 0.0")
                    positions_ordered.append(0.0)
                else:
                    positions_ordered.append(name_pos_map[jn])

            pt = JointTrajectoryPoint()
            pt.positions = positions_ordered
            pt.time_from_start = _to_duration((i + 1) * dt)  # start from dt to avoid all-zero timestamps
            points.append(pt)

            if i == 0:
                start_positions = positions_ordered

        if len(points) == 0:
            raise RuntimeError("No valid trajectory points were created from the first n solutions.")

        # trajectory_start RobotState
        rs = RobotState()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        js.position = start_positions
        rs.joint_state = js

        # RobotTrajectory
        jt = JointTrajectory()
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.joint_names = joint_names
        jt.points = points

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = jt  # multi_dof_joint_trajectory left empty

        # DisplayTrajectory
        msg = DisplayTrajectory()
        msg.model_id = model_id
        msg.trajectory_start = rs
        msg.trajectory = [robot_traj]
        return msg


def main() -> None:
    rclpy.init()
    node = None
    try:
        node = IkSolutionsTrajectoryPublisher()
        rclpy.spin(node)
    except Exception as e:
        # Make errors visible in ROS logs
        if node is not None:
            node.get_logger().error(str(e))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
