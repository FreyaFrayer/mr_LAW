#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize many IK solutions of Franka Panda in RViz simultaneously by publishing a MarkerArray.

- Reads a JSON file with multiple IK solutions (joint_names + joint_positions per solution)
- Uses MoveIt 2 Python API (moveit_py) RobotState to compute forward kinematics
- Parses the Panda URDF to extract visual meshes and their local offsets
- Publishes one semi-transparent robot per IK solution (many robots at once)

RViz:
- Add a "MarkerArray" display for the published topic (default: /panda_ik_solutions_markers)
- Set Fixed Frame to /panda_link0 (or whatever RobotModel.model_frame is)
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from urdf_parser_py.urdf import URDF, Mesh, Box, Cylinder, Sphere

# MoveItPy (moveit_py) bindings
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder


# ------------------------ math helpers ------------------------ #
def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a 3x3 rotation matrix from RPY (URDF convention: Rz*Ry*Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr,  cr]], dtype=float)
    return Rz @ Ry @ Rx


def _xyz_rpy_to_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Return a 4x4 homogeneous transform from xyz + rpy."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.array([xyz[0], xyz[1], xyz[2]], dtype=float)
    return T


def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n > 0.0:
        x, y, z, w = x / n, y / n, z / n, w / n
    return (x, y, z, w)


def _matrix_to_pose(T: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


# ------------------------ URDF visual extraction ------------------------ #
@dataclass(frozen=True)
class VisualElement:
    link_name: str
    origin_T_link: np.ndarray  # link_T_visual (4x4)
    geom_type: str  # "mesh" | "box" | "cylinder" | "sphere"
    mesh_resource: Optional[str] = None
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    box_size: Optional[Tuple[float, float, float]] = None
    cyl_radius: Optional[float] = None
    cyl_length: Optional[float] = None
    sphere_radius: Optional[float] = None


def _extract_link_visuals(urdf_xml: str) -> Dict[str, List[VisualElement]]:
    robot = URDF.from_xml_string(urdf_xml)

    visuals_by_link: Dict[str, List[VisualElement]] = {}
    for link in robot.links:
        link_visuals: List = []
        if getattr(link, "visual", None) is not None:
            link_visuals.append(link.visual)
        if getattr(link, "visuals", None):
            for v in link.visuals:
                if v is not None and v not in link_visuals:
                    link_visuals.append(v)

        if not link_visuals:
            continue

        out: List[VisualElement] = []
        for v in link_visuals:
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if getattr(v, "origin", None) is not None:
                if v.origin.xyz is not None:
                    xyz = tuple(float(x) for x in v.origin.xyz)
                if v.origin.rpy is not None:
                    rpy = tuple(float(r) for r in v.origin.rpy)
            T = _xyz_rpy_to_matrix(xyz, rpy)

            geom = v.geometry
            if isinstance(geom, Mesh):
                scale = (1.0, 1.0, 1.0)
                if getattr(geom, "scale", None) is not None and len(geom.scale) == 3:
                    scale = (float(geom.scale[0]), float(geom.scale[1]), float(geom.scale[2]))
                out.append(
                    VisualElement(
                        link_name=link.name,
                        origin_T_link=T,
                        geom_type="mesh",
                        mesh_resource=str(geom.filename),
                        mesh_scale=scale,
                    )
                )
            elif isinstance(geom, Box):
                out.append(
                    VisualElement(
                        link_name=link.name,
                        origin_T_link=T,
                        geom_type="box",
                        box_size=(float(geom.size[0]), float(geom.size[1]), float(geom.size[2])),
                    )
                )
            elif isinstance(geom, Cylinder):
                out.append(
                    VisualElement(
                        link_name=link.name,
                        origin_T_link=T,
                        geom_type="cylinder",
                        cyl_radius=float(geom.radius),
                        cyl_length=float(geom.length),
                    )
                )
            elif isinstance(geom, Sphere):
                out.append(
                    VisualElement(
                        link_name=link.name,
                        origin_T_link=T,
                        geom_type="sphere",
                        sphere_radius=float(geom.radius),
                    )
                )
            else:
                continue

        if out:
            visuals_by_link[link.name] = out

    return visuals_by_link


# ------------------------ config utils (FIX planning pipelines) ------------------------ #
def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively merge src into dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_if_exists(path: str) -> Optional[dict]:
    if not path or (not os.path.exists(path)):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        return data
    return None


def _ensure_minimal_planning_pipelines(cfg_dict: Dict) -> None:
    """
    Ensure MoveItCpp can see at least one planning pipeline.
    This is a *fallback* in case your config package didn't provide moveit_cpp.yaml / planning pipeline configs.
    """
    pp = cfg_dict.get("planning_pipelines")
    if not isinstance(pp, dict):
        cfg_dict["planning_pipelines"] = {}
        pp = cfg_dict["planning_pipelines"]

    # pipeline_names is what MoveItCpp expects
    names = pp.get("pipeline_names")
    if not isinstance(names, list) or len(names) == 0:
        pp["pipeline_names"] = ["ompl"]

    # ensure ompl sub-config exists
    if "ompl" not in pp or not isinstance(pp.get("ompl"), dict):
        pp["ompl"] = {}

    ompl = pp["ompl"]
    # minimal required keys (typical defaults)
    ompl.setdefault("planning_plugin", "ompl_interface/OMPLPlanner")
    ompl.setdefault(
        "request_adapters",
        "default_planner_request_adapters/AddTimeOptimalParameterization "
        "default_planner_request_adapters/FixWorkspaceBounds "
        "default_planner_request_adapters/FixStartStateBounds "
        "default_planner_request_adapters/FixStartStateCollision "
        "default_planner_request_adapters/FixStartStatePathConstraints",
    )
    ompl.setdefault("start_state_max_bounds_error", 0.1)


def _get_moveit_config_dict() -> Tuple[Dict, str]:
    """
    Build a MoveIt config dict for Panda that includes moveit_cpp.yaml + planning pipelines.
    Returns: (config_dict, selected_config_package_name)
    """
    candidates = ["moveit_resources_panda_moveit_config", "panda_moveit_config"]
    last_err = None

    for pkg in candidates:
        try:
            pkg_share = get_package_share_directory(pkg)

            # Build configs with explicit planning pipeline + moveit_cpp
            builder = MoveItConfigsBuilder(robot_name="panda", package_name=pkg)

            # Some distros/builds require explicit component calls, but they are safe to call.
            for fn in ("robot_description", "robot_description_semantic", "robot_description_kinematics"):
                if hasattr(builder, fn):
                    builder = getattr(builder, fn)()

            # planning_pipelines(): in some versions it accepts pipelines=[...]
            if hasattr(builder, "planning_pipelines"):
                try:
                    builder = builder.planning_pipelines()
                except TypeError:
                    builder = builder.planning_pipelines(pipelines=["ompl"])

            # moveit_cpp(): loads config/moveit_cpp.yaml (very important for planning_pipelines.pipeline_names)
            if hasattr(builder, "moveit_cpp"):
                try:
                    builder = builder.moveit_cpp()
                except TypeError:
                    # some versions take file_path=...
                    builder = builder.moveit_cpp(file_path=None)

            moveit_cfg = builder.to_moveit_configs()
            cfg_dict = moveit_cfg.to_dict()

            # ---- Extra safeguard: merge YAMLs manually if they exist ----
            # In some releases, MoveItConfigsBuilder may not auto-include moveit_cpp.yaml in to_dict().
            moveit_cpp_yaml = os.path.join(pkg_share, "config", "moveit_cpp.yaml")
            ompl_yaml = os.path.join(pkg_share, "config", "ompl_planning.yaml")

            y = _load_yaml_if_exists(moveit_cpp_yaml)
            if y:
                _deep_update(cfg_dict, y)

            y = _load_yaml_if_exists(ompl_yaml)
            if y:
                _deep_update(cfg_dict, y)

            # Final fallback: ensure minimal planning pipeline exists
            _ensure_minimal_planning_pipelines(cfg_dict)

            return cfg_dict, pkg

        except PackageNotFoundError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not find any Panda MoveIt config package {candidates}. Last error: {last_err}"
    )


def _extract_urdf_xml(moveit_cfg_dict: Dict) -> str:
    if "robot_description" not in moveit_cfg_dict:
        raise KeyError("moveit config dict does not contain 'robot_description'")

    urdf_xml = moveit_cfg_dict["robot_description"]
    if not isinstance(urdf_xml, str) or "<robot" not in urdf_xml:
        raise ValueError("robot_description is not plain URDF XML (string).")
    return urdf_xml


# ------------------------ main node ------------------------ #
class PandaIKSolutionsMarkers(Node):
    def __init__(self):
        super().__init__("panda_ik_solutions_markers")

        # ROS parameters (use --ros-args -p xxx:=yyy)
        self.declare_parameter("json_path", "p1.json")
        self.declare_parameter("topic", "/panda_ik_solutions_markers")
        self.declare_parameter("alpha", 0.18)  # transparency for robot meshes
        self.declare_parameter("use_embedded_materials", False)
        self.declare_parameter("max_solutions", 0)  # 0 = all
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("target_sphere_scale", 0.02)
        self.declare_parameter("eef_points", True)
        self.declare_parameter("eef_sphere_scale", 0.01)

        self._json_path = self.get_parameter("json_path").get_parameter_value().string_value
        self._topic = self.get_parameter("topic").get_parameter_value().string_value
        self._alpha = float(self.get_parameter("alpha").value)
        self._use_embedded_materials = bool(self.get_parameter("use_embedded_materials").value)
        self._max_solutions = int(self.get_parameter("max_solutions").value)
        self._publish_hz = float(self.get_parameter("publish_hz").value)
        self._target_sphere_scale = float(self.get_parameter("target_sphere_scale").value)
        self._eef_points = bool(self.get_parameter("eef_points").value)
        self._eef_sphere_scale = float(self.get_parameter("eef_sphere_scale").value)

        self.get_logger().info(f"Loading IK solutions from: {self._json_path}")
        with open(self._json_path, "r", encoding="utf-8") as f:
            self._ik_data = json.load(f)

        meta = self._ik_data.get("meta", {})
        self._group_name = str(meta.get("group", "panda_arm"))
        self._tip_link = str(meta.get("tip_link", "panda_link8"))
        self._target_point = meta.get("target_point", None)

        # Build MoveIt config + instantiate MoveItPy so we can compute FK
        self.get_logger().info("Building MoveIt config for Panda (with planning pipelines) ...")
        moveit_cfg_dict, used_pkg = _get_moveit_config_dict()
        self.get_logger().info(f"Using MoveIt config package: {used_pkg}")

        # Optional: print the pipelines it thinks it has (debug)
        pp = moveit_cfg_dict.get("planning_pipelines", {})
        self.get_logger().info(f"planning_pipelines.pipeline_names = {pp.get('pipeline_names', None)}")

        self.get_logger().info("Starting MoveItPy (for FK/robot model) ...")
        self._moveit = MoveItPy(
            node_name="panda_ik_viz_moveit_py",
            config_dict=moveit_cfg_dict,
            provide_planning_service=False,  # we only need robot model + FK
        )

        self._robot_model = self._moveit.get_robot_model()
        self._model_frame = self._robot_model.model_frame  # usually "panda_link0"
        self.get_logger().info(f"RobotModel loaded. model_frame = '{self._model_frame}'")

        if not self._robot_model.has_joint_model_group(self._group_name):
            raise RuntimeError(
                f"Joint model group '{self._group_name}' not found. "
                f"Available groups: {self._robot_model.joint_model_group_names}"
            )

        self._jmg = self._robot_model.get_joint_model_group(self._group_name)
        self._link_names = list(self._jmg.link_model_names)

        # Parse URDF visuals
        urdf_xml = _extract_urdf_xml(moveit_cfg_dict)
        self.get_logger().info("Parsing URDF visuals (meshes) ...")
        self._visuals_by_link = _extract_link_visuals(urdf_xml)

        # Keep visuals only for links in planning group (reduces marker count)
        self._visual_links = [ln for ln in self._link_names if ln in self._visuals_by_link]

        self.get_logger().info("Precomputing markers for IK solutions ...")
        self._marker_array = self._build_marker_array()

        self._pub = self.create_publisher(MarkerArray, self._topic, 1)

        period = 1.0 / max(self._publish_hz, 1e-3)
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Publishing MarkerArray on {self._topic} at {self._publish_hz:.2f} Hz. "
            f"RViz Fixed Frame should be '{self._model_frame}'."
        )

    def _on_timer(self):
        now = self.get_clock().now().to_msg()
        for m in self._marker_array.markers:
            m.header.stamp = now
        self._pub.publish(self._marker_array)

    def _build_marker_array(self) -> MarkerArray:
        marker_array = MarkerArray()

        # Clear all previous markers on this topic
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # Target point marker (if provided in JSON)
        if isinstance(self._target_point, dict) and all(k in self._target_point for k in ("x", "y", "z")):
            m = Marker()
            m.header.frame_id = self._model_frame
            m.ns = "target_point"
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = float(self._target_point["x"])
            m.pose.position.y = float(self._target_point["y"])
            m.pose.position.z = float(self._target_point["z"])
            s = self._target_sphere_scale
            m.scale.x = s
            m.scale.y = s
            m.scale.z = s
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)

        robot_state = RobotState(self._robot_model)

        solutions: List[Dict] = list(self._ik_data.get("solutions", []))
        if self._max_solutions and self._max_solutions > 0:
            solutions = solutions[: self._max_solutions]

        eef_points_marker = None
        if self._eef_points:
            eef_points_marker = Marker()
            eef_points_marker.header.frame_id = self._model_frame
            eef_points_marker.ns = "eef_points"
            eef_points_marker.id = 0
            eef_points_marker.type = Marker.SPHERE_LIST
            eef_points_marker.action = Marker.ADD
            s = self._eef_sphere_scale
            eef_points_marker.scale.x = s
            eef_points_marker.scale.y = s
            eef_points_marker.scale.z = s
            eef_points_marker.color.r = 1.0
            eef_points_marker.color.g = 0.3
            eef_points_marker.color.b = 0.3
            eef_points_marker.color.a = 0.9
            eef_points_marker.points = []

        # Color settings for all robot meshes
        base_r, base_g, base_b = 0.2, 0.7, 1.0
        alpha = max(0.01, min(self._alpha, 1.0))

        for sol_idx, sol in enumerate(solutions):
            joint_names = sol.get("joint_names", [])
            joint_positions = sol.get("joint_positions", [])
            if len(joint_names) != len(joint_positions):
                self.get_logger().warn(f"Solution {sol_idx}: joint_names/positions mismatch; skipping.")
                continue

            # Fill joint positions into RobotState
            # (This property setter exists in moveit_py; if you prefer, you can replace with set_variable_positions)
            robot_state.joint_positions = {str(n): float(v) for n, v in zip(joint_names, joint_positions)}
            robot_state.update()

            if eef_points_marker is not None:
                T_tip = robot_state.get_global_link_transform(self._tip_link)
                p = Point()
                p.x = float(T_tip[0, 3])
                p.y = float(T_tip[1, 3])
                p.z = float(T_tip[2, 3])
                eef_points_marker.points.append(p)

            ns = f"sol_{sol_idx:03d}"
            local_id = 0

            for link_name in self._visual_links:
                T_link = robot_state.get_global_link_transform(link_name)
                for vis in self._visuals_by_link.get(link_name, []):
                    if vis.geom_type != "mesh" or not vis.mesh_resource:
                        continue

                    T_vis = T_link @ vis.origin_T_link
                    pose = _matrix_to_pose(T_vis)

                    m = Marker()
                    m.header.frame_id = self._model_frame
                    m.ns = ns
                    m.id = local_id
                    local_id += 1

                    m.type = Marker.MESH_RESOURCE
                    m.action = Marker.ADD
                    m.pose = pose
                    m.mesh_resource = vis.mesh_resource
                    m.mesh_use_embedded_materials = bool(self._use_embedded_materials)

                    m.scale.x = vis.mesh_scale[0]
                    m.scale.y = vis.mesh_scale[1]
                    m.scale.z = vis.mesh_scale[2]

                    m.color.r = base_r
                    m.color.g = base_g
                    m.color.b = base_b
                    m.color.a = alpha

                    marker_array.markers.append(m)

        if eef_points_marker is not None:
            marker_array.markers.append(eef_points_marker)

        self.get_logger().info(
            f"Built MarkerArray with {len(marker_array.markers)} markers ({len(solutions)} solutions)."
        )
        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PandaIKSolutionsMarkers()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
