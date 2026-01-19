#!/usr/bin/env python3
import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def sample_rainbow_rgba(n: int):
    """
    n 个点的颜色：红→橙→黄→绿→青→蓝→紫（线性插值渐变）
    返回 [(r,g,b,a), ...]
    """
    anchors = [
        (1.0, 0.0, 0.0),   # 红
        (1.0, 0.5, 0.0),   # 橙
        (1.0, 1.0, 0.0),   # 黄
        (0.0, 1.0, 0.0),   # 绿
        (0.0, 1.0, 1.0),   # 青
        (0.0, 0.0, 1.0),   # 蓝
        (0.6, 0.0, 0.8),   # 紫（偏紫罗兰）
    ]

    if n <= 0:
        return []
    if n == 1:
        r, g, b = anchors[0]
        return [(r, g, b, 1.0)]

    m = len(anchors)
    out = []
    for i in range(n):
        t = i / (n - 1)            # 0~1
        s = t * (m - 1)            # 0~(m-1)
        idx = int(math.floor(s))
        if idx >= m - 1:
            r, g, b = anchors[-1]
        else:
            frac = s - idx
            r0, g0, b0 = anchors[idx]
            r1, g1, b1 = anchors[idx + 1]
            r = r0 + (r1 - r0) * frac
            g = g0 + (g1 - g0) * frac
            b = b0 + (b1 - b0) * frac
        out.append((r, g, b, 1.0))
    return out


class WaypointViz(Node):
    def __init__(self):
        super().__init__("waypoint_viz")

        # 参数
        self.declare_parameter("json_path", "")
        self.declare_parameter("frame_id", "panda_link0")   # RViz Fixed Frame 建议设成同一个
        self.declare_parameter("topic", "/waypoints_markers")
        self.declare_parameter("sphere_scale", 0.02)        # 点大小（米）
        self.declare_parameter("line_width", 0.005)         # 线宽（米）
        self.declare_parameter("show_labels", True)         # 是否显示 p1/p2... 文本
        self.declare_parameter("label_z_offset", 0.03)      # 文本往上抬高多少（米）
        self.declare_parameter("publish_rate", 1.0)         # Hz：周期发布（RViz 更稳定）

        json_path = self.get_parameter("json_path").value
        if not json_path:
            self.get_logger().error(
                "json_path 为空！用法示例：\n"
                "  --ros-args -p json_path:=/abs/path/target.json"
            )
            raise RuntimeError("json_path not provided")

        self.json_path = Path(json_path)
        self.frame_id = self.get_parameter("frame_id").value
        self.topic = self.get_parameter("topic").value

        # 用 TRANSIENT_LOCAL 让 RViz 后启动也能看到（类似“latched”）
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(MarkerArray, self.topic, qos)

        self._cached = None
        self._last_mtime = None

        rate = float(self.get_parameter("publish_rate").value)
        period = 1.0 / rate if rate > 0 else 1.0
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(f"Waypoints JSON: {self.json_path}")
        self.get_logger().info(f"Publishing MarkerArray: {self.topic}")
        self.get_logger().info(f"Frame: {self.frame_id}")

    def load_points(self):
        data = json.loads(self.json_path.read_text(encoding="utf-8"))
        targets = data.get("targets", [])
        points = []
        names = []
        for t in targets:
            if not all(k in t for k in ("x", "y", "z")):
                continue
            p = Point(x=float(t["x"]), y=float(t["y"]), z=float(t["z"]))
            points.append(p)
            names.append(str(t.get("name", "")))
        return points, names

    def build_markers(self):
        points, names = self.load_points()
        n = len(points)
        if n == 0:
            self.get_logger().warn("targets 为空：JSON 里没找到可用的 (x,y,z) 点。")
            return MarkerArray()

        rgba_list = sample_rainbow_rgba(n)
        colors = [ColorRGBA(r=r, g=g, b=b, a=a) for (r, g, b, a) in rgba_list]

        now = self.get_clock().now().to_msg()

        sphere_scale = float(self.get_parameter("sphere_scale").value)
        line_width = float(self.get_parameter("line_width").value)
        show_labels = bool(self.get_parameter("show_labels").value)
        label_z_offset = float(self.get_parameter("label_z_offset").value)

        ma = MarkerArray()

        # 1) 点：SPHERE_LIST（每个点一个颜色）
        spheres = Marker()
        spheres.header.frame_id = self.frame_id
        spheres.header.stamp = now
        spheres.ns = "waypoints"
        spheres.id = 0
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        spheres.pose.orientation.w = 1.0
        spheres.scale.x = sphere_scale
        spheres.scale.y = sphere_scale
        spheres.scale.z = sphere_scale
        spheres.points = points
        spheres.colors = colors
        ma.markers.append(spheres)

        # 2) 线：LINE_STRIP（每个顶点一个颜色，RViz 会沿线插值）
        line = Marker()
        line.header.frame_id = self.frame_id
        line.header.stamp = now
        line.ns = "waypoints"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = line_width
        line.points = points
        line.colors = colors
        ma.markers.append(line)

        # 3) 可选：文本标签（p1/p2/...）
        if show_labels:
            for i, (p, name) in enumerate(zip(points, names)):
                if not name:
                    name = f"p{i+1}"
                txt = Marker()
                txt.header.frame_id = self.frame_id
                txt.header.stamp = now
                txt.ns = "waypoint_labels"
                txt.id = 100 + i
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = p.x
                txt.pose.position.y = p.y
                txt.pose.position.z = p.z + label_z_offset
                txt.pose.orientation.w = 1.0
                txt.scale.z = 0.03      # 字高（米）
                txt.text = name
                txt.color.r = 1.0
                txt.color.g = 1.0
                txt.color.b = 1.0
                txt.color.a = 1.0
                ma.markers.append(txt)

        return ma

    def on_timer(self):
        try:
            mtime = self.json_path.stat().st_mtime
        except FileNotFoundError:
            self.get_logger().error(f"找不到文件：{self.json_path}")
            return

        if self._cached is None or self._last_mtime != mtime:
            try:
                self._cached = self.build_markers()
                self._last_mtime = mtime
                self.get_logger().info(
                    f"已加载 targets，可视化 marker 数量：{len(self._cached.markers)}"
                )
            except Exception as e:
                self.get_logger().error(f"解析/构建 marker 失败：{e}")
                return

        # 更新时间戳，让 RViz 显示更稳定
        now = self.get_clock().now().to_msg()
        for mk in self._cached.markers:
            mk.header.stamp = now

        self.pub.publish(self._cached)


def main():
    rclpy.init()
    node = WaypointViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
