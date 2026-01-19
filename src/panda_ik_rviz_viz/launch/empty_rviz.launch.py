from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            # 不传 -d，就用 RViz 默认空界面（一般只有 Grid）
            arguments=[],
        ),
    ])