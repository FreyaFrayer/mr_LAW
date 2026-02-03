from setuptools import find_packages, setup

package_name = "panda_ik_window_viz"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/ik_window_compare_rviz.launch.py",
                "launch/ik_window_compare_overlay_tint_rviz.launch.py",
                "launch/ik_window_compare_overlay_tint_control_rviz.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/rviz",
            [
                "rviz/ik_window_compare.rviz",
                "rviz/ik_window_compare_overlay_tint.rviz",
                "rviz/ik_window_compare_overlay_tint_control.rviz",
            ],
        ),
        ("share/" + package_name, ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="RViz2 visualization for panda_ik_window (greedy vs window playback).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ik_window_compare_player = panda_ik_window_viz.ik_window_compare_player:main",
            "urdf_mesh_tint_markers = panda_ik_window_viz.urdf_mesh_tint_markers:main",
        ],
    },
)
