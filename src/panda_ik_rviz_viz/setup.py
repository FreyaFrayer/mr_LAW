from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'panda_ik_rviz_viz'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cao',
    maintainer_email='1684924265@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "publish_ik_solutions = panda_ik_rviz_viz.publish_ik_solutions:main",
            "benchmark_ik_solutions = panda_ik_rviz_viz.benchmark_ik_solutions:main",
            "benchmark_two_stage_planning = panda_ik_rviz_viz.benchmark_two_stage_planning:main",
        ],
    },
)
