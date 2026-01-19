# panda_ik_benchmark

A small MoveIt2 (MoveItPy) benchmark:
1) sample N reachable Cartesian points p1..pN for Panda
2) sample 200 IK solutions for each point
3) compare greedy path vs global-optimal path (DP, equivalent to exhaustive enumeration)

## Build

```bash
cd ~/ws_moveit2
colcon build --packages-select panda_ik_benchmark
source install/setup.bash
```

## Run (recommended)

```bash
ros2 launch panda_ik_benchmark ik_benchmark.launch.py num_points:=3 seed:=7
```

Data will be saved under `./data/<timestamp>/`.
