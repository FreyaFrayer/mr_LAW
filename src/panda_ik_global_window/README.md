# panda_ik_global_window

MoveIt2 (MoveItPy) + Franka Panda IK sampling benchmark.

This package samples multiple IK solutions for a sequence of reachable Cartesian target points, then compares **three** IK-selection strategies:

1. **greedy**: at each segment, pick the IK solution that makes the **current** segment time minimal.
2. **global**: (conceptually) enumerate **all full paths** (乘法原理) and pick the path with the **minimum total time** for `p0->p1->...->pn`.
   - Implementation uses a layered-DAG shortest-path DP (enumeration-equivalent) and **only returns the full path** result (no `0-1`, `0-1-2`, ... prefix outputs).
3. **window**: receding-horizon lookahead; within a window of future points, enumerate (equivalently via DP) all paths inside the window, then **commit only the first** decision.

`summary.json` **does not compute optimization rates** (per requirement).

## Build

```bash
cd ~/ws_moveit2
colcon build --packages-select panda_ik_global_window
source install/setup.bash
```

## Run (recommended)

```bash
ros2 launch panda_ik_global_window ik_benchmark.launch.py num_points:=3 seed:=7 window_size:=3
```

Data will be saved under `./data/<timestamp>/`.

## Key outputs

- `targets.json`: start `p0` joint positions + sampled Cartesian target points `p1..pN`
- `p1.json`..`pN.json`: IK solutions for each target point
- `summary.json`: unified report (greedy / global / window)
