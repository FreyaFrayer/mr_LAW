# panda_ik_window

MoveIt2 (MoveItPy) + Franka Panda IK sampling benchmark (window-size sweep).

This package:

1. **Generates a dataset** (depends on `seed`):
   - Samples `num_points = n` reachable Cartesian targets `p1..pN` (with `p0` as the start state).
   - For each point `p_i`, samples exactly `num_solutions = m` IK solutions (stored in `p{i}.json`).

2. **Evaluates window policies** (depends only on `window_size = ws`):
   - For a single dataset, runs `ws = 1, 2, ..., n`.
   - `ws = 1` is equivalent to **greedy**.
   - `ws = n` is equivalent to **global optimal**.
   - The window logic is optimized so that when `remain <= ws`, it solves the remaining points **once** (full DP) and finishes.

`summary.json` records results **by `ws` only** (no separate greedy/global blocks).

## Build

```bash
cd ~/ws_moveit2
colcon build --packages-select panda_ik_window
source install/setup.bash
```

## Run (single experiment)

```bash
ros2 launch panda_ik_window ik_benchmark.launch.py num_points:=8 seed:=7
```

Data will be saved under:

```
<data_root>/<timestamp>/
```

(`data_root` defaults to `./data_three`.)

## Batch run

See `batch_ik_window.py` at the package root. It runs one dataset per seed and writes a CSV summary.

Typical usage:

```bash
python3 batch_ik_window.py --num-points 8 --seeds 7,8,9
```

## Key outputs

- `targets.json`: start `p0` joint positions + sampled Cartesian target points `p1..pN`
- `p1.json`..`pN.json`: IK solutions for each target point
- `summary.json`: unified report with a `window.results_by_ws` section (ws=1..n)
