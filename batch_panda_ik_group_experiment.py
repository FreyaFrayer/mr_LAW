#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch experiment controller for `panda_ik_benchmark`.

What it does
------------
For each (num_points, num_solutions) combination:
  - run 5 seeds (7..11)
  - parse greedy & DP summaries
  - compute optimization rate: (T_greedy - T_dp) / T_greedy
  - compute mean + variance across the 5 seeds
  - write a wide CSV table:
        rows = num_points
        cols = num_solutions
        cell = "mean;var"   (semicolon-separated, so CSV delimiter remains comma)

Notes
-----
1) Your current `run_benchmark.py` clamps num_points to <= 9:
       if n > 9: n = 9
   If you really want 2..20, you MUST patch that upper bound.

2) TOTG (`time_model:=totg`) makes DP O(n*K^2) very expensive because
   `segment_time_matrix_s()` uses nested loops and calls TOTG for each pair.

Usage
-----
  # Make sure you sourced your ROS2 workspace first, e.g.:
  #   source /opt/ros/<distro>/setup.bash
  #   source <your_ws>/install/setup.bash

  python3 batch_panda_ik_group_experiment.py

Optional:
  python3 batch_panda_ik_group_experiment.py --time-model totg --allow-clamp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OneRunParsed:
    success: bool
    run_dir: Optional[Path] = None
    greedy_time_s: Optional[float] = None
    dp_time_s: Optional[float] = None
    opt_rate: Optional[float] = None
    actual_num_points: Optional[int] = None
    error: str = ""


_RUN_DIR_RE = re.compile(r".*\[run\]\s+data\s+dir\s*=\s*(?P<path>.+)\s*$")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _parse_summary_total_time(path: Path) -> Tuple[Optional[float], Optional[int], str]:
    """Return (total_time_s, actual_num_points, error)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, None, f"failed to read/parse JSON: {e}"

    total_time = _safe_float(payload.get("total_time_s"))
    if total_time is None:
        return None, None, "missing/invalid total_time_s"

    targets = payload.get("targets", [])
    actual_n = None
    try:
        actual_n = int(len(targets))
    except Exception:
        actual_n = None

    return total_time, actual_n, ""


def run_one(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    num_solutions: int,
    seed: int,
    time_model: str,
    data_root: Path,
    workdir: Path,
    log_dir: Path,
    allow_clamp: bool,
) -> OneRunParsed:
    """Run one experiment by calling `ros2 launch ...` and parse its output summaries."""

    # Each run uses its own data_root (so even same-second timestamps won't collide)
    data_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ros2",
        "launch",
        pkg,
        launch_file,
        f"num_points:={int(num_points)}",
        f"num_solutions:={int(num_solutions)}",
        f"seed:={int(seed)}",
        f"time_model:={str(time_model)}",
        f"data_root:={str(data_root)}",
    ]

    log_path = log_dir / f"np{num_points:02d}_ns{num_solutions:03d}_seed{seed:02d}.log"

    run_dir: Optional[Path] = None

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()

                m = _RUN_DIR_RE.match(line)
                if m:
                    p = m.group("path").strip()
                    # The node prints the path as-is; we normalize to an absolute Path if possible.
                    # If it's relative, treat it as relative to workdir.
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = (workdir / pp).resolve()
                    run_dir = pp

            rc = proc.wait()

        if rc != 0:
            # Even in failure, sometimes run_dir exists; we keep it for debugging.
            return OneRunParsed(success=False, run_dir=run_dir, error=f"ros2 launch exited with code {rc}. See log: {log_path}")

        if run_dir is None:
            # Fallback: find newest timestamp directory under data_root
            cand = [p for p in data_root.iterdir() if p.is_dir()]
            if cand:
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                run_dir = cand[0]

        if run_dir is None:
            return OneRunParsed(success=False, error=f"Cannot determine run_dir from logs. See log: {log_path}")

        greedy_json = run_dir / "summary_tradition.json"
        dp_json = run_dir / "summary_enumeration.json"

        if not greedy_json.exists() or not dp_json.exists():
            return OneRunParsed(
                success=False,
                run_dir=run_dir,
                error=(
                    "Missing summary json(s). Expected: "
                    f"{greedy_json.name} and {dp_json.name}. See log: {log_path}"
                ),
            )

        greedy_t, greedy_n, err1 = _parse_summary_total_time(greedy_json)
        if greedy_t is None:
            return OneRunParsed(success=False, run_dir=run_dir, error=f"Greedy summary parse failed: {err1}")

        dp_t, dp_n, err2 = _parse_summary_total_time(dp_json)
        if dp_t is None:
            return OneRunParsed(success=False, run_dir=run_dir, error=f"DP summary parse failed: {err2}")

        # sanity: actual num_points
        actual_n = greedy_n
        if actual_n is None and dp_n is not None:
            actual_n = dp_n

        if actual_n is not None and int(actual_n) != int(num_points):
            msg = (
                f"Requested num_points={num_points}, but summaries contain {actual_n} targets. "
                "Your code likely clamps num_points (current run_benchmark.py clamps to <=9)."
            )
            if not allow_clamp:
                return OneRunParsed(
                    success=False,
                    run_dir=run_dir,
                    greedy_time_s=float(greedy_t),
                    dp_time_s=float(dp_t),
                    actual_num_points=int(actual_n),
                    error=msg,
                )

        if greedy_t <= 0.0:
            return OneRunParsed(success=False, run_dir=run_dir, greedy_time_s=float(greedy_t), dp_time_s=float(dp_t), error="greedy total_time_s <= 0")

        opt = (float(greedy_t) - float(dp_t)) / float(greedy_t)

        return OneRunParsed(
            success=True,
            run_dir=run_dir,
            greedy_time_s=float(greedy_t),
            dp_time_s=float(dp_t),
            opt_rate=float(opt),
            actual_num_points=int(actual_n) if actual_n is not None else None,
        )

    except KeyboardInterrupt:
        raise
    except FileNotFoundError as e:
        return OneRunParsed(success=False, error=f"Command not found: {e}. Is ROS2 sourced? cmd={cmd}")
    except Exception as e:
        return OneRunParsed(success=False, run_dir=run_dir, error=f"Unexpected error: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=str, default="panda_ik_benchmark", help="ROS2 package name")
    ap.add_argument("--launch", type=str, default="ik_benchmark.launch.py", help="Launch file")
    ap.add_argument("--time-model", type=str, default="totg", choices=["totg", "auto", "trapezoid"], help="time_model launch arg")

    ap.add_argument("--num-points-min", type=int, default=3)
    ap.add_argument("--num-points-max", type=int, default=3)
    ap.add_argument(
        "--num-solutions",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100],
        help="List of num_solutions values",
    )
    ap.add_argument("--seed-min", type=int, default=7)
    ap.add_argument("--seed-max", type=int, default=12)

    ap.add_argument(
        "--workspace",
        type=str,
        default=str(Path.cwd()),
        help="Working directory used for ros2 launch (affects relative data paths)",
    )
    ap.add_argument(
        "--data-base",
        type=str,
        default=str((Path.cwd() / "data" / "batch_runs").resolve()),
        help="Base directory to store all run outputs (each run uses its own data_root)",
    )
    ap.add_argument(
        "--csv-dir",
        type=str,
        default=str((Path.cwd() / "data" / "csv").resolve()),
        help="Directory to save the aggregated CSV",
    )
    ap.add_argument(
        "--allow-clamp",
        action="store_true",
        help="If set, accept runs where actual num_points != requested (e.g. code clamps to <=9).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the ros2 launch commands; do not execute.",
    )

    ap.add_argument(
        "--write-raw",
        action="store_true",
        help="Also write a per-seed long-form CSV (<timestamp>_raw_opt_rates.csv) for debugging.",
    )

    args = ap.parse_args()

    pkg = str(args.pkg)
    launch_file = str(args.launch)
    time_model = str(args.time_model)

    np_min = int(args.num_points_min)
    np_max = int(args.num_points_max)
    ns_list = [int(x) for x in list(args.num_solutions)]

    seed_min = int(args.seed_min)
    seed_max = int(args.seed_max)
    seeds = list(range(seed_min, seed_max + 1))

    if np_min > np_max:
        print("num-points-min > num-points-max", file=sys.stderr)
        return 2
    if seed_min > seed_max:
        print("seed-min > seed-max", file=sys.stderr)
        return 2

    workdir = Path(args.workspace).resolve()
    data_base = Path(args.data_base).resolve()
    csv_dir = Path(args.csv_dir).resolve()

    group_ts = _now_ts()

    # Each run will have its own data_root under this group directory
    group_data_base = data_base / group_ts
    group_log_dir = group_data_base / "logs"

    csv_dir.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / f"{group_ts}_group_results.csv"

    num_points_values = list(range(np_min, np_max + 1))

    # Collect optimization rates per (np, ns)
    rates: Dict[Tuple[int, int], List[float]] = {(npv, nsv): [] for npv in num_points_values for nsv in ns_list}

    total_runs = len(num_points_values) * len(ns_list) * len(seeds)
    run_counter = 0

    print(f"[group] timestamp          = {group_ts}")
    print(f"[group] workspace (cwd)     = {workdir}")
    print(f"[group] data_base           = {group_data_base}")
    print(f"[group] csv_out             = {out_csv}")
    print(f"[group] total runs planned  = {total_runs}")

    for npv in num_points_values:
        for nsv in ns_list:
            print(f"\n[combo] num_points={npv}, num_solutions={nsv}")
            combo_rates: List[float] = []

            for seed in seeds:
                run_counter += 1
                tag = f"np{npv:02d}_ns{nsv:03d}_seed{seed:02d}"

                # Make data_root unique per run (so timestamps cannot collide)
                run_data_root = group_data_base / tag

                print(f"  [run {run_counter:4d}/{total_runs}] seed={seed} ...")

                cmd_preview = (
                    f"ros2 launch {pkg} {launch_file} "
                    f"num_points:={npv} num_solutions:={nsv} seed:={seed} time_model:={time_model} data_root:={run_data_root}"
                )
                if args.dry_run:
                    print("    DRY_RUN:", cmd_preview)
                    continue

                parsed = run_one(
                    pkg=pkg,
                    launch_file=launch_file,
                    num_points=npv,
                    num_solutions=nsv,
                    seed=seed,
                    time_model=time_model,
                    data_root=run_data_root,
                    workdir=workdir,
                    log_dir=group_log_dir,
                    allow_clamp=bool(args.allow_clamp),
                )

                if not parsed.success:
                    print(f"    FAIL: {parsed.error}")
                    # keep placeholder NaN for this seed
                    combo_rates.append(float("nan"))
                    continue

                assert parsed.opt_rate is not None
                combo_rates.append(float(parsed.opt_rate))

                print(
                    f"    OK: greedy={parsed.greedy_time_s:.6f}s dp={parsed.dp_time_s:.6f}s "
                    f"opt_rate={parsed.opt_rate:.6f} run_dir={parsed.run_dir}"
                )

            # store
            rates[(npv, nsv)] = combo_rates

    if args.dry_run:
        print("\n[dry-run] Done. No CSV written.")
        return 0

    # Build the wide table: cell = "mean;var" (variance uses ddof=0).
    # Per your requirement, we only compute mean/var when we have *all* 5 seeds.
    def cell_str(vals: List[float]) -> str:
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size != len(seeds):
            return ""
        mean = float(np.mean(finite))
        var = float(np.var(finite, ddof=0))
        return f"{mean:.6f};{var:.6f}"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["num_points"] + [str(x) for x in ns_list]
        w.writerow(header)
        for npv in num_points_values:
            row = [str(npv)]
            for nsv in ns_list:
                row.append(cell_str(rates.get((npv, nsv), [])))
            w.writerow(row)

    print(f"\n[done] wrote: {out_csv}")
    print("Cell format is 'mean;var' (semicolon-separated).")

    if args.write_raw:
        raw_csv = csv_dir / f"{group_ts}_raw_opt_rates.csv"
        with open(raw_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["num_points", "num_solutions", "seed", "opt_rate"])  # opt_rate may be empty if failed
            for npv in num_points_values:
                for nsv in ns_list:
                    vals = rates.get((npv, nsv), [])
                    for i, seed in enumerate(seeds):
                        v = vals[i] if i < len(vals) else float("nan")
                        w.writerow([npv, nsv, seed, "" if not np.isfinite(v) else f"{float(v):.8f}"])

        print(f"[done] wrote: {raw_csv} (per-seed opt_rate)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
