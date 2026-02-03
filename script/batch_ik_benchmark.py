#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for panda_ik_benchmark (ROS2).

It will:
1) Loop num_points=5, num_solutions in [50,100,150,200,250]
2) For each group, run seeds 7..11
3) Parse each run's data_root/<timestamp>/summary.json
4) Collect optimization_rate r1..rN into CSV with:
   - rows: r1..rN
   - columns: seed (7..11)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated ints, e.g. '50,100,150'."""
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_str() -> str:
    """Human-readable local time with timezone offset (e.g. 2026-01-26 14:03:21+08:00)."""
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def _find_latest_run_dir(data_root: Path) -> Path:
    """
    Find latest <data_root>/<timestamp>/ directory.

    Benchmark timestamps use: YYYYMMDD_HHMMSS
    Lexicographic order works for 'latest'.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run dirs under: {data_root}")
    return sorted(dirs, key=lambda d: d.name)[-1]


def _wait_for_summary_json(data_root: Path, timeout_s: float = 60.0) -> Path:
    """Wait until <data_root>/<latest_ts>/summary.json exists."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            run_dir = _find_latest_run_dir(data_root)
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                return summary_path
        except Exception as e:
            last_err = e
        time.sleep(0.2)
    raise TimeoutError(
        f"Timeout waiting for summary.json under {data_root}. Last error: {last_err}"
    )


def _extract_rates(summary_json: Path, num_points_expected: int) -> List[float]:
    """Return [r1..rN] from one run summary.json."""
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vals = data.get("optimization_rate", {}).get("values", [])
    if not isinstance(vals, list) or not vals:
        raise ValueError(f"No optimization_rate.values in {summary_json}")

    def _point_index(v: Dict) -> int:
        p = str(v.get("point", ""))
        if p.startswith("p"):
            try:
                return int(p[1:])
            except Exception:
                return 10**9
        return 10**9

    vals_sorted = sorted(vals, key=_point_index)
    rates = [float(v.get("r_i", math.nan)) for v in vals_sorted]

    # Normalize length to num_points_expected
    if len(rates) < num_points_expected:
        rates = rates + [math.nan] * (num_points_expected - len(rates))
    elif len(rates) > num_points_expected:
        rates = rates[:num_points_expected]

    return rates


def _run_ros2_launch(
    *,
    pkg: str,
    launch_file: str,
    num_points: int,
    num_solutions: int,
    seed: int,
    time_model: str,
    data_root: Path,
    log_file: Path | None,
    extra_launch_args: Sequence[str],
) -> int:
    """Run one ros2 launch call. Return process return code."""
    cmd = [
        "ros2",
        "launch",
        pkg,
        launch_file,
        f"num_points:={num_points}",
        f"num_solutions:={num_solutions}",
        f"seed:={seed}",
        f"time_model:={time_model}",
        f"data_root:={str(data_root)}",
    ]
    cmd.extend(extra_launch_args)

    print("\n[batch] running:")
    print(" ".join(cmd))

    if log_file is None:
        proc = subprocess.run(cmd)
        return int(proc.returncode)

    _ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _write_rates_csv(
    *,
    out_csv: Path,
    seeds: Sequence[int],
    rates_by_seed: Dict[int, List[float]],
    num_points: int,
) -> None:
    """
    Write CSV with:
      - rows: r1..rN
      - columns: seed values
    """
    _ensure_dir(out_csv.parent)

    header = ["rate"] + [str(s) for s in seeds]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(1, num_points + 1):
            row_name = f"r{i}"
            row: List[str] = [row_name]
            for s in seeds:
                vals = rates_by_seed.get(s)
                v = math.nan
                if vals is not None and (i - 1) < len(vals):
                    v = float(vals[i - 1])
                row.append(str(v))
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch experiments for panda_ik_benchmark: collect optimization rates into CSV.",
    )
    ap.add_argument("--pkg", default="panda_ik_benchmark", help="ROS2 package name.")
    ap.add_argument("--launch", default="ik_benchmark.launch.py", help="Launch file name.")
    ap.add_argument("--num-points", type=int, default=25, help="Fixed num_points (default: 5).")
    ap.add_argument(
        "--num-solutions-list",
        type=str,
        default="100", # 50,100,150,200,250,300,400,500,700,900
        help="Comma-separated list of num_solutions values.",
    )
    ap.add_argument("--seed-start", type=int, default=7, help="Seed start (inclusive).")
    ap.add_argument("--seed-end", type=int, default=11, help="Seed end (inclusive).")
    ap.add_argument("--time-model", default="totg", choices=["auto", "totg", "trapezoid"], help="Time model.")
    ap.add_argument("--base-data-root", type=str, default="data/batch_data", help="Base dir for raw outputs.")
    ap.add_argument("--csv-dir", type=str, default="data/batch_csv", help="Dir for aggregated CSV outputs.")
    ap.add_argument("--log-dir", type=str, default="data/batch_logs", help="Dir for per-run ros2 logs.")
    ap.add_argument("--no-logs", action="store_true", help="Print ros2 output to console instead of logs.")
    ap.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra launch args appended verbatim (e.g. 'totg_vel_scale:=0.5').",
    )

    args = ap.parse_args()

    num_points = int(args.num_points)
    num_solutions_list = _parse_int_list(args.num_solutions_list)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    time_model = str(args.time_model)

    base_data_root = Path(args.base_data_root)
    csv_dir = Path(args.csv_dir)
    log_dir = Path(args.log_dir)
    extra_args = [a for a in args.extra.split() if a.strip()]

    # progress counter across ALL runs (ns x seed)
    total_rounds = len(num_solutions_list) * len(seeds)
    round_idx = 0

    print("[batch] settings")
    print(f"  num_points         = {num_points}")
    print(f"  num_solutions_list  = {num_solutions_list}")
    print(f"  seeds              = {seeds}")
    print(f"  time_model         = {time_model}")
    print(f"  base_data_root      = {base_data_root}")
    print(f"  csv_dir             = {csv_dir}")
    print(f"  log_dir             = {log_dir} (enabled={not args.no_logs})")
    if extra_args:
        print(f"  extra launch args   = {extra_args}")

    # Main loops
    for ns in num_solutions_list:
        rates_by_seed: Dict[int, List[float]] = {}

        print("\n" + "=" * 80)
        print(f"[batch] group: num_points={num_points}, num_solutions={ns}, time_model={time_model}")
        print("=" * 80)

        for seed in seeds:
            round_idx += 1
            print(
                f"\n[batch] 当前轮次/总轮次: {round_idx}/{total_rounds} | "
                f"ns={ns}, seed={seed} | START @ {_now_str()}"
            )
            # Unique data_root for each (ns, seed); benchmark still creates inner <timestamp> dir.
            run_data_root = base_data_root / f"np{num_points}" / f"ns{ns}" / f"seed{seed}"

            log_file = None
            if not args.no_logs:
                log_file = log_dir / f"np{num_points}_ns{ns}_seed{seed}.log"

            rc = _run_ros2_launch(
                pkg=str(args.pkg),
                launch_file=str(args.launch),
                num_points=num_points,
                num_solutions=ns,
                seed=seed,
                time_model=time_model,
                data_root=run_data_root,
                log_file=log_file,
                extra_launch_args=extra_args,
            )

            if rc != 0:
                print(f"[batch][{_now_str()}][ERROR] ros2 launch failed (returncode={rc}) for ns={ns}, seed={seed}")
                rates_by_seed[seed] = [math.nan] * num_points
                continue

            try:
                summary_path = _wait_for_summary_json(run_data_root, timeout_s=60.0)
                rates = _extract_rates(summary_path, num_points_expected=num_points)
                rates_by_seed[seed] = rates

                end_ts = _now_str()
                print(f"[batch][{end_ts}] summary: {summary_path}")
                print(f"[batch][{end_ts}] r[1..{num_points}] = {rates}")
            except Exception as e:
                print(f"[batch][{_now_str()}][ERROR] Failed to parse summary for ns={ns}, seed={seed}: {e}")
                rates_by_seed[seed] = [math.nan] * num_points

        # One CSV per num_solutions group
        out_csv = csv_dir / f"np{num_points}_ns{ns}_time_{time_model}.csv"
        _write_rates_csv(
            out_csv=out_csv,
            seeds=seeds,
            rates_by_seed=rates_by_seed,
            num_points=num_points,
        )
        print(f"\n[batch] CSV written: {out_csv}\n")

    print("[batch] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
