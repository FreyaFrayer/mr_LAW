#!/usr/bin/env python3
"""
Plot segment-time improvement rate (global/window vs greedy) from a summary.json.

The summary.json is expected to follow the "panda_ik_global_window_summary" format, e.g.:

  summary["greedy"]["segment_times_s"]
  summary["global"]["segment_times_s"]
  summary["window"]["segment_times_s"]

"Improvement rate" is defined as:
    (greedy_time - method_time) / greedy_time

So:
  - positive value  => method is faster than greedy (time reduced)
  - negative value  => method is slower than greedy

Outputs:
  1) <out_dir>/<prefix>_segment_rate.csv
  2) <out_dir>/<prefix>_segment_rate.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_improvement_rate(method: np.ndarray, greedy: np.ndarray) -> np.ndarray:
    """Compute (greedy - method) / greedy, with division-by-zero guard."""
    method = np.asarray(method, dtype=float)
    greedy = np.asarray(greedy, dtype=float)

    if method.shape != greedy.shape:
        raise ValueError(f"Shape mismatch: method {method.shape} vs greedy {greedy.shape}")

    rate = np.zeros_like(greedy, dtype=float)
    mask = greedy != 0.0
    rate[mask] = (greedy[mask] - method[mask]) / greedy[mask]
    # keep zeros where greedy==0
    return rate


def _infer_point_labels(summary: Dict[str, Any], n: int) -> List[str]:
    """
    Infer x-axis labels (path points) for each segment.

    Preference order:
      1) greedy.final_path.segments[*].to  (e.g. p1, p2, ...)
      2) targets[*].name
      3) fallback: ["seg1", "seg2", ...]
    """
    # 1) from greedy.final_path.segments
    greedy_segs = (
        summary.get("greedy", {})
        .get("final_path", {})
        .get("segments", [])
    )
    if isinstance(greedy_segs, list) and len(greedy_segs) == n:
        labels = []
        for i, seg in enumerate(greedy_segs):
            labels.append(str(seg.get("to", f"seg{i+1}")))
        return labels

    # 2) from targets
    targets = summary.get("targets", [])
    if isinstance(targets, list) and len(targets) == n:
        labels = []
        for i, t in enumerate(targets):
            labels.append(str(t.get("name", f"seg{i+1}")))
        return labels

    # 3) fallback
    return [f"seg{i+1}" for i in range(n)]


def plot_segment_rates(summary_json: str, out_dir: str = "figs", prefix: str = "seg_rate") -> None:
    summary_json = Path(summary_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    greedy_seg = np.asarray(summary["greedy"]["segment_times_s"], dtype=float)
    global_seg = np.asarray(summary["global"]["segment_times_s"], dtype=float)
    window_seg = np.asarray(summary["window"]["segment_times_s"], dtype=float)

    if not (len(greedy_seg) == len(global_seg) == len(window_seg)):
        raise ValueError(
            "segment_times_s length mismatch: "
            f"greedy={len(greedy_seg)}, global={len(global_seg)}, window={len(window_seg)}"
        )

    n = len(greedy_seg)
    labels = _infer_point_labels(summary, n)

    # --- rates ---
    rate_global = _safe_improvement_rate(global_seg, greedy_seg)
    rate_window = _safe_improvement_rate(window_seg, greedy_seg)

    # --- table (also useful for debugging) ---
    df = pd.DataFrame(
        {
            "greedy_time_s": greedy_seg,
            "global_time_s": global_seg,
            "window_time_s": window_seg,
            "global_impr_rate": rate_global,
            "window_impr_rate": rate_window,
            "global_impr_rate_pct": rate_global * 100.0,
            "window_impr_rate_pct": rate_window * 100.0,
        },
        index=pd.Index(labels, name="point"),
    )

    csv_path = out_dir / f"{prefix}_segment_rate.csv"
    df.to_csv(csv_path, float_format="%.10g")

    # --- plot ---
    x = np.arange(n)

    fig = plt.figure(figsize=(10, 4.5), dpi=160)
    ax = fig.add_subplot(111)

    ax.plot(x, df["global_impr_rate_pct"].to_numpy(), marker="o", label="global vs greedy")
    ax.plot(x, df["window_impr_rate_pct"].to_numpy(), marker="o", label="window vs greedy")
    ax.axhline(0.0, linewidth=1, alpha=0.35)

    # Title: include meta info if present
    meta = summary.get("meta", {})
    group = meta.get("group", "")
    window_size = meta.get("window_size", None)
    title_parts = ["Segment improvement rate vs greedy"]
    if group:
        title_parts.append(f"(group={group})")
    if window_size is not None:
        title_parts.append(f"(window_size={window_size})")
    ax.set_title(" ".join(title_parts))

    ax.set_xlabel("path point (segment end)")
    ax.set_ylabel("improvement rate (%)  = (greedy - method) / greedy * 100")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig_path = out_dir / f"{prefix}_segment_rate.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # --- print summary ---
    greedy_total = float(np.sum(greedy_seg))
    global_total = float(np.sum(global_seg))
    window_total = float(np.sum(window_seg))

    total_impr_global = (greedy_total - global_total) / greedy_total if greedy_total != 0 else 0.0
    total_impr_window = (greedy_total - window_total) / greedy_total if greedy_total != 0 else 0.0

    print("[OK] saved:")
    print(f"  {csv_path}")
    print(f"  {fig_path}")
    print("")
    print("[Totals]")
    print(f"  greedy  : {greedy_total:.6f} s")
    print(f"  global  : {global_total:.6f} s  (impr {total_impr_global*100:.3f}%)")
    print(f"  window  : {window_total:.6f} s  (impr {total_impr_window*100:.3f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="path to summary.json")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures/csv")
    parser.add_argument("--prefix", default="seg_rate", help="filename prefix for saved outputs")
    args = parser.parse_args()

    plot_segment_rates(args.json, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()

