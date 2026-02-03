import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator


def _infer_num_points(summary: dict) -> int:
    # Prefer meta.num_points (matches provided summary format)
    meta = summary.get("meta", {})
    if "num_points" in meta:
        try:
            n = int(meta["num_points"])
            if n > 0:
                return n
        except Exception:
            pass

    # Fallback: infer from targets length (targets are p1..pN)
    targets = summary.get("targets", [])
    if isinstance(targets, list) and len(targets) > 0:
        return len(targets)

    # Fallback: infer from first ws cumulative_times_s length
    window = summary.get("window", {})
    results_by_ws = window.get("results_by_ws", {})
    if isinstance(results_by_ws, dict) and len(results_by_ws) > 0:
        first_key = next(iter(results_by_ws.keys()))
        first = results_by_ws.get(first_key, {})
        times = first.get("cumulative_times_s", [])
        if isinstance(times, list) and len(times) > 0:
            return len(times)

    raise ValueError(
        "Could not infer num_points from summary.json (meta.num_points / targets / cumulative_times_s)."
    )


def plot_cumulative_times_summary(
    summary_path: str,
    out_dir: str = "figs",
    prefix: str = "cumulative_times",
    dpi: int = 200,
    fig_w: float = 8.0,
    fig_h: float = 6.0,
    y_major_step: float = 0.5,
    y_minor_step: float = None,
    y_top: float = None,
):
    summary_path = Path(summary_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    window = summary.get("window", {})
    results_by_ws = window.get("results_by_ws", None)
    if not isinstance(results_by_ws, dict) or len(results_by_ws) == 0:
        raise ValueError("summary.json must contain window.results_by_ws with at least one ws entry.")

    num_points = _infer_num_points(summary)

    # ws keys are often strings; sort by numeric value
    ws_keys = sorted(results_by_ws.keys(), key=lambda x: int(str(x)))
    ws_vals = [int(str(k)) for k in ws_keys]
    ws_min = min(ws_vals)
    ws_max = max(ws_vals)  # expected == num_points, but using max() is safer

    # x axis: point index 0..num_points (0 is start at time 0)
    x = np.arange(num_points + 1)

    # colors for non-highlight curves
    other_colors = list(plt.cm.tab20.colors)
    other_i = 0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)

    # Collect (ws -> (total_time, color, alpha)) for bottom-right ranking text
    ws_total_color = []

    for ws_key in ws_keys:
        ws = int(str(ws_key))
        entry = results_by_ws.get(ws_key, {})
        cumulative = entry.get("cumulative_times_s", None)
        if cumulative is None:
            raise ValueError(f"ws={ws} is missing cumulative_times_s.")
        if not isinstance(cumulative, list):
            raise ValueError(f"ws={ws} cumulative_times_s must be a list.")
        if len(cumulative) != num_points:
            raise ValueError(
                f"ws={ws} cumulative_times_s length mismatch: "
                f"expected {num_points}, got {len(cumulative)}."
            )

        # 0 time at origin, then arrival times for p1..pN
        y = np.array([0.0] + [float(v) for v in cumulative], dtype=float)

        # Styling:
        # - ws=1: clear red
        # - ws=last (typically ws=num_points): clear blue
        # - others: slightly transparent + other colors
        if ws == ws_min:
            color = "red"
            alpha = 1.0
            lw = 0.6
            zorder = 1
        elif ws == ws_max:
            color = "yellow"
            alpha = 1.0
            lw = 1.7
            zorder = 2
        else:
            color = other_colors[other_i % len(other_colors)]
            other_i += 1
            alpha = 0.85
            lw = 0.6
            zorder = 3

        ax.plot(
            x,
            y,
            label=f"ws {ws}",
            color=color,
            alpha=alpha,
            linewidth=lw,
            zorder=zorder,
        )

        ws_total_color.append((ws, float(y[-1]), color, float(alpha)))

    ax.set_title("Cumulative arrival time (per window size)")
    ax.set_xlabel("path point index")
    ax.set_ylabel("cumulative time (s)")
    ax.set_xticks(x)
    ax.set_xlim(0, num_points)

    # y-axis range
    if y_top is None:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(bottom=0.0, top=float(y_top))

    # y-axis "resolution" (tick density):
    # - major ticks every y_major_step seconds
    # - minor ticks every y_minor_step seconds
    if y_major_step is not None:
        try:
            y_major_step = float(y_major_step)
            if y_major_step > 0:
                ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
        except Exception:
            pass

    if y_minor_step is not None:
        try:
            y_minor_step = float(y_minor_step)
            if y_minor_step > 0:
                ax.yaxis.set_minor_locator(MultipleLocator(y_minor_step))
                ax.grid(True, which="minor", alpha=0.15, linestyle=":")
        except Exception:
            pass

    ax.grid(True, which="major", alpha=0.3)

    # Legend: explain ws is short for window size
    leg = ax.legend(
        ncol=min(6, len(ws_keys)),
        fontsize=8,
        title="ws = window size",
        loc="upper left",
    )
    leg.get_title().set_fontsize(8)

    # =========================
    # Bottom-right: total time list (longest -> shortest)
    # =========================
    ws_total_color.sort(key=lambda t: (-t[1], t[0]))  # desc by total time, then ws

    # Text layout (axes coordinates)
    x_right = 0.985
    y_bottom = 0.02

    # If many ws (e.g. up to 15), shrink a bit to keep it readable.
    n_lines = len(ws_total_color)
    font_size = 8
    if n_lines >= 12:
        font_size = 7

    line_h = 0.032 if font_size == 8 else 0.028

    # Header above the list
    ax.text(
        x_right,
        y_bottom + line_h * n_lines,
        "total time (s)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=font_size,
        color="black",
        alpha=0.85,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=10,
    )

    # Lines: longest at top, shortest at bottom
    for i, (ws, total_t, color, alpha) in enumerate(ws_total_color):
        y_text = y_bottom + line_h * (n_lines - 1 - i)
        ax.text(
            x_right,
            y_text,
            f"ws {ws}:  {total_t:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=font_size,
            color=color,
            alpha=alpha,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=10,
        )

    fig.tight_layout()

    out_path = out_dir / f"{prefix}_lines.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved:\n  {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="path to the window summary json")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures")
    parser.add_argument("--prefix", default="cumulative_times", help="filename prefix for saved figure")
    parser.add_argument("--dpi", type=int, default=320, help="figure DPI (output image resolution)")
    parser.add_argument("--fig_w", type=float, default=9.0, help="figure width (inches)")
    parser.add_argument("--fig_h", type=float, default=7.0, help="figure height (inches)")
    parser.add_argument("--y_major_step", type=float, default=0.5, help="y major tick step (seconds)")
    parser.add_argument("--y_minor_step", type=float, default=None, help="y minor tick step (seconds)")
    parser.add_argument("--y_top", type=float, default=None, help="y-axis upper bound (seconds)")
    args = parser.parse_args()

    plot_cumulative_times_summary(
        args.summary,
        args.out_dir,
        args.prefix,
        dpi=args.dpi,
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        y_major_step=args.y_major_step,
        y_minor_step=args.y_minor_step,
        y_top=args.y_top,
    )


if __name__ == "__main__":
    main()
