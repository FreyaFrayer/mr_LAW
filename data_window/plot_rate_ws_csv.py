import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _infer_title_suffix(csv_path: Path) -> str:
    """
    Infer a nice suffix for plot titles from filenames like:
        np16_ws3.csv   ->  "np=16, ws=3"
    Fallback:
        use filename stem
    """
    m = re.search(r"np(\d+)_ws(\d+)", csv_path.stem)
    if m:
        np_v = int(m.group(1))
        ws_v = int(m.group(2))
        return f"np={np_v}, ws={ws_v}"
    return csv_path.stem


def _default_prefix(csv_path: Path) -> str:
    """
    Use the file name without *all* suffixes as prefix.

    Examples:
        np16_ws3.csv      -> np16_ws3
        np16_ws3.csv.txt  -> np16_ws3
    """
    name = csv_path.name
    # Remove suffixes from the end (right to left)
    for suf in reversed(csv_path.suffixes):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name or csv_path.stem


def plot_rate_ws_csv(csv_path: str, out_dir: str = "figs", prefix: Optional[str] = None):
    """
    Visualize rate CSV in the same style as plot_rate_csv.py, but tailored for files named
    like `np16_ws3.csv`.

    Expected CSV format (example):
        rate,7,8,9,10
        r1, ...
        r2, ...
        ...

    - First column must be 'rate' (r1, r2, ...)
    - Other columns are different seeds (or configs) and should be numeric-sortable.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = _default_prefix(csv_path)

    # --- load ---
    df = pd.read_csv(csv_path)
    # Be tolerant of accidental spaces in header
    df.columns = [str(c).strip() for c in df.columns]

    if "rate" not in df.columns:
        raise ValueError("CSV must have a column named 'rate' as the first column (r1, r2, ...).")

    df = df.set_index("rate")
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # columns are seeds (strings); try to keep them ordered by numeric value
    try:
        col_order = sorted(df.columns, key=lambda x: int(str(x)))
        df = df[col_order]
    except Exception:
        pass

    values = df.to_numpy()
    title_suffix = _infer_title_suffix(csv_path)

    # =========================
    # 1) Heatmap
    # =========================
    fig = plt.figure(figsize=(9, max(3.2, 0.55 * len(df))), dpi=160)
    ax = fig.add_subplot(111)

    im = ax.imshow(values, aspect="auto")
    ax.set_title("Optimization rate heatmap ({})".format(title_suffix))
    ax.set_xlabel("seed")
    ax.set_ylabel("run (r_i)")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns])
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index])

    # annotate each cell
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, "{:.3f}".format(values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = out_dir / "{}_heatmap.png".format(prefix)
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 2) Line plot (per seed)
    # =========================
    x = np.arange(len(df.index))
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = fig.add_subplot(111)

    for seed in df.columns:
        ax.plot(x, df[seed].to_numpy(), marker="o", label="seed {}".format(seed))

    ax.set_title("Optimization rate across runs (per seed) ({})".format(title_suffix))
    ax.set_xlabel("run (r_i)")
    ax.set_ylabel("optimization rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in df.index])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(df.columns)), fontsize=8)
    fig.tight_layout()

    line_path = out_dir / "{}_lines.png".format(prefix)
    fig.savefig(line_path, bbox_inches="tight")
    plt.close(fig)

    print("[OK] saved:\n  {}\n  {}".format(heatmap_path, line_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to the ws rate csv (e.g., np16_ws3.csv)")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures")
    parser.add_argument(
        "--prefix",
        default=None,
        help="filename prefix for saved figures; default: use the csv filename (without suffixes)",
    )
    args = parser.parse_args()

    plot_rate_ws_csv(args.csv, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()

