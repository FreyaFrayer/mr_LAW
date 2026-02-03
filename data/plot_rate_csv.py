import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rate_csv(csv_path: str, out_dir: str = "figs", prefix: str = "rate"):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    df = pd.read_csv(csv_path)
    if "rate" not in df.columns:
        raise ValueError("CSV must have a column named 'rate' as the first column (r1, r2, ...).")

    df = df.set_index("rate")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # columns are seeds (strings); try to keep them ordered by numeric value
    try:
        seed_order = sorted(df.columns, key=lambda x: int(str(x)))
        df = df[seed_order]
    except Exception:
        pass

    values = df.to_numpy()

    # =========================
    # 1) Heatmap
    # =========================
    fig = plt.figure(figsize=(9, max(3.2, 0.55 * len(df))), dpi=160)
    ax = fig.add_subplot(111)

    im = ax.imshow(values, aspect="auto")
    ax.set_title("Optimization rate heatmap")
    ax.set_xlabel("seed")
    ax.set_ylabel("run (r_i)")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns])
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index])

    # annotate each cell
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = out_dir / f"{prefix}_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 2) Line plot (per seed)
    # =========================
    x = np.arange(len(df.index))
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = fig.add_subplot(111)

    for seed in df.columns:
        ax.plot(x, df[seed].to_numpy(), marker="o", label=f"seed {seed}")

    ax.set_title("Optimization rate across runs (per seed)")
    ax.set_xlabel("run (r_i)")
    ax.set_ylabel("optimization rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in df.index])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(df.columns)), fontsize=8)
    fig.tight_layout()

    line_path = out_dir / f"{prefix}_lines.png"
    fig.savefig(line_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved:\n  {heatmap_path}\n  {line_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to the rate csv")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures")
    parser.add_argument("--prefix", default="rate", help="filename prefix for saved figures")
    args = parser.parse_args()

    plot_rate_csv(args.csv, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()

