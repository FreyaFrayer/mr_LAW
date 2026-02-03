import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_np_ws(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract (np, ws) from filename like:
        np16_ws3.csv
        np16_ws3.csv.txt
    Returns (None, None) if not matched.
    """
    m = re.search(r"np(\d+)_ws(\d+)", path.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _load_rate_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a rate CSV with format:
        rate,7,8,9,10
        r1, ...
        r2, ...
    Returns a DataFrame indexed by 'rate' (r1, r2, ...) and numeric columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "rate" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'rate' column.")
    df = df.set_index("rate")
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def collect_seed_series(
    batch_dir: Path,
    np_value: int,
    seed_value: int,
) -> Tuple[pd.DataFrame, List[int], List[Path]]:
    """
    From batch_dir, find files matching given np and any ws, then extract the given seed column.
    Return:
      - df_ws: index=runs, columns=ws, values=rate for that seed (NaN if missing)
      - ws_list: ws values included (sorted)
      - paths: file paths corresponding to ws_list (same order)
    """
    # Accept both *.csv and *.csv.txt
    candidates: List[Path] = []
    for ext in ("*.csv", "*.csv.txt", "*.CSV", "*.CSV.TXT"):
        candidates.extend(batch_dir.glob(ext))

    matched: List[Tuple[int, Path]] = []
    for p in candidates:
        np_i, ws_i = _extract_np_ws(p)
        if np_i is None or ws_i is None:
            continue
        if np_i == np_value:
            matched.append((ws_i, p))

    matched.sort(key=lambda t: t[0])
    if not matched:
        raise FileNotFoundError(
            f"No files found in '{batch_dir}' matching pattern 'np{np_value}_ws*.csv' (or .csv.txt)."
        )

    ws_list: List[int] = []
    paths: List[Path] = []
    series_by_ws: Dict[int, pd.Series] = {}

    seed_col = str(seed_value)

    for ws_i, p in matched:
        df = _load_rate_csv(p)
        if seed_col not in df.columns:
            print(f"[WARN] {p.name}: seed column '{seed_col}' not found. Skipped.")
            continue

        s = df[seed_col].copy()
        s = pd.to_numeric(s, errors="coerce")
        series_by_ws[ws_i] = s
        ws_list.append(ws_i)
        paths.append(p)

    if not series_by_ws:
        raise ValueError(
            f"Found matching files for np={np_value}, but none contained seed column '{seed_col}'."
        )

    # Align on union of run indices (r1, r2, ...)
    all_index = sorted(
        set().union(*[set(s.index) for s in series_by_ws.values()]),
        key=lambda x: int(re.sub(r"[^0-9]", "", str(x)) or "0"),
    )

    df_ws = pd.DataFrame(index=all_index)
    for ws_i in ws_list:
        df_ws[str(ws_i)] = series_by_ws[ws_i].reindex(all_index)

    return df_ws, ws_list, paths


def plot_np_seed_across_ws(
    batch_dir: str,
    np_value: int,
    seed_value: int,
    out_dir: str = "figs",
    prefix: Optional[str] = None,
):
    batch_dir = Path(batch_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = f"np{np_value}_seed{seed_value}"

    df_ws, ws_list, paths = collect_seed_series(batch_dir, np_value, seed_value)

    # Heatmap uses zeros for NaN so text annotation is readable
    values_heat = df_ws.fillna(0.0).to_numpy()

    # =========================
    # 1) Heatmap: runs x ws
    # =========================
    fig = plt.figure(figsize=(9, max(3.2, 0.55 * len(df_ws))), dpi=160)
    ax = fig.add_subplot(111)

    im = ax.imshow(values_heat, aspect="auto")
    ax.set_title(f"Optimization rate heatmap (np={np_value}, seed={seed_value})")
    ax.set_xlabel("ws")
    ax.set_ylabel("run (r_i)")

    ax.set_xticks(np.arange(df_ws.shape[1]))
    ax.set_xticklabels([str(c) for c in df_ws.columns])
    ax.set_yticks(np.arange(df_ws.shape[0]))
    ax.set_yticklabels([str(i) for i in df_ws.index])

    for i in range(values_heat.shape[0]):
        for j in range(values_heat.shape[1]):
            ax.text(j, i, f"{values_heat[i, j]:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_path = out_dir / f"{prefix}_ws_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    # =========================
    # 2) Line plot: one line per ws
    # =========================
    x = np.arange(len(df_ws.index))
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = fig.add_subplot(111)

    for ws in df_ws.columns:
        ax.plot(x, df_ws[ws].to_numpy(), marker="o", label=f"ws {ws}")

    ax.set_title(f"Optimization rate across runs (np={np_value}, seed={seed_value})")
    ax.set_xlabel("run (r_i)")
    ax.set_ylabel("optimization rate")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in df_ws.index])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(4, len(df_ws.columns)), fontsize=8)
    fig.tight_layout()
    line_path = out_dir / f"{prefix}_ws_lines.png"
    fig.savefig(line_path, bbox_inches="tight")
    plt.close(fig)

    print("[OK] files used:")
    for p in paths:
        print(f"  - {p}")
    print("[OK] saved:")
    print(f"  {heatmap_path}")
    print(f"  {line_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="batch_csv", help="directory containing np*_ws*.csv files")
    parser.add_argument("--np", type=int, required=True, help="np value, e.g. 16")
    parser.add_argument("--seed", type=int, required=True, help="seed value (CSV column), e.g. 7")
    parser.add_argument("--out_dir", default="figs", help="output directory for figures")
    parser.add_argument(
        "--prefix",
        default=None,
        help="filename prefix for saved figures; default: np{np}_seed{seed}",
    )
    args = parser.parse_args()

    plot_np_seed_across_ws(args.dir, args.np, args.seed, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()

