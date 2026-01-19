#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取类似如下格式的 CSV 并画曲线图：

表头:  num_points,50,100,150,...
行首:  2,0.048344;0.002746,0.048460;0.001311,...
单元格: "mean;var"（分号左边均值，右边方差）

功能：
- x 轴：表头第二列开始的列名（如 50,100,...）
- 每一行：一条曲线，label 来自第一列（如 2,3,4...）
- what=mean      画均值
- what=var       画方差
- what=mean_std  画均值 + 标准差阴影带（std = sqrt(var)）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_cell(cell: str):
    """把 'mean;var' -> (mean, var). 解析失败返回 (nan, nan)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan, np.nan

    s = str(cell).strip()
    if not s:
        return np.nan, np.nan

    if ";" not in s:
        # 兼容：只有均值没有方差的情况
        try:
            return float(s), np.nan
        except ValueError:
            return np.nan, np.nan

    mean_str, var_str = s.split(";", 1)
    try:
        mean = float(mean_str.strip())
    except ValueError:
        mean = np.nan

    try:
        var = float(var_str.strip())
    except ValueError:
        var = np.nan

    return mean, var


def main():
    parser = argparse.ArgumentParser(
        description="Plot curves from CSV cells formatted as 'mean;var'."
    )
    parser.add_argument("-i", "--input", required=True, help="输入 CSV 路径")
    parser.add_argument(
        "-o",
        "--out",
        default="plot.png",
        help="输出图片路径（如 plot.png / plot.pdf）",
    )
    parser.add_argument(
        "--what",
        choices=["mean", "var", "mean_std"],
        default="mean_std",
        help="画什么：mean / var / mean_std(均值+std阴影)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="图标题（可选）",
    )
    parser.add_argument(
        "--xlabel",
        default="num_points",
        help="x轴标签（默认 num_points）",
    )
    parser.add_argument(
        "--ylabel",
        default="value",
        help="y轴标签（默认 value；你也可以改成 error/accuracy 等）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示（有些服务器环境可能不支持）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    # 以字符串读入，避免 "0.01;0.02" 被误解析
    df = pd.read_csv(input_path, dtype=str)

    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要两列：第一列为曲线ID，其余列为 x 轴各点。")

    series_col = df.columns[0]
    x_cols = list(df.columns[1:])

    # x 轴来自列名（如 "50","100"...），尽量转成 float
    try:
        x = np.array([float(c) for c in x_cols], dtype=float)
    except ValueError as e:
        raise ValueError(
            f"列名（除第一列外）需要是数字（用于 x 轴），但解析失败：{x_cols}"
        ) from e

    # 准备画图
    plt.figure()
    ax = plt.gca()

    for _, row in df.iterrows():
        curve_id = str(row[series_col]).strip()
        if curve_id == "" or curve_id.lower() == "nan":
            continue

        means = []
        vars_ = []
        for c in x_cols:
            m, v = parse_cell(row[c])
            means.append(m)
            vars_.append(v)

        means = np.array(means, dtype=float)
        vars_ = np.array(vars_, dtype=float)

        label = f"{series_col}={curve_id}"

        if args.what == "mean":
            ax.plot(x, means, marker="o", label=label)

        elif args.what == "var":
            ax.plot(x, vars_, marker="o", label=label)

        else:  # mean_std
            line = ax.plot(x, means, marker="o", label=label)[0]
            # 用 std 阴影带：mean ± sqrt(var)
            std = np.sqrt(vars_)
            # 只在 std 为有限值的地方画阴影
            valid = np.isfinite(means) & np.isfinite(std)

            if valid.any():
                color = line.get_color()  # 跟曲线同色（不手动指定颜色）
                ax.fill_between(
                    x[valid],
                    (means - std)[valid],
                    (means + std)[valid],
                    alpha=0.2,
                    color=color,
                    linewidth=0,
                )

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    if args.title:
        ax.set_title(args.title)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(ncol=2, fontsize=9)

    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=200)
    print(f"已保存图片: {out_path.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

