#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
05_disjoint_cv_r2.py  ->  method1_disjoint_cv_r2.png
=============================================================================
Structure-disjoint (group-disjoint CV) held-out R^2 bar chart.

Required input CSV (in --data-dir):
    method1_disjoint_cv_report.csv
        cols: property, heldout_r2 [, split_kind, status]

Usage:
    python 05_disjoint_cv_r2.py --data-dir data/figure_inputs
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, style_ax, need, parse_common_args, resolve_dirs)


def fig_disjoint_cv(data_dir, outdir):
    csv = need(data_dir, 'method1_disjoint_cv_report.csv')
    name = 'method1_disjoint_cv_r2.png'
    if csv is None:
        print(f"  [SKIP] {name}: missing method1_disjoint_cv_report.csv")
        return False
    df = pd.read_csv(csv)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    if df.empty or 'property' not in df.columns or 'heldout_r2' not in df.columns:
        print(f"  [SKIP] {name}: report missing 'property'/'heldout_r2' (or no ok rows)")
        return False
    df['label'] = df['property'].astype(str) + ' / ' + df.get('split_kind', '').astype(str)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 6.5))
    fig.patch.set_facecolor('white')
    style_ax(ax, xlabel='', ylabel=r'Held-out $R^2$ (group-disjoint CV)')
    vals = pd.to_numeric(df['heldout_r2'], errors='coerce')
    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in vals.fillna(0)]
    ax.bar(np.arange(len(df)), vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df['label'], rotation=75, ha='right', fontsize=7)
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 5: structure-disjoint held-out R^2.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_disjoint_cv(data_dir, outdir)


if __name__ == "__main__":
    main()
