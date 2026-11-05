#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
09_tier2_building_block_recurrence.py  ->  tier2_building_block_recurrence.png
=============================================================================
Recurrence of the ten most frequent Tier-2 donor / acceptor building blocks.

This figure uses only value_counts on SMILES strings, so it does NOT require
RDKit (unlike figures 8, 10, and 12).

Required input CSV (in --data-dir):
    high_performance_pairs.csv    cols: donor_smiles, acceptor_smiles

Usage:
    python 09_tier2_building_block_recurrence.py --data-dir data/figure_inputs
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, need, parse_common_args, resolve_dirs)


def fig_building_block_recurrence(data_dir, outdir):
    name = 'tier2_building_block_recurrence.png'
    hp_csv = need(data_dir, 'high_performance_pairs.csv')
    if hp_csv is None:
        print(f"  [SKIP] {name}: missing high_performance_pairs.csv")
        return False
    hp = pd.read_csv(hp_csv)

    donor_counts = hp['donor_smiles'].value_counts().head(10)
    acc_counts = hp['acceptor_smiles'].value_counts().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    for ax, counts, color in [
        (axes[0], donor_counts, COLORS['donor']),
        (axes[1], acc_counts, COLORS['acceptor']),
    ]:
        style_ax(ax, xlabel='', ylabel='Number of Tier-2 pairs containing this molecule')
        x = np.arange(len(counts))
        ax.bar(x, counts.values, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for i, c in enumerate(counts.values):
            ax.annotate(str(c), xy=(i, c), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(counts))])
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 9: Tier-2 building-block recurrence.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_building_block_recurrence(data_dir, outdir)


if __name__ == "__main__":
    main()
