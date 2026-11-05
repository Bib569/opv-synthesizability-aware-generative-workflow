#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
04_device_properties.py  ->  device_properties_distribution_method1_ridge_gp.png
=============================================================================
Ridge+GP surrogate device-property distributions (4 histograms, a)-d)).

Required input CSV (in --data-dir):
    device_properties_distribution_method1_ridge_gp_data.csv
        cols: surrogate_PCE, surrogate_Jsc, surrogate_FF, surrogate_Voc

Usage:
    python 04_device_properties.py --data-dir data/figure_inputs
=============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, panel_label, need,
                           parse_common_args, resolve_dirs)


def fig_device_properties(data_dir, outdir):
    csv = need(data_dir, 'device_properties_distribution_method1_ridge_gp_data.csv')
    name = 'device_properties_distribution_method1_ridge_gp.png'
    if csv is None:
        print(f"  [SKIP] {name}: missing device_properties_distribution_method1_ridge_gp_data.csv")
        return False
    df = pd.read_csv(csv)
    cols_map = {
        'PCE (%)':                    ('surrogate_PCE', COLORS['pce'], 18.0),
        r'$J_{sc}$ (mA cm$^{-2}$)':    ('surrogate_Jsc', COLORS['jsc'], 25.0),
        'FF':                          ('surrogate_FF', COLORS['ff'], 0.8),
        r'$V_{oc}$ (V)':               ('surrogate_Voc', COLORS['voc'], 1.0),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')
    panel_labels = ['a)', 'b)', 'c)', 'd)']
    for ax, (label, (col, color, thresh)), plabel in zip(axes.ravel(), cols_map.items(), panel_labels):
        style_ax(ax, xlabel=label, ylabel='Frequency')
        panel_label(ax, plabel, outside=True)
        ax.texts[-1].set_position((-0.06, 1.06))
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=60, color=color, alpha=0.72, edgecolor='black', linewidth=0.4)
                ax.axvline(thresh, color=COLORS['threshold'], linestyle='--', linewidth=2.0,
                           label=f'Threshold = {thresh}')
                ax.axvline(data.mean(), color=COLORS['mean_line'], linestyle='-', linewidth=1.8,
                           label=f'Mean = {data.mean():.2f}')
                pct = (data >= thresh).sum() / len(data) * 100
                ax.annotate(f'{pct:.1f}% above threshold', xy=(0.97, 0.93),
                            xycoords='axes fraction', ha='right', va='top',
                            fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='gray', linewidth=0.8, alpha=0.9))
                ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Column N/A', transform=ax.transAxes, ha='center', va='center', fontsize=12)
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 4: Ridge+GP device-property distributions.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_device_properties(data_dir, outdir)


if __name__ == "__main__":
    main()
