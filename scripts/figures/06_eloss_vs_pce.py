#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
06_eloss_vs_pce.py  ->  eloss_vs_pce_corrected.png
=============================================================================
Energy-loss vs PCE for reference and generated pairs.

Required input CSVs (in --data-dir):
    eloss_vs_pce_training_data.csv            cols: E_loss, PCE_max_pct, excluded_unphysical
    eloss_vs_pce_generated_corrected_data.csv cols: E_loss_pred_physical, surrogate_PCE

Usage:
    python 06_eloss_vs_pce.py --data-dir data/figure_inputs
=============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, need, parse_common_args, resolve_dirs)


def fig_eloss_vs_pce(data_dir, outdir):
    train_csv = need(data_dir, 'eloss_vs_pce_training_data.csv')
    gen_csv = need(data_dir, 'eloss_vs_pce_generated_corrected_data.csv')
    name = 'eloss_vs_pce_corrected.png'
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    style_ax(ax, xlabel=r'$E_{\mathrm{loss}}$ (eV)', ylabel='PCE (%)')
    plotted = False
    if train_csv is not None:
        tdf = pd.read_csv(train_csv)
        e, p = tdf['E_loss'], tdf['PCE_max_pct']
        unphys = tdf['excluded_unphysical'].astype(bool)
        phys = ~unphys
        ax.scatter(e[phys], p[phys], s=30, color=COLORS['donor'], alpha=0.7,
                   label='Training data, physical', edgecolor='black', linewidth=0.3)
        ax.scatter(e[unphys], p[unphys], s=30, color='red', marker='x', linewidth=1.2,
                   label=r'Training data, excluded (unphysical, $E_{loss}$<0.25 eV)')
        plotted = True
    if gen_csv is not None:
        gdf = pd.read_csv(gen_csv)
        ax.scatter(gdf['E_loss_pred_physical'], gdf['surrogate_PCE'], s=8,
                   color=COLORS['acceptor'], alpha=0.35, label='Generated (predicted PCE)')
        plotted = True
    ax.axvline(0.25, color='grey', linestyle=':', linewidth=1.2,
               label='Detailed-balance floor (0.25 eV)')
    if not plotted:
        plt.close(fig)
        print(f"  [SKIP] {name}: missing eloss_vs_pce_training_data.csv and "
              f"eloss_vs_pce_generated_corrected_data.csv")
        return False
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 6: energy-loss vs PCE (corrected).")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_eloss_vs_pce(data_dir, outdir)


if __name__ == "__main__":
    main()
