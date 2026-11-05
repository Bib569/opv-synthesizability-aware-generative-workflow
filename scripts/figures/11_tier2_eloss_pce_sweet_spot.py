#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
11_tier2_eloss_pce_sweet_spot.py  ->  tier2_eloss_pce_sweet_spot.png
=============================================================================
Tier-2 energy-loss vs predicted-PCE sweet-spot scatter (lightweight CSV redraw).

Required input CSV (in --data-dir):
    tier2_eloss_pce_sweet_spot_data.csv    cols: E_loss, PCE, is_sweet

Usage:
    python 11_tier2_eloss_pce_sweet_spot.py --data-dir data/figure_inputs
=============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, need, parse_common_args, resolve_dirs)

ELOSS_SWEET_MAX = 0.55
PCE_SWEET_MIN = 17.0
_SWEET_COLOR = '#8E24AA'


def fig_sweet_spot(data_dir, outdir):
    csv = need(data_dir, 'tier2_eloss_pce_sweet_spot_data.csv')
    name = 'tier2_eloss_pce_sweet_spot.png'
    if csv is None:
        print(f"  [SKIP] {name}: missing tier2_eloss_pce_sweet_spot_data.csv")
        return False
    df = pd.read_csv(csv)
    valid = df
    sweet = df[df['is_sweet'] == 1]
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    style_ax(ax, xlabel=r'$E_{\mathrm{loss}}$ (eV)', ylabel='Predicted PCE (%)')
    ax.scatter(valid['E_loss'], valid['PCE'], s=14, color='#9E9E9E', alpha=0.45,
               edgecolor='none', label=f'Tier-2 shortlist (n={len(valid):,})')
    ax.scatter(sweet['E_loss'], sweet['PCE'], s=32, color=_SWEET_COLOR, alpha=0.9,
               edgecolor='black', linewidth=0.4,
               label=f'Sweet-spot subset (n={len(sweet):,})')
    ax.axvline(ELOSS_SWEET_MAX, color=COLORS['threshold'], linestyle='--', linewidth=1.6,
               label=fr'$E_{{\mathrm{{loss}}}} \leq {ELOSS_SWEET_MAX}$ eV')
    ax.axhline(PCE_SWEET_MIN, color='#1565C0', linestyle=':', linewidth=1.6,
               label=fr'PCE $\geq {PCE_SWEET_MIN}\%$')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95)
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 11: Tier-2 energy-loss/PCE sweet-spot.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_sweet_spot(data_dir, outdir)


if __name__ == "__main__":
    main()
