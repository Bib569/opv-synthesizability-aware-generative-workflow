#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
07_sascore_rascore_tango_4d.py  ->  sascore_rascore_tango_4d_combined.png
=============================================================================
Combined donor+acceptor 4D SAScore x RAScore x TANGO-inspired figure.

Required input CSVs (in --data-dir):
    sascore_rascore_tango_4d_donor_data.csv
    sascore_rascore_tango_4d_acceptor_data.csv
        cols: SAScore, RAScore, combined_synthesizability,
              tango_inspired_screen (or TANGO_score)

Usage:
    python 07_sascore_rascore_tango_4d.py --data-dir data/figure_inputs
=============================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, style_ax3d, panel_label_3d, need,
                           parse_common_args, resolve_dirs)


def _load_4d(csv_path):
    merged = pd.read_csv(csv_path)
    tango_col = ('tango_inspired_screen' if 'tango_inspired_screen' in merged.columns
                 else ('TANGO_score' if 'TANGO_score' in merged.columns else None))
    if tango_col is None:
        return None, None
    return merged, tango_col


def _draw_4d_panel(fig, ax, merged, tango_col, plabel,
                   threshold_sa=6.0, threshold_ra=0.5, threshold_tango=0.5, show_zlabel=True):
    style_ax3d(ax, tick_labelsize=13)
    scatter = ax.scatter(merged['SAScore'], merged['RAScore'], merged[tango_col],
                         c=merged['combined_synthesizability'], cmap='RdYlGn',
                         s=8, alpha=0.5, edgecolors='none', depthshade=False)
    ax.set_xlabel('SAScore\n(1=easy, 10=hard)', fontsize=13, fontweight='bold', labelpad=14)
    ax.set_ylabel('RAScore\n(0=hard, 1=easy)', fontsize=13, fontweight='bold', labelpad=14)
    if show_zlabel:
        ax.set_zlabel('TANGO-inspired score\n(higher=more synthesizable)', fontsize=13,
                      fontweight='bold', labelpad=14)
    else:
        ax.set_zlabel('')
    cbar_pad = 0.13 if not show_zlabel else 0.02
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=cbar_pad)
    cbar.set_label('Combined Synthesizability', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    n_pass = ((merged['SAScore'] <= threshold_sa) & (merged['RAScore'] >= threshold_ra) &
              (merged[tango_col] >= threshold_tango)).sum()
    pct = 100 * n_pass / len(merged)
    stats_text = (f"$n$ = {len(merged):,}\n"
                  f"SA\u2264{threshold_sa} & RA\u2265{threshold_ra} & TANGO-inspired\u2265{threshold_tango}:\n"
                  f"  {n_pass:,} ({pct:.1f}%)")
    ax.text2D(0.02, 0.88, stats_text, fontsize=12, family='serif', verticalalignment='top',
              transform=ax.transAxes,
              bbox=dict(facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.9))
    panel_label_3d(ax, plabel)
    ax.view_init(elev=25, azim=45)
    return cbar


def fig_4d_combined(data_dir, outdir):
    d_csv = need(data_dir, 'sascore_rascore_tango_4d_donor_data.csv')
    a_csv = need(data_dir, 'sascore_rascore_tango_4d_acceptor_data.csv')
    name = 'sascore_rascore_tango_4d_combined.png'
    if d_csv is None or a_csv is None:
        miss = ['sascore_rascore_tango_4d_donor_data.csv'] if d_csv is None else []
        miss += ['sascore_rascore_tango_4d_acceptor_data.csv'] if a_csv is None else []
        print(f"  [SKIP] {name}: missing {', '.join(miss)}")
        return False
    d_merged, d_tango = _load_4d(d_csv)
    a_merged, a_tango = _load_4d(a_csv)
    if d_merged is None or a_merged is None:
        print(f"  [SKIP] {name}: 4D CSV lacks a TANGO column (tango_inspired_screen/TANGO_score)")
        return False
    fig = plt.figure(figsize=(18, 8.5))
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    _draw_4d_panel(fig, ax1, d_merged, d_tango, 'a)', show_zlabel=False)
    cbar_b = _draw_4d_panel(fig, ax2, a_merged, a_tango, 'b)', show_zlabel=False)
    plt.tight_layout()
    b_pos = ax2.get_position()
    cbar_pos = cbar_b.ax.get_position()
    tango_x = b_pos.x1 + 0.62 * (cbar_pos.x0 - b_pos.x1)
    tango_y = (b_pos.y0 + b_pos.y1) / 2.0
    fig.text(tango_x, tango_y, 'TANGO-inspired score\n(higher=more synthesizable)',
             rotation=90, rotation_mode='anchor', ha='center', va='center',
             fontsize=13, fontweight='bold')
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 7: combined SAScore/RAScore/TANGO 4D figure.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_4d_combined(data_dir, outdir)


if __name__ == "__main__":
    main()
