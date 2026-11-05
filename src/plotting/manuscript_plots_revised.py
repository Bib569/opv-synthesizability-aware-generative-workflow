#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Producer of the Fig. 5 plot data (sascore_rascore_tango_4d_{donor,acceptor}_data.csv).
# Usage: python manuscript_plots_revised.py --data-dir <dir with merged/ and synthesizability/> --outdir <out>.
# Needs HPC outputs that are NOT in the repository: <data-dir>/merged/plot_data/*_molecule_predictions_data.csv
# (a copy of data/generated_libraries/*_molecule_predictions.csv) and <data-dir>/synthesizability/plot_data/
# sascore_vs_rascore_*_data.csv (written by stage 3, hpc/sub_synth_scoring_240.sh, then merged).
# The CSVs it produces for the paper are committed in data/figure_inputs/.
# -----------------------------------------------------------------------------
"""
Revised Manuscript Plots — Corrected and Enhanced
==================================================

Regenerates selected plots with fixes:
  1. donor_space_analysis / acceptor_space_analysis:
     - Color by quality_score using 'coolwarm' (not green/viridis)
     - Show actual n (not capped at 20000) -- OR if subsampled, state original n
     - Rename "Mean Inception Score" → "Internal Diversity (IntDiv)"
  2. device_properties_distribution.png REMOVED (Method 2 artifacts)
     - Only keep device_properties_distribution_method1_ridge_gp.png
  3. discriminator_accuracy — show only epochs 0 to ~15 (uses available data)
  4. sascore_vs_rascore — move "Pass both" annotation above SA threshold box
  5. NEW: 4D SAScore × RAScore × TANGO scatter (3 scores in one plot)

Reads from existing plot_data CSVs. No re-computation needed.

Usage:
    python manuscript_plots_revised.py --data-dir ./latest_results --outdir ./revised_plots

Requirements:
    pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════════

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.minor.width'] = 1.2
plt.rcParams['ytick.minor.width'] = 1.2
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False

COLORS = {
    'donor': '#1B7F79', 'acceptor': '#D4602E',
    'sascore': '#2C5F8A', 'rascore': '#C62828', 'combined': '#3A8C3F',
    'threshold': '#888888', 'mean_line': '#212121',
    'd_loss': '#2C5F8A', 'g_loss': '#D94F30', 'equilibrium': '#3A8C3F',
    'accuracy': '#2C5F8A', 'target': '#3A8C3F',
}


def _style_ax(ax, xlabel='', ylabel='', fontsize=13):
    """Publication-quality 2D axis styling."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    ax.tick_params(width=2.0, length=6, labelsize=11, direction='in',
                   top=True, right=True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    ax.grid(False)


def _style_ax3d(ax):
    """Publication-quality 3D axis styling."""
    ax.tick_params(labelsize=10)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(1.5)
    ax.yaxis.pane.set_linewidth(1.5)
    ax.zaxis.pane.set_linewidth(1.5)


# ══════════════════════════════════════════════════════════════════════════════════
# PLOT 1 & 2: SPACE ANALYSIS (fixed colors + label)
# ══════════════════════════════════════════════════════════════════════════════════

def create_space_analysis_revised(mol_type, merged_dir, output_dir):
    """
    Fixes:
    - Color by quality_score using 'coolwarm' (violet-to-red spectrum)
    - Shows 'n = X (of Y total)' when subsampled
    - Renames z-axis from "Mean Inception Score" to "Internal Diversity (IntDiv)"
    """
    csv_name = f'{mol_type}_hybrid_12method_visualization_data.csv'
    csv_path = os.path.join(merged_dir, csv_name)
    if not os.path.exists(csv_path):
        print(f"  ✗ Not found: {csv_path}")
        return False

    print(f"  Loading {csv_name}...")
    df = pd.read_csv(csv_path)
    similarities = df['similarities'].values
    prop_scores = df['prop_scores'].values
    mfd_scores = df['mfd_scores'].values
    mis_scores = df['mis_scores'].values
    color_vals = df['quality_score'].values if 'quality_score' in df.columns else mis_scores

    n_original = len(similarities)
    max_pts = 20000
    if n_original > max_pts:
        idx = np.random.RandomState(42).choice(n_original, max_pts, replace=False)
        similarities = similarities[idx]
        prop_scores = prop_scores[idx]
        mfd_scores = mfd_scores[idx]
        mis_scores = mis_scores[idx]
        color_vals = color_vals[idx]
        n_label = f"$n$ = {max_pts:,} (of {n_original:,} total)"
    else:
        n_label = f"$n$ = {n_original:,}"

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.30, wspace=0.30)

    # 3D scatter
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    _style_ax3d(ax1)

    scatter = ax1.scatter(
        similarities, prop_scores, mis_scores,
        c=color_vals, cmap='coolwarm', s=12, alpha=0.6, depthshade=False,
    )
    ax1.set_xlabel('Similarity to Training Set', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Property Match Score', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Internal Diversity (IntDiv)', fontsize=12, fontweight='bold', labelpad=10)

    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.55, pad=0.08)
    cbar.set_label('Quality Score', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Stats box
    mfd_val = np.mean(mfd_scores)
    stats_text = (
        f"{n_label}\n"
        f"Avg Similarity = {np.mean(similarities):.3f}\n"
        f"Avg Property Match = {np.mean(prop_scores):.3f}\n"
        f"Avg IntDiv = {np.mean(mis_scores):.3f}\n"
        f"Global FP-PCA-FD = {mfd_val:.3f}"
    )
    fig.text(0.02, 0.92, stats_text, fontsize=9, family='serif',
             verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', linewidth=1.2, alpha=0.9))

    # Histograms
    color = COLORS['donor'] if mol_type == 'donor' else COLORS['acceptor']
    ax2 = fig.add_subplot(gs[1, 0])
    _style_ax(ax2, xlabel='Similarity Score', ylabel='Frequency')
    ax2.hist(similarities, bins=50, color=color, alpha=0.75,
             edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(similarities), color=COLORS['mean_line'],
                linestyle='-', linewidth=1.8,
                label=f'Mean: {np.mean(similarities):.3f}')
    ax2.legend(fontsize=9, loc='upper right')

    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3, xlabel='Property Match Score', ylabel='Frequency')
    ax3.hist(prop_scores, bins=50, color=color, alpha=0.75,
             edgecolor='black', linewidth=0.5)
    ax3.axvline(np.mean(prop_scores), color=COLORS['mean_line'],
                linestyle='-', linewidth=1.8,
                label=f'Mean: {np.mean(prop_scores):.3f}')
    ax3.legend(fontsize=9, loc='upper left')

    out_path = os.path.join(output_dir, f'{mol_type}_space_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════════
# PLOT 3 & 4: DISCRIMINATOR ACCURACY (epochs 0–15 or all available)
# ══════════════════════════════════════════════════════════════════════════════════

def create_discriminator_accuracy_plot(mol_type, plot_data_dir, output_dir, max_epoch=15):
    """Show discriminator accuracy only up to max_epoch for cleaner visualization."""
    csv_path = os.path.join(plot_data_dir, f'discriminator_accuracy_{mol_type}_optimized.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    # Filter to max_epoch
    df = df[df['epoch'] <= max_epoch].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['epoch'], df['discriminator_accuracy_percent'],
            color=COLORS['accuracy'], linewidth=2.5, marker='o', markersize=6)
    ax.axhline(50.0, color=COLORS['target'], linestyle='--', linewidth=2.0,
               label='Target (50% = equilibrium)')
    _style_ax(ax, xlabel='Epoch', ylabel='Discriminator Accuracy (%)')
    ax.set_xlim(-0.5, max_epoch + 0.5)
    ax.set_ylim(35, 60)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'discriminator_accuracy_{mol_type}_optimized.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════════
# PLOT 5 & 6: SAScore vs RAScore (annotation repositioned)
# ══════════════════════════════════════════════════════════════════════════════════

def create_sascore_vs_rascore_revised(mol_type, synth_plot_data_dir, output_dir,
                                      threshold_sa=6.0, threshold_ra=0.5):
    """
    Fix: Move 'Pass both' annotation to top-left (above SA threshold region).
    """
    csv_path = os.path.join(synth_plot_data_dir, f'sascore_vs_rascore_{mol_type}_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(df["SAScore"], df["RAScore"],
                         c=df["combined_synthesizability"],
                         cmap='RdYlGn', alpha=0.4, s=8, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label('Combined Score', fontsize=12, fontweight='bold')

    # Threshold lines
    ax.axvline(threshold_sa, color=COLORS['threshold'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'SA threshold = {threshold_sa}')
    ax.axhline(threshold_ra, color=COLORS['threshold'], linestyle=':',
               linewidth=1.5, alpha=0.8, label=f'RA threshold = {threshold_ra}')

    # "Pass both" annotation — TOP LEFT (above SA threshold region)
    n_both = ((df["SAScore"] <= threshold_sa) & (df["RAScore"] >= threshold_ra)).sum()
    pct = 100 * n_both / len(df)
    ax.text(0.03, 0.97, f'Pass both: {n_both:,} ({pct:.1f}%)',
            transform=ax.transAxes, fontsize=11, va='top', ha='left', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                      edgecolor='darkgreen', linewidth=1.2, alpha=0.85))

    _style_ax(ax, xlabel='SAScore (1=easy, 10=hard)',
              ylabel='RAScore (0=hard, 1=easy)')
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'sascore_vs_rascore_{mol_type}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return True


# ══════════════════════════════════════════════════════════════════════════════════
# PLOT 7 & 8: 4D PLOT — SAScore × RAScore × TANGO (new combined view)
# ══════════════════════════════════════════════════════════════════════════════════

def create_4d_synth_tango_plot(mol_type, synth_results_dir, mol_predictions_path,
                               output_dir, threshold_sa=6.0, threshold_ra=0.5):
    """
    3D scatter: X=SAScore, Y=RAScore, Z=TANGO, color=combined_synthesizability.
    Merges synthesizability scores with TANGO from molecule predictions.
    """
    # Load synthesizability scores
    synth_path = os.path.join(synth_results_dir,
                              f'{mol_type}_synthesizability_scored.csv')
    if not os.path.exists(synth_path):
        print(f"  ✗ Not found: {synth_path}")
        return False

    synth_df = pd.read_csv(synth_path)

    # Load TANGO scores from molecule predictions
    if not os.path.exists(mol_predictions_path):
        print(f"  ✗ Not found: {mol_predictions_path}")
        return False

    pred_df = pd.read_csv(mol_predictions_path)
    tango_col = 'tango_inspired_screen'
    if tango_col not in pred_df.columns:
        print(f"  ✗ '{tango_col}' column not in {mol_predictions_path}")
        return False

    # Merge on SMILES
    pred_df = pred_df.rename(columns={'smiles': 'smiles'})
    merged = synth_df.merge(pred_df[['smiles', tango_col]], on='smiles', how='inner')
    merged = merged.dropna(subset=['SAScore', 'RAScore', tango_col])

    if len(merged) == 0:
        print(f"  ✗ No overlapping molecules for {mol_type} 4D plot")
        return False

    print(f"    Merged {len(merged)} molecules with all three scores")

    # Subsample if too large
    max_pts = 20000
    if len(merged) > max_pts:
        merged = merged.sample(max_pts, random_state=42)
        print(f"    Sub-sampled to {max_pts} points")

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    _style_ax3d(ax)

    scatter = ax.scatter(
        merged['SAScore'], merged['RAScore'], merged[tango_col],
        c=merged['combined_synthesizability'],
        cmap='RdYlGn', s=10, alpha=0.5, edgecolors='none', depthshade=False
    )

    ax.set_xlabel('SAScore\n(1=easy, 10=hard)', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_ylabel('RAScore\n(0=hard, 1=easy)', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_zlabel('TANGO Score\n(higher=more synthesizable)', fontsize=11,
                  fontweight='bold', labelpad=12)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Combined Synthesizability', fontsize=11, fontweight='bold')

    # Stats annotation
    n_pass_all = ((merged['SAScore'] <= threshold_sa) &
                  (merged['RAScore'] >= threshold_ra) &
                  (merged[tango_col] >= 0.5)).sum()
    pct_all = 100 * n_pass_all / len(merged)
    stats_text = (
        f"$n$ = {len(merged):,}\n"
        f"SA≤{threshold_sa} & RA≥{threshold_ra} & TANGO≥0.5:\n"
        f"  {n_pass_all:,} ({pct_all:.1f}%)"
    )
    fig.text(0.02, 0.95, stats_text, fontsize=9, family='serif',
             verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', linewidth=1.0, alpha=0.9))

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'sascore_rascore_tango_4d_{mol_type}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")

    # Save plot data
    plot_data_out = os.path.join(output_dir, f'sascore_rascore_tango_4d_{mol_type}_data.csv')
    merged[['smiles', 'SAScore', 'RAScore', tango_col, 'combined_synthesizability']].to_csv(
        plot_data_out, index=False)
    print(f"  ✓ Plot data: {plot_data_out}")
    return True


# ══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Revised manuscript plots")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("latest_results"),
                    help="Root directory with merged/, synthesizability/, etc.")
    ap.add_argument("--plot-data-dir", type=Path, default=None,
                    help="Directory with GAN training plot CSVs (gan_losses, disc_accuracy)")
    ap.add_argument("--outdir", type=Path, default=Path("revised_manuscript_plots"),
                    help="Output directory for revised plots")
    ap.add_argument("--max-disc-epoch", type=int, default=15,
                    help="Max epoch to show in discriminator accuracy plots")
    args = ap.parse_args()

    data_dir = args.data_dir
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)

    # Resolve sub-directories
    merged_dir = data_dir / "merged"
    merged_plot_data = merged_dir / "plot_data"
    synth_dir = data_dir / "synthesizability"
    synth_plot_data = synth_dir / "plot_data"
    synth_results = synth_dir / "results"

    # GAN training plot data (gan_losses, disc_accuracy) - may be separate
    if args.plot_data_dir:
        gan_plot_data = args.plot_data_dir
    else:
        # Try common locations
        gan_plot_data = data_dir / "plot_data"
        if not gan_plot_data.exists():
            gan_plot_data = data_dir.parent / "plot_data"

    print("=" * 70)
    print("REVISED MANUSCRIPT PLOTS")
    print("=" * 70)
    print(f"  Data dir       : {data_dir}")
    print(f"  Merged dir     : {merged_dir}")
    print(f"  Synth dir      : {synth_dir}")
    print(f"  GAN plot data  : {gan_plot_data}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 70)
    print()

    produced = 0
    skipped = 0

    # ── 1 & 2: Space analysis (fixed) ─────────────────────────────────────────
    print("[1/8] Donor Space Analysis (revised: coolwarm colors, IntDiv label)")
    if create_space_analysis_revised("donor", str(merged_dir), str(output_dir)):
        produced += 1
    else:
        skipped += 1

    print("\n[2/8] Acceptor Space Analysis (revised)")
    if create_space_analysis_revised("acceptor", str(merged_dir), str(output_dir)):
        produced += 1
    else:
        skipped += 1

    # ── 3 & 4: Discriminator accuracy (truncated epochs) ─────────────────────
    print(f"\n[3/8] Discriminator Accuracy — Donor (epochs 0–{args.max_disc_epoch})")
    if create_discriminator_accuracy_plot("donor", str(gan_plot_data), str(output_dir),
                                          max_epoch=args.max_disc_epoch):
        produced += 1
    else:
        skipped += 1

    print(f"\n[4/8] Discriminator Accuracy — Acceptor (epochs 0–{args.max_disc_epoch})")
    if create_discriminator_accuracy_plot("acceptor", str(gan_plot_data), str(output_dir),
                                          max_epoch=args.max_disc_epoch):
        produced += 1
    else:
        skipped += 1

    # ── 5 & 6: SAScore vs RAScore (repositioned annotation) ──────────────────
    print("\n[5/8] SAScore vs RAScore — Donor (annotation repositioned)")
    if create_sascore_vs_rascore_revised("donor", str(synth_plot_data), str(output_dir)):
        produced += 1
    else:
        skipped += 1

    print("\n[6/8] SAScore vs RAScore — Acceptor (annotation repositioned)")
    if create_sascore_vs_rascore_revised("acceptor", str(synth_plot_data), str(output_dir)):
        produced += 1
    else:
        skipped += 1

    # ── 7 & 8: 4D SAScore × RAScore × TANGO ──────────────────────────────────
    print("\n[7/8] 4D SAScore×RAScore×TANGO — Donor")
    donor_pred_path = str(merged_plot_data / "donor_molecule_predictions_data.csv")
    if create_4d_synth_tango_plot("donor", str(synth_results), donor_pred_path, str(output_dir)):
        produced += 1
    else:
        skipped += 1

    print("\n[8/8] 4D SAScore×RAScore×TANGO — Acceptor")
    acc_pred_path = str(merged_plot_data / "acceptor_molecule_predictions_data.csv")
    if create_4d_synth_tango_plot("acceptor", str(synth_results), acc_pred_path, str(output_dir)):
        produced += 1
    else:
        skipped += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Produced: {produced}/8")
    print(f"  Skipped:  {skipped}/8")
    print(f"  Output:   {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
