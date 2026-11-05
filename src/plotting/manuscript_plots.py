#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Pipeline stage 1/2 plotting (original exploratory plot set; writes PNGs only).
# Run from the flat run directory built by `sh hpc/setup_run_dir.sh` (see docs/PIPELINE.md);
# the script expects DMPNN.csv / Guzik_data.csv and its sibling scripts in that directory.
# Reads <script dir>/merged/ (the consolidated merge outputs created by hpc/sub_140.sh or
# hpc/sub_postgen_full_140.sh) and writes to <script dir>/manuscript_plots_output/. Writes several of the plot-data CSVs
# that scripts/figures/ consumes (all committed in data/figure_inputs/). The manuscript figures themselves are
# regenerated from those CSVs with scripts/figures/run_all.py, not with this script.
# -----------------------------------------------------------------------------
"""
Manuscript-Ready Plots for GAN Pipeline Results
=================================================
Recreates the following plots from merged/ data with publication-quality
formatting — no titles, serif fonts, bold axis borders, 300 dpi:

  1. donor_space_analysis.png            (3D scatter + histograms, no percentiles)
  2. acceptor_space_analysis.png         (3D scatter + histograms, no percentiles)
  3. device_properties_distribution_method1_ridge_gp.png
  4. device_properties_distribution_method2_physics.png
  5. device_properties_distribution.png  (blended: Ridge+GP and Physics only, no SISSO)
  6. tango_distribution.png
  7. approach_comparison.png
  8. gan_losses_donor_optimized.png       (enhanced, full epochs)
  9. gan_losses_acceptor_optimized.png    (enhanced, full epochs)
 10. discriminator_accuracy_donor_optimized.png   (enhanced, full epochs)
 11. discriminator_accuracy_acceptor_optimized.png (enhanced, full epochs)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC/headless
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# GLOBAL STYLE CONFIGURATION (manuscript-ready)
# ============================================================================
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

# Proper dedicated output folder
_base_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    'dpi': 300,
    'merged_dir': os.path.join(_base_dir, 'merged'),
    'output_dir': os.path.join(_base_dir, 'manuscript_plots_output'),
}

# Deep, high-contrast color palette
COLORS = {
    'donor':       '#1B7F79',   # deep teal
    'acceptor':    '#D4602E',   # burnt orange
    'd_loss':      '#2C5F8A',   # steel navy
    'g_loss':      '#D94F30',   # coral red
    'equilibrium': '#3A8C3F',   # forest green
    'accuracy':    '#2C5F8A',   # steel navy
    'target':      '#3A8C3F',   # forest green
    'pce':         '#C62828',   # deep red
    'jsc':         '#1565C0',   # deep blue
    'ff':          '#2E7D32',   # deep green
    'voc':         '#E65100',   # deep orange
    'bar_donor':   '#1565C0',   # blue
    'bar_acc':     '#D84315',   # red-orange
    'threshold':   '#C62828',   # red for thresholds
    'mean_line':   '#212121',   # near-black for mean
}


def _style_ax(ax, xlabel='', ylabel='', fontsize=13):
    """Apply consistent publication styling to a 2D axis."""
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
    """Apply consistent publication styling to a 3D axis."""
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


# ============================================================================
# 1 & 2: DONOR / ACCEPTOR SPACE ANALYSIS
# ============================================================================

def create_space_analysis_plot(mol_type):
    """Create 3D scatter + 2 histogram space analysis plot.
    mol_type: 'donor' or 'acceptor'
    No titles, no 50th/75th percentile planes.
    Uses quality_score for color to get meaningful color variation.
    Stats box and threshold/method info in top-left corner.
    """
    csv_name = f'{mol_type}_hybrid_12method_visualization_data.csv'
    csv_path = os.path.join(CONFIG['merged_dir'], csv_name)
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return

    print(f"  Loading {csv_name}...")
    df = pd.read_csv(csv_path)
    similarities = df['similarities'].values
    prop_scores  = df['prop_scores'].values
    mfd_scores   = df['mfd_scores'].values
    mis_scores   = df['mis_scores'].values
    # Use quality_score for color since mfd_scores is constant
    color_vals   = df['quality_score'].values if 'quality_score' in df.columns else mfd_scores

    # Sub-sample for performance if very large
    n = len(similarities)
    max_pts = 20000
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        similarities = similarities[idx]
        prop_scores  = prop_scores[idx]
        mfd_scores   = mfd_scores[idx]
        mis_scores   = mis_scores[idx]
        color_vals   = color_vals[idx]
        print(f"    Sub-sampled {n} → {max_pts} points for plotting")

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.30, wspace=0.30)

    # 3D scatter (top spanning both columns)
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    _style_ax3d(ax1)

    scatter = ax1.scatter(
        similarities, prop_scores, mis_scores,
        c=color_vals, cmap='viridis', s=12, alpha=0.6,
        depthshade=False,
    )
    ax1.set_xlabel('Similarity to Training Set', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Property Match Score', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Mean Inception Score', fontsize=12, fontweight='bold', labelpad=10)

    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.55, pad=0.08)
    cbar.set_label('Quality Score', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Statistics text box — TOP LEFT of 3D plot (includes Global MFD)
    mfd_val = np.mean(mfd_scores)
    stats_text = (
        f"$n$ = {len(similarities)}\n"
        f"Avg Sim = {np.mean(similarities):.3f}\n"
        f"Avg Prop = {np.mean(prop_scores):.3f}\n"
        f"Avg MIS = {np.mean(mis_scores):.3f}\n"
        f"Global MFD = {mfd_val:.3f}"
    )
    fig.text(0.02, 0.92, stats_text, fontsize=9, family='serif',
             verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', linewidth=1.2, alpha=0.9))

    # Histogram 1: Similarities
    ax2 = fig.add_subplot(gs[1, 0])
    _style_ax(ax2, xlabel='Similarity Score', ylabel='Frequency')
    ax2.hist(similarities, bins=50, color=COLORS['donor'] if mol_type == 'donor' else COLORS['acceptor'],
             alpha=0.75, edgecolor='black', linewidth=0.5)
    ax2.axvline(np.mean(similarities), color=COLORS['mean_line'], linestyle='-', linewidth=1.8,
                label=f'Mean: {np.mean(similarities):.3f}')
    ax2.legend(fontsize=9, loc='upper right')

    # Histogram 2: Property Scores
    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3, xlabel='Property Match Score', ylabel='Frequency')
    ax3.hist(prop_scores, bins=50, color=COLORS['donor'] if mol_type == 'donor' else COLORS['acceptor'],
             alpha=0.75, edgecolor='black', linewidth=0.5)
    ax3.axvline(np.mean(prop_scores), color=COLORS['mean_line'], linestyle='-', linewidth=1.8,
                label=f'Mean: {np.mean(prop_scores):.3f}')
    ax3.legend(fontsize=9, loc='upper left')

    out_path = os.path.join(CONFIG['output_dir'], f'{mol_type}_space_analysis.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 3 & 4: DEVICE PROPERTIES per METHOD
# ============================================================================

def create_device_properties_method_plot(method_tag, cols_map, out_filename):
    """Create 2x2 subplot of PCE/Jsc/FF/Voc histograms for one prediction method.
    Threshold and method legends in TOP LEFT.
    """
    csv_name = f'{out_filename.replace(".png", "_data.csv")}'
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', csv_name)
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return

    print(f"  Loading {csv_name}...")
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')

    for ax, (label, (col, color, thresh)) in zip(axes.ravel(), cols_map.items()):
        _style_ax(ax, xlabel=label, ylabel='Frequency')
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=60, color=color, alpha=0.72, edgecolor='black', linewidth=0.4)
                ax.axvline(thresh, color=COLORS['threshold'], linestyle='--', linewidth=2.0,
                           label=f'Threshold = {thresh}')
                ax.axvline(data.mean(), color=COLORS['mean_line'], linestyle='-', linewidth=1.8,
                           label=f'Mean = {data.mean():.2f}')
                pct = (data >= thresh).sum() / len(data) * 100
                # Put percentage as annotation instead of title
                ax.annotate(f'{pct:.1f}% above threshold',
                            xy=(0.97, 0.93), xycoords='axes fraction',
                            ha='right', va='top', fontsize=10, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='gray',
                                      linewidth=0.8, alpha=0.9))
                # Legend in TOP LEFT
                ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Column N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], out_filename)
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 5: SCREENING-SURROGATE DEVICE PROPERTIES (Method 1 only)
#
# CHANGED: Method 2 (the Scharber-type physics estimate) has been REMOVED from
# this figure, matching its removal from the quantitative blend in
# GAN_240_final.py. Reason: its PCE/FF/Voc distributions were dominated by
# hard-coded clipping ceilings (PCE clipped at 21%, FF bounded 0.70-0.85, Voc
# bounded 0.3-2.0 V) rather than by emergent physics, and its mean parameter
# balance (high Voc, modest Jsc) does not resemble how real record-efficiency
# devices reach their performance. Plotting it beside Method 1 invited the
# reader to treat it as an independent quantitative prediction, which it is not.
# The frontier-orbital intuition is retained ONLY as the qualitative
# 'favorable_homo_lumo_alignment' flag, plotted separately.
# ============================================================================

def create_blended_device_properties_plot():
    """2x2 histograms of the Method-1 screening surrogate's device properties.

    Labelled explicitly as a SCREEN, not a validated predictor. The threshold
    lines use the values the filter code actually applies (FF = 0.75).
    """
    csv1 = os.path.join(CONFIG['merged_dir'], 'plot_data',
                        'device_properties_distribution_method1_ridge_gp_data.csv')
    csv_blend = os.path.join(CONFIG['merged_dir'], 'plot_data',
                             'device_properties_distribution_data.csv')

    src = csv_blend if os.path.exists(csv_blend) else csv1
    if not os.path.exists(src):
        print("  ✗ Screening-surrogate device data not found")
        return

    print(f"  Loading {os.path.basename(src)} (Method 1 only; Method 2 removed)...")
    df = pd.read_csv(src)

    # Prefer the final blended columns; fall back to the raw surrogate columns.
    candidates = {
        'PCE (%)':                  (['PCE', 'surrogate_PCE'], COLORS['pce'], 18.0),
        r'$J_{sc}$ (mA cm$^{-2}$)': (['Jsc', 'surrogate_Jsc'], COLORS['jsc'], 25.0),
        'FF':                       (['FF', 'surrogate_FF'],   COLORS['ff'],  0.75),
        r'$V_{oc}$ (V)':            (['Voc', 'surrogate_Voc'], COLORS['voc'], 1.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')

    for ax, (label, (cols, color, thresh)) in zip(axes.ravel(), candidates.items()):
        _style_ax(ax, xlabel=label, ylabel='Frequency')
        col = next((c for c in cols if c in df.columns), None)
        if col is None:
            ax.text(0.5, 0.5, 'Column N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            continue
        data = df[col].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            continue
        ax.hist(data, bins=60, color=color, alpha=0.72,
                edgecolor='black', linewidth=0.4, label='Screening surrogate')
        ax.axvline(thresh, color=COLORS['threshold'], linestyle='--', linewidth=2.0,
                   label=f'Threshold = {thresh}')
        ax.axvline(data.mean(), color=COLORS['mean_line'], linestyle='-', linewidth=1.8,
                   label=f'Mean = {data.mean():.2f}')
        pct = (data >= thresh).sum() / len(data) * 100
        ax.annotate(f'{pct:.1f}% above threshold',
                    xy=(0.97, 0.93), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='gray',
                              linewidth=0.8, alpha=0.9))
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'device_properties_distribution.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 6: TANGO DISTRIBUTION
# ============================================================================

def create_tango_distribution_plot():
    """1x2 subplot: Donor and Acceptor TANGO score histograms."""
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'tango_distribution_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ TANGO data not found: {csv_path}")
        return

    print(f"  Loading tango_distribution_data.csv...")
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    for ax, mol_type, color in [(ax1, 'Donor', COLORS['donor']),
                                 (ax2, 'Acceptor', COLORS['acceptor'])]:
        _style_ax(ax, xlabel='TANGO Score', ylabel='Frequency')
        subset = df[df['Type'] == mol_type]['TANGO_Score'].dropna()
        valid = subset[subset > 0]

        if len(valid) > 0:
            ax.hist(valid, bins=50, color=color, alpha=0.72,
                    edgecolor='black', linewidth=0.5)
            mean_val = np.mean(valid)
            ax.axvline(mean_val, color=COLORS['threshold'], linestyle='--',
                       linewidth=2.0,
                       label=f'Mean = {mean_val:.3f}')
            ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'tango_distribution.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 7: APPROACH COMPARISON
# ============================================================================

def create_approach_comparison_plot():
    """Grouped bar chart: Donors vs Acceptors per approach."""
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'approach_comparison_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Approach data not found: {csv_path}")
        return

    print(f"  Loading approach_comparison_data.csv...")
    df = pd.read_csv(csv_path)


    # Clean approach names for display
    df['approach_label'] = df['approach'].str.replace('_', ' ')

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel='', ylabel='Molecule Count')

    x = np.arange(len(df))
    w = 0.35
    bars1 = ax.bar(x - w/2, df['donor_count'], w, color=COLORS['bar_donor'],
                   alpha=0.85, edgecolor='black', linewidth=0.5, label='Donors')
    bars2 = ax.bar(x + w/2, df['acceptor_count'], w, color=COLORS['bar_acc'],
                   alpha=0.85, edgecolor='black', linewidth=0.5, label='Acceptors')

    ax.set_xticks(x)
    ax.set_xticklabels(df['approach_label'], rotation=40, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)



    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'approach_comparison.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 8 & 9: GAN LOSSES (ENHANCED, MANUSCRIPT-READY, FULL EPOCHS)
# ============================================================================

def create_gan_losses_plot(tag):
    """Enhanced GAN loss plot for donor_optimized or acceptor_optimized.

    Uses full-epoch data (_full.csv) to show all 12+ epochs and equilibrium.
    D and G final value annotations are placed inside the plot area.

    Features:
      - Dual-line plot with shaded regions
      - Equilibrium reference with annotation
      - Convergence region highlight
      - Clean, professional aesthetics
    """
    # Try full-epoch CSV first, fallback to regular
    csv_full = os.path.join(CONFIG['merged_dir'], 'plot_data', f'gan_losses_{tag}_full.csv')
    csv_reg  = os.path.join(CONFIG['merged_dir'], f'gan_losses_{tag}.csv')
    csv_path = csv_full if os.path.exists(csv_full) else csv_reg

    if not os.path.exists(csv_path):
        print(f"  ✗ GAN losses data not found: {csv_path}")
        return

    print(f"  Loading {os.path.basename(csv_path)} ({csv_path})...")
    df = pd.read_csv(csv_path)

    epochs = df['epoch'].values
    d_loss = df['discriminator_loss'].values
    g_loss = df['generator_loss'].values
    n_epochs = len(epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel='Epoch', ylabel='Loss')

    # Shaded region between D and G losses to show convergence gap
    ax.fill_between(epochs, d_loss, g_loss, alpha=0.08, color='#9E9E9E',
                    label='D–G gap')

    # Main loss curves with markers
    ax.plot(epochs, d_loss, 'o-', color=COLORS['d_loss'], linewidth=2.8,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['d_loss'],
            label='Discriminator Loss', zorder=5)
    ax.plot(epochs, g_loss, 's-', color=COLORS['g_loss'], linewidth=2.8,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['g_loss'],
            label='Generator Loss', zorder=5)

    # Equilibrium reference line
    eq_val = np.log(2)  # ≈ 0.693
    ax.axhline(y=eq_val, color=COLORS['equilibrium'], linestyle='--',
               linewidth=2.2, alpha=0.85, zorder=3)
    # Annotate equilibrium — positioned to avoid clutter
    eq_x = epochs[int(n_epochs * 0.6)]  # Place at ~60% of x-range
    ax.annotate(
        r'Equilibrium ($\ln 2 \approx 0.693$)',
        xy=(eq_x, eq_val),
        xytext=(eq_x, eq_val - 0.004),
        fontsize=10, fontweight='bold', color=COLORS['equilibrium'],
        ha='center', va='top',
    )

    # Compute Y-range first (used for adaptive annotation placement)
    all_loss = np.concatenate([d_loss, g_loss, [eq_val]])

    # Final values annotation — placed INSIDE the box, well left of the legend
    gap = abs(d_loss[-1] - g_loss[-1])
    # Small fixed offset for annotation text
    y_offset = 0.002

    # When curves converge (gap < 0.005), move text further left to avoid marker overlap
    x_frac = 0.55 if gap < 0.005 else 0.70
    ann_x = epochs[0] + (epochs[-1] - epochs[0]) * x_frac

    ax.annotate(
        f'D = {d_loss[-1]:.4f}',
        xy=(epochs[-1], d_loss[-1]),
        xytext=(ann_x, d_loss[-1] + y_offset),
        fontsize=9.5, color=COLORS['d_loss'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['d_loss'], lw=1.2),
        ha='right',
    )
    ax.annotate(
        f'G = {g_loss[-1]:.4f}',
        xy=(epochs[-1], g_loss[-1]),
        xytext=(ann_x, g_loss[-1] - y_offset),
        fontsize=9.5, color=COLORS['g_loss'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['g_loss'], lw=1.2),
        ha='right',
    )

    # Y-axis range with padding
    y_min = min(all_loss) - 0.005
    y_max = max(all_loss) + 0.005
    ax.set_ylim(y_min, y_max)

    # Integer x-ticks for all epochs
    ax.set_xticks(epochs)
    ax.set_xticklabels([str(int(e)) for e in epochs])

    # Legend
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='black',
              fancybox=False, ncol=1)

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], f'gan_losses_{tag}.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 10 & 11: DISCRIMINATOR ACCURACY (ENHANCED, MANUSCRIPT-READY, FULL EPOCHS)
# ============================================================================

def create_discriminator_accuracy_plot(tag):
    """Enhanced discriminator accuracy plot with full-epoch data.

    Features:
      - Gradient-filled area under curve
      - 50% target equilibrium with annotation
      - Marker highlighting convergence point
      - Clean, professional aesthetics
    """
    # Try full-epoch CSV first, fallback to regular
    csv_full = os.path.join(CONFIG['merged_dir'], 'plot_data', f'discriminator_accuracy_{tag}_full.csv')
    csv_reg  = os.path.join(CONFIG['merged_dir'], f'discriminator_accuracy_{tag}.csv')
    csv_path = csv_full if os.path.exists(csv_full) else csv_reg

    if not os.path.exists(csv_path):
        print(f"  ✗ Discriminator accuracy data not found: {csv_path}")
        return

    print(f"  Loading {os.path.basename(csv_path)} ({csv_path})...")
    df = pd.read_csv(csv_path)

    epochs = df['epoch'].values
    accuracy = df['discriminator_accuracy_percent'].values
    n_epochs = len(epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel='Epoch', ylabel='Accuracy (%)')

    # Shaded region under the accuracy curve
    ax.fill_between(epochs, 0, accuracy, alpha=0.12, color=COLORS['accuracy'])

    # Main accuracy curve with markers
    ax.plot(epochs, accuracy, 'o-', color=COLORS['accuracy'], linewidth=2.8,
            markersize=10, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['accuracy'],
            label='Discriminator Accuracy', zorder=5)

    # 50% target line
    ax.axhline(y=50.0, color=COLORS['target'], linestyle='--', linewidth=2.2,
               alpha=0.85, zorder=3)
    ax.annotate(
        'Target Equilibrium (50%)',
        xy=(epochs[int(n_epochs * 0.3)], 50.0),
        xytext=(epochs[int(n_epochs * 0.3)], 55),
        fontsize=10, fontweight='bold', color=COLORS['target'],
        ha='center', va='bottom',
        arrowprops=dict(arrowstyle='->', color=COLORS['target'], lw=1.5),
    )

    # Highlight the epoch closest to 50% (equilibrium crossing)
    closest_idx = np.argmin(np.abs(accuracy - 50.0))
    ax.plot(epochs[closest_idx], accuracy[closest_idx], '*',
            color='#FFD600', markersize=18, markeredgecolor='black',
            markeredgewidth=1.5, zorder=6)
    ax.annotate(
        f'Closest to 50%\n({accuracy[closest_idx]:.1f}%)',
        xy=(epochs[closest_idx], accuracy[closest_idx]),
        xytext=(epochs[closest_idx] + 0.5, accuracy[closest_idx] - 8),
        fontsize=9, fontweight='bold', color='#424242',
        arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2),
    )

    # Final value label
    ax.annotate(
        f'{accuracy[-1]:.1f}%',
        xy=(epochs[-1], accuracy[-1]),
        xytext=(epochs[-1] - 0.8, accuracy[-1] + 5),
        fontsize=10, color=COLORS['accuracy'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['accuracy'], lw=1.2),
    )

    ax.set_ylim(0, max(accuracy) + 10)
    ax.set_xticks(epochs)
    ax.set_xticklabels([str(int(e)) for e in epochs])

    ax.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='black',
              fancybox=False)

    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], f'discriminator_accuracy_{tag}.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 12: MOSES-QUARTET SUMMARY (Validity / Uniqueness / Novelty / IntDiv)
#     Source: plot_data/generative_metrics_data.csv (from postgen_pipeline.py)
# ============================================================================

def create_moses_quartet_plot():
    """Grouped bar chart of Validity / Uniqueness / Novelty / IntDiv.

    Plots the pooled(all), pooled(excluding Training_Set_Augmentation), and
    per-approach rows side by side so the augmentation-dominance caveat is
    visible directly in the figure, not just in the text report.
    """
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'generative_metrics_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    metrics = ['validity', 'uniqueness', 'novelty', 'internal_diversity']
    metric_labels = ['Validity', 'Uniqueness', 'Novelty', 'Internal Diversity\n(IntDiv)']

    for side in ('donor', 'acceptor'):
        rows = df[df['set'].astype(str).str.startswith(side)].copy()
        if rows.empty:
            continue
        # Order: pooled_all, pooled_excl_augmentation, then per-approach
        def _sort_key(s):
            if s.endswith('pooled_all_approaches'):
                return (0, s)
            if 'excluding_Training_Set_Augmentation' in s:
                return (1, s)
            return (2, s)
        rows = rows.iloc[rows['set'].map(_sort_key).argsort()]
        labels = [s.replace(f'{side}_', '').replace('approach_', '').replace('_', ' ')
                  for s in rows['set']]

        fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(labels)), 6.5))
        fig.patch.set_facecolor('white')
        _style_ax(ax, xlabel='', ylabel='Score')
        x = np.arange(len(labels))
        w = 0.2
        palette = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']
        for i, (m, lab, c) in enumerate(zip(metrics, metric_labels, palette)):
            if m not in rows.columns:
                continue
            ax.bar(x + (i - 1.5) * w, rows[m].values, w, label=lab, color=c,
                   edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='upper right', ncol=2)
        plt.tight_layout()
        out_path = os.path.join(CONFIG['output_dir'], f'moses_quartet_{side}.png')
        plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 13: THREE-WAY DISTANCE COMPARISON (FP-PCA-FD vs PFD vs real FCD)
# ============================================================================

def create_distance_comparison_plot():
    """Bar chart comparing the three distinctly-labelled distance metrics.

    FP-PCA-FD  = relabelled legacy fingerprint-PCA Frechet distance (NOT FCD)
    PFD        = Property-Embedding Frechet Distance (domain-appropriate, primary)
    FCD        = real Frechet ChemNet Distance (ChEMBL-trained, cross-field ref.)
    """
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'generative_metrics_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    pooled = df[df['set'].astype(str).str.endswith('pooled_all_approaches')].copy()
    if pooled.empty:
        print("  ✗ No pooled rows found for distance comparison")
        return

    cols = ['fp_pca_frechet_distance', 'property_embedding_frechet_distance',
            'frechet_chemnet_distance']
    col_labels = ['FP-PCA-FD\n(legacy, relabelled)', 'PFD\n(property-embedding, primary)',
                  'FCD\n(ChemNet, cross-field ref.)']

    # ------------------------------------------------------------------------
    # LOG y-AXIS IS ESSENTIAL HERE, not cosmetic.
    #
    # The three distances live on completely different numeric scales because
    # they are computed in completely different embedding spaces:
    #   FP-PCA-FD : 2-component PCA of a fingerprint  -> O(1-20)
    #   PFD       : penultimate activations of a property model -> O(100-1000)
    #   FCD       : ChemNet activation space          -> O(10-100)
    # On a shared LINEAR axis scaled to PFD (~820 for acceptors), an
    # FP-PCA-FD bar of 3.09 is 0.4% of the axis height and is invisible -- which
    # is precisely why the acceptor FP-PCA-FD bar appeared "missing" in the
    # earlier figure even though the value existed. A log axis makes all three
    # readable simultaneously. Values are printed on top of each bar as well so
    # the figure is unambiguous, and genuinely-unavailable values (NaN) are
    # labelled 'n/a' instead of silently producing no bar.
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel='', ylabel='Fréchet-type distance (log scale)')
    ax.set_yscale('log')
    x = np.arange(len(pooled))
    w = 0.25
    palette = ['#455A64', '#1B7F79', '#D4602E']
    for i, (c, lab, col) in enumerate(zip(cols, col_labels, palette)):
        if c not in pooled.columns:
            continue
        vals = pd.to_numeric(pooled[c], errors='coerce').values
        pos = x + (i - 1) * w
        finite = np.isfinite(vals)
        ax.bar(pos[finite], vals[finite], w, label=lab, color=col,
               edgecolor='black', linewidth=0.6)
        # Annotate every bar with its value; mark NaN entries explicitly.
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                ax.annotate(f'{v:.2f}', xy=(xi, v), xytext=(0, 3),
                           textcoords='offset points', ha='center',
                           fontsize=8, fontweight='bold')
            else:
                ax.annotate('n/a', xy=(xi, ax.get_ylim()[0]), xytext=(0, 3),
                           textcoords='offset points', ha='center',
                           fontsize=8, style='italic', color='#B00020')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_pooled_all_approaches', '') for s in pooled['set']])
    ax.legend(fontsize=9, loc='upper left')
    ax.annotate("Different embedding spaces -> different numeric scales; log axis.\n"
                "'n/a' = not computable for that set (see output_metrics_postgen.txt).",
                xy=(0.99, 0.02), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=8, style='italic', color='#424242')
    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'distance_metric_comparison.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 14: SASCORE DISTRIBUTION (donor / acceptor)
# ============================================================================

def create_sascore_distribution_plot():
    """Histogram of RDKit SAScore for generated donors and acceptors.

    SAScore range is 1 (easy) to 10 (hard); the RDKit/literature convention
    threshold of 6 is drawn for reference. This is a structural heuristic
    screen, not a validated synthesizability probability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    any_data = False
    for ax, side, color in zip(axes, ('donor', 'acceptor'), (COLORS['donor'], COLORS['acceptor'])):
        csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', f'{side}_molecule_predictions_data.csv')
        if not os.path.exists(csv_path):
            ax.set_title(f'{side.capitalize()} (data not available)')
            ax.axis('off')
            continue
        df = pd.read_csv(csv_path)
        if 'sascore' not in df.columns:
            ax.set_title(f'{side.capitalize()} (sascore column not available)')
            ax.axis('off')
            continue
        sas = pd.to_numeric(df['sascore'], errors='coerce').dropna()
        if sas.empty:
            ax.set_title(f'{side.capitalize()} (no valid SAScore values)')
            ax.axis('off')
            continue
        any_data = True
        _style_ax(ax, xlabel='SAScore (1=easy, 10=hard)', ylabel='Frequency')
        ax.hist(sas, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.5)
        ax.axvline(6.0, color='red', linestyle='--', linewidth=1.5,
                   label='Literature threshold (6.0)')
        ax.axvline(sas.mean(), color='black', linestyle='-', linewidth=1.5,
                   label=f'Mean: {sas.mean():.2f}')
        ax.legend(fontsize=9)
    if any_data:
        plt.tight_layout()
        out_path = os.path.join(CONFIG['output_dir'], 'sascore_distribution.png')
        plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved: {out_path}")
    else:
        plt.close(fig)
        print("  ✗ SAScore distribution skipped (no data on either side)")


# ============================================================================
# 15: E_LOSS vs PCE -- TRAINING (DMPNN.csv) vs GENERATED (predicted, gated)
# ============================================================================

def create_eloss_vs_pce_plot():
    """Scatter of E_loss vs PCE: training data (with excluded artefacts marked)
    vs generated pairs (predicted E_loss, physics-gated).

    Training-side E_loss/PCE_max come directly from DMPNN.csv (a fixed input
    file, not a plot-data export) -- NOT from stage_a_dataset_diagnostics.csv,
    which is a transposed summary-statistics table (one row per diagnostic
    key), not a per-row molecule table.
    """
    dmpnn_path = os.path.join(_base_dir, 'DMPNN.csv')
    gen_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'pair_predictions_data.csv')
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel=r'$E_{loss}$ (eV)', ylabel='PCE (%)')
    plotted = False

    if os.path.exists(dmpnn_path):
        tdf = pd.read_csv(dmpnn_path)
        eloss_col = 'E_loss' if 'E_loss' in tdf.columns else None
        pce_col = 'PCE_max(%)' if 'PCE_max(%)' in tdf.columns else ('PCE_max' if 'PCE_max' in tdf.columns else None)
        if eloss_col and pce_col:
            e = pd.to_numeric(tdf[eloss_col], errors='coerce')
            p = pd.to_numeric(tdf[pce_col], errors='coerce')
            ok = e.notna() & p.notna()
            unphys = ok & (e < 0.25)
            phys = ok & ~unphys
            ax.scatter(e[phys], p[phys], s=30, color=COLORS['donor'], alpha=0.7,
                      label='Training (DMPNN.csv), physical', edgecolor='black', linewidth=0.3)
            ax.scatter(e[unphys], p[unphys], s=30, color='none', edgecolor='red',
                      marker='x', linewidth=1.2,
                      label='Training, excluded (unphysical, <0.25 eV)')
            plotted = True
        else:
            print(f"  [warn] DMPNN.csv missing E_loss/PCE_max columns for training overlay")

    if os.path.exists(gen_path):
        gdf = pd.read_csv(gen_path)
        ecol = 'eloss_pred_physical' if 'eloss_pred_physical' in gdf.columns else None
        if ecol and 'PCE' in gdf.columns:
            e = pd.to_numeric(gdf[ecol], errors='coerce')
            p = pd.to_numeric(gdf['PCE'], errors='coerce')
            ok = e.notna() & p.notna()
            if ok.sum() > 20000:
                idx = np.random.RandomState(42).choice(np.where(ok)[0], 20000, replace=False)
                e, p = e.iloc[idx], p.iloc[idx]
            ax.scatter(e, p, s=8, color=COLORS['acceptor'], alpha=0.35,
                      label='Generated (predicted, physics-gated)')
            plotted = True

    ax.axvline(0.25, color='grey', linestyle=':', linewidth=1.2,
              label='Detailed-balance floor (0.25 eV)')
    if plotted:
        ax.legend(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(CONFIG['output_dir'], 'eloss_vs_pce.png')
        plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved: {out_path}")
    else:
        plt.close(fig)
        print("  ✗ E_loss vs PCE plot skipped (no usable data)")


# ============================================================================
# 16: METHOD-1 DISJOINT-CV PREDICTED-VS-ACTUAL (honest held-out error)
# ============================================================================

def create_method1_disjoint_cv_plot():
    """Bar chart of held-out R^2 per property/split from method1_disjoint_cv_report.csv,
    replacing an in-sample-only picture with the honest grouped-disjoint result.
    """
    # This report is written by GAN_240_final.py into its top-level results/
    # directory, NOT by either merge stage, so it can legitimately sit in
    # several places depending on which stages ran. Search all of them rather
    # than failing on the first miss (the earlier version only checked two
    # paths and reported "not found" even when the file existed in results/).
    _candidates = [
        os.path.join(CONFIG['merged_dir'], 'results', 'method1_disjoint_cv_report.csv'),
        os.path.join(CONFIG['merged_dir'], 'results_postgen', 'method1_disjoint_cv_report.csv'),
        os.path.join(CONFIG['merged_dir'], 'plot_data', 'method1_disjoint_cv_report.csv'),
        os.path.join(CONFIG['merged_dir'], 'method1_disjoint_cv_report.csv'),
        os.path.join(_base_dir, 'results', 'method1_disjoint_cv_report.csv'),
        os.path.join(_base_dir, 'method1_disjoint_cv_report.csv'),
    ]
    csv_path = next((p for p in _candidates if os.path.exists(p)), None)
    if csv_path is None:
        print("  ✗ Data not found: method1_disjoint_cv_report.csv")
        print("      searched: " + "; ".join(_candidates))
        return
    df = pd.read_csv(csv_path)
    if df.empty or 'property' not in df.columns or 'heldout_r2' not in df.columns:
        print("  ✗ method1_disjoint_cv_report.csv missing expected 'property'/'heldout_r2' columns")
        return
    df = df[df['status'] == 'ok'].copy() if 'status' in df.columns else df.copy()
    if df.empty:
        print("  ✗ method1_disjoint_cv_report.csv has no completed CV rows (status=='ok')")
        return
    df['label'] = df['property'].astype(str) + ' / ' + df.get('split_kind', '').astype(str)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 6.5))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel='', ylabel=r'Held-out $R^2$ (group-disjoint CV)')
    vals = pd.to_numeric(df['heldout_r2'], errors='coerce')
    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in vals.fillna(0)]
    ax.bar(np.arange(len(df)), vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df['label'], rotation=75, ha='right', fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(CONFIG['output_dir'], 'method1_disjoint_cv_r2.png')
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")


# ============================================================================
# 17 & 18: INDIVIDUAL DONOR / ACCEPTOR PROPERTY DISTRIBUTIONS
#     Source: plot_data/donor_molecule_predictions_data.csv,
#             plot_data/acceptor_molecule_predictions_data.csv
# ============================================================================

def _grid_hist(df, cols_present, out_path, ncols=3, color='#1565C0'):
    n = len(cols_present)
    if n == 0:
        return False
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols_present):
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if data.empty:
            ax.axis('off')
            continue
        _style_ax(ax, xlabel=col, ylabel='Frequency', fontsize=10)
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor='black', linewidth=0.4)
        ax.axvline(data.mean(), color='black', linewidth=1.2,
                  label=f'Mean: {data.mean():.3f}')
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return True


def create_donor_property_distributions_plot():
    """Grid of predicted donor-side property histograms (electronic + Guzik-derived)."""
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'donor_molecule_predictions_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    preferred = ['don_HOMO', 'don_LUMO', 'don_HL_gap', 'don_abs_wavelength',
                 'guz_HOMO_D', 'guz_LUMO_D', 'guz_Ehl_D', 'guz_Eg_D',
                 'guz_donor_PDI', 'guz_donor_Mn_kDa', 'guz_donor_Mw_kDa',
                 'guz_hole_mobility_blend', 'guz_electron_mobility_blend']
    cols_present = [c for c in preferred if c in df.columns]
    out_path = os.path.join(CONFIG['output_dir'], 'donor_property_distributions.png')
    if _grid_hist(df, cols_present, out_path, color=COLORS['donor']):
        print(f"  ✓ Saved: {out_path}")
    else:
        print("  ✗ Donor property distributions skipped (no matching columns)")


def create_acceptor_property_distributions_plot():
    """Grid of predicted acceptor-side property histograms (Eb, Q20, EA, IE, etc.)."""
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'acceptor_molecule_predictions_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    preferred = ['acc_HOMO', 'acc_LUMO', 'acc_HL_gap', 'acc_Diff_L_Lp1',
                 'acc_Eb', 'acc_Q20', 'acc_EA', 'acc_IE', 'acc_abs_wavelength',
                 'guz_HOMO_A', 'guz_LUMO_A', 'guz_Ehl_A', 'guz_Eg_A']
    cols_present = [c for c in preferred if c in df.columns]
    out_path = os.path.join(CONFIG['output_dir'], 'acceptor_property_distributions.png')
    if _grid_hist(df, cols_present, out_path, color=COLORS['acceptor']):
        print(f"  ✓ Saved: {out_path}")
    else:
        print("  ✗ Acceptor property distributions skipped (no matching columns)")


# ============================================================================
# 19: PER-APPROACH MFD / INTDIV / TANGO BAR CHARTS (extends tango_by_approach)
# ============================================================================

def create_per_approach_full_stats_plot():
    """Three-panel bar chart: per-approach MFD, IntDiv and TANGO mean,
    extending the existing tango_by_approach pattern to all three metrics.
    Source: plot_data/approach_full_stats_data.csv (from merge --stage generation)
    """
    csv_path = os.path.join(CONFIG['merged_dir'], 'plot_data', 'approach_full_stats_data.csv')
    if not os.path.exists(csv_path):
        print(f"  ✗ Data not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if 'approach' not in df.columns or 'side' not in df.columns:
        print("  ✗ approach_full_stats_data.csv missing expected 'approach'/'side' columns")
        return

    panels = [('fp_pca_frechet_distance', None, 'Fingerprint-PCA FD', '#455A64'),
              ('internal_diversity', 'internal_diversity_std', 'Internal Diversity', '#1B7F79'),
              ('tango_inspired_mean', 'tango_inspired_std', 'TANGO-inspired score', '#D4602E')]

    for side in ('Donor', 'Acceptor'):
        sub = df[df['side'] == side].copy()
        per_approach = sub[~sub['approach'].astype(str).str.startswith('POOLED')].copy()
        if per_approach.empty:
            print(f"  ✗ No per-approach rows for {side}")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True)
        fig.patch.set_facecolor('white')
        x = np.arange(len(per_approach))
        for ax, (col, errcol, label, color) in zip(axes, panels):
            _style_ax(ax, ylabel=label)
            if col in per_approach.columns:
                vals = pd.to_numeric(per_approach[col], errors='coerce')
                yerr = (pd.to_numeric(per_approach[errcol], errors='coerce')
                        if errcol and errcol in per_approach.columns else None)
                ax.bar(x, vals, color=color, alpha=0.85, edgecolor='black', linewidth=0.6,
                      yerr=yerr, capsize=3)
            else:
                ax.axis('off')
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(per_approach['approach'], rotation=60, ha='right', fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(CONFIG['output_dir'], f'per_approach_full_stats_{side.lower()}.png')
        plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved: {out_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("MANUSCRIPT-READY PLOT GENERATION")
    print("=" * 80)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"\nOutput directory: {CONFIG['output_dir']}\n")

    # ------------------------------------------------------------------------
    # Honest success/failure accounting. A previous version printed
    # "ALL PLOTS GENERATED" unconditionally even when several figures had been
    # skipped for missing input data, which hid real problems. Every figure is
    # now recorded as produced / skipped / errored and the final banner
    # reflects reality. _expected_outputs maps the label to the PNG filename(s)
    # so existence is verified on disk rather than inferred.
    # ------------------------------------------------------------------------
    _RESULTS = []          # (label, status, detail)

    def _run_plot(label, fn, expected, *args, **kwargs):
        """Call a plot function, then verify its expected PNG(s) exist."""
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"  ✗ ERRORED: {label} -> {type(exc).__name__}: {exc}")
            _RESULTS.append((label, "ERRORED", f"{type(exc).__name__}: {exc}"))
            return
        names = [expected] if isinstance(expected, str) else list(expected)
        made = [n for n in names if os.path.exists(os.path.join(CONFIG['output_dir'], n))]
        if len(made) == len(names):
            _RESULTS.append((label, "OK", ", ".join(names)))
        elif made:
            _RESULTS.append((label, "PARTIAL",
                            "made: %s | missing: %s" % (", ".join(made),
                                                        ", ".join(n for n in names if n not in made))))
        else:
            _RESULTS.append((label, "SKIPPED (no input data)", ", ".join(names)))

    # 1. Donor Space Analysis (no percentile planes)
    print("\n[1/18] Donor Space Analysis")
    _run_plot("Donor space analysis", create_space_analysis_plot,
              "donor_space_analysis.png", 'donor')

    # 2. Acceptor Space Analysis (no percentile planes)
    print("\n[2/18] Acceptor Space Analysis")
    _run_plot("Acceptor space analysis", create_space_analysis_plot,
              "acceptor_space_analysis.png", 'acceptor')

    # 3. Device Properties — Method 1 (Ridge + GP)
    print("\n[3/18] Device Properties — Method 1 screening surrogate (Ridge + GP)")
    _run_plot("Device properties (Method 1 screen)", create_device_properties_method_plot,
              "device_properties_distribution_method1_ridge_gp.png",
        'ridge_gp',
        {
            'PCE (%)':               ('surrogate_PCE', COLORS['pce'], 18.0),
            r'$J_{sc}$ (mA cm$^{-2}$)': ('surrogate_Jsc', COLORS['jsc'], 25.0),
            'FF':                    ('surrogate_FF',  COLORS['ff'],  0.8),
            r'$V_{oc}$ (V)':        ('surrogate_Voc', COLORS['voc'], 1.0),
        },
        'device_properties_distribution_method1_ridge_gp.png',
    )

    # ------------------------------------------------------------------------
    # Method 2 (Scharber-type physics estimate) is DELIBERATELY NOT PLOTTED.
    #
    # It was removed from the quantitative blend in GAN_240_final.py because it
    # is not scientifically defensible as a device-property predictor: its
    # PCE/FF/Voc distributions are dominated by hard-coded clipping ceilings
    # (PCE clipped at 21%, FF bounded 0.70-0.85, Voc bounded 0.3-2.0 V) rather
    # than by emergent physics -- ~98% of its PCE values exceeded 18% and its
    # maximum of exactly 21.000% is a code constant, not a discovered optimum.
    # Its mean parameter balance (high Voc ~1.66 V, modest Jsc ~16 mA/cm2) also
    # does not resemble how real record-efficiency OSCs reach their
    # performance. Continuing to publish a standalone figure of it would
    # reintroduce exactly the claim the removal was meant to retract, so the
    # figure is dropped here too. Any residual
    # device_properties_distribution_method2_physics_data.csv in plot_data/ is
    # stale output from a pre-removal run and should be ignored/deleted.
    # ------------------------------------------------------------------------

    # 4. Blended Device Properties (Method 1 screen only)
    print("\n[4/18] Blended Device Properties (Method 1 screen only)")
    _run_plot("Device properties (blended, Method 1 only)",
              create_blended_device_properties_plot,
              "device_properties_distribution.png")

    # 5. TANGO Distribution
    print("\n[5/18] TANGO-inspired Synthesizability Distribution")
    _run_plot("TANGO-inspired heuristic distribution",
              create_tango_distribution_plot, "tango_distribution.png")

    # 6. Approach Comparison
    print("\n[6/18] Approach Comparison")
    _run_plot("Per-approach molecule counts",
              create_approach_comparison_plot, "approach_comparison.png")

    # 7. GAN Losses — Donor
    print("\n[7/18] GAN Losses — Donor Optimized")
    _run_plot("GAN losses (donor)", create_gan_losses_plot,
              "gan_losses_donor_optimized.png", 'donor_optimized')

    # 8. GAN Losses — Acceptor
    print("\n[8/18] GAN Losses — Acceptor Optimized")
    _run_plot("GAN losses (acceptor)", create_gan_losses_plot,
              "gan_losses_acceptor_optimized.png", 'acceptor_optimized')

    # 9. Discriminator Accuracy — Donor
    print("\n[9/18] Discriminator Accuracy — Donor Optimized")
    _run_plot("Discriminator accuracy (donor)", create_discriminator_accuracy_plot,
              "discriminator_accuracy_donor_optimized.png", 'donor_optimized')

    # 10. Discriminator Accuracy — Acceptor
    print("\n[10/18] Discriminator Accuracy — Acceptor Optimized")
    _run_plot("Discriminator accuracy (acceptor)", create_discriminator_accuracy_plot,
              "discriminator_accuracy_acceptor_optimized.png", 'acceptor_optimized')

    # 11. MOSES quartet (Validity/Uniqueness/Novelty/IntDiv), pooled + per-approach
    print("\n[11/18] MOSES Quartet (Validity/Uniqueness/Novelty/IntDiv)")
    _run_plot("MOSES quartet", create_moses_quartet_plot,
              ["moses_quartet_donor.png", "moses_quartet_acceptor.png"])

    # 12. Three-way distance comparison (FP-PCA-FD vs PFD vs real FCD)
    print("\n[12/18] Distance Metric Comparison (FP-PCA-FD / PFD / FCD)")
    _run_plot("Distance metric comparison", create_distance_comparison_plot,
              "distance_metric_comparison.png")

    # 13. SAScore distribution
    print("\n[13/18] SAScore Distribution")
    _run_plot("SAScore distribution", create_sascore_distribution_plot,
              "sascore_distribution.png")

    # 14. E_loss vs PCE (training vs generated)
    print("\n[14/18] E_loss vs PCE (training vs generated)")
    _run_plot("E_loss vs PCE", create_eloss_vs_pce_plot, "eloss_vs_pce.png")

    # 15. Method 1 disjoint-CV held-out R^2
    print("\n[15/18] Method-1 Disjoint-CV Held-out R^2")
    _run_plot("Method-1 disjoint-CV held-out R^2", create_method1_disjoint_cv_plot,
              "method1_disjoint_cv_r2.png")

    # 16. Donor property distributions
    print("\n[16/18] Donor Property Distributions")
    _run_plot("Donor property distributions", create_donor_property_distributions_plot,
              "donor_property_distributions.png")

    # 17. Acceptor property distributions
    print("\n[17/18] Acceptor Property Distributions")
    _run_plot("Acceptor property distributions", create_acceptor_property_distributions_plot,
              "acceptor_property_distributions.png")

    # 18. Per-approach FP-PCA-FD / IntDiv / TANGO
    print("\n[18/18] Per-Approach FP-PCA-FD / IntDiv / TANGO")
    _run_plot("Per-approach distance/diversity/TANGO", create_per_approach_full_stats_plot,
              ["per_approach_full_stats_donor.png", "per_approach_full_stats_acceptor.png"])

    # ------------------------------------------------------------------------
    # HONEST FINAL SUMMARY
    # ------------------------------------------------------------------------
    n_ok = sum(1 for _, s, _ in _RESULTS if s == "OK")
    n_partial = sum(1 for _, s, _ in _RESULTS if s == "PARTIAL")
    n_skipped = sum(1 for _, s, _ in _RESULTS if s.startswith("SKIPPED"))
    n_error = sum(1 for _, s, _ in _RESULTS if s == "ERRORED")

    print("\n" + "=" * 80)
    print("PLOT GENERATION SUMMARY")
    print("=" * 80)
    print(f"  Produced fully : {n_ok}/{len(_RESULTS)}")
    print(f"  Partial        : {n_partial}")
    print(f"  Skipped        : {n_skipped}   (input CSV absent -- see per-figure note above)")
    print(f"  Errored        : {n_error}")
    print(f"\nOutput directory: {CONFIG['output_dir']}")

    if n_partial or n_skipped or n_error:
        print("\n" + "-" * 80)
        print("NOT FULLY PRODUCED -- these need their input data before they can be made:")
        print("-" * 80)
        for label, status, detail in _RESULTS:
            if status != "OK":
                print(f"  [{status}] {label}")
                print(f"           -> {detail}")
        print("-" * 80)
        print("Common causes:")
        print("  * eloss_vs_pce.png            : DMPNN.csv on the cluster lacks the E_loss")
        print("                                  column (stale 59-column copy; the current")
        print("                                  file has 61 columns incl. V_OC_sim + E_loss).")
        print("  * method1_disjoint_cv_r2.png  : results/method1_disjoint_cv_report.csv was")
        print("                                  not copied into merged/results/.")
        print("  * per_approach_full_stats_*   : plot_data/approach_full_stats_data.csv is")
        print("                                  produced by")
        print("                                  'merge_rank_outputs_final.py --stage generation'")
        print("                                  (or --recompute-pooled-stats-only), which must")
        print("                                  run before plotting.")
        print("  * donor/acceptor property     : Guzik_data.csv absent on the cluster means")
        print("    distributions (thin)          every guz_* target is missing from the grid.")
    else:
        print("\n✓ Every figure was produced successfully.")

    print("=" * 80)
