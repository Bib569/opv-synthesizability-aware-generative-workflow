#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
_common_style.py
=============================================================================
Shared style, colours, and small helpers used by every single-figure script
in this folder.  This is the ONLY file the 12 figure scripts import, and it is
identical in behaviour to the style block of the original
plot_all_12_manuscript_figures.py, so each figure is drawn in the EXACT SAME
publication format that already exists.

Keep this file in the SAME folder as the 12 figure scripts.  It contains no
figure logic of its own; it only defines the global Matplotlib style, the
colour palette, the axis-styling helpers, panel labels, a CLI-argument helper,
and a small file-existence helper.
=============================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE  (identical to plot_all_12_manuscript_figures.py)
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Nimbus Roman No9 L',
                               'Liberation Serif', 'STIXGeneral']
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

DPI = 300

COLORS = {
    'donor': '#1B7F79', 'acceptor': '#D4602E',
    'd_loss': '#2C5F8A', 'g_loss': '#D94F30', 'equilibrium': '#3A8C3F',
    'accuracy': '#2C5F8A', 'target': '#3A8C3F',
    'pce': '#C62828', 'jsc': '#1565C0', 'ff': '#2E7D32', 'voc': '#E65100',
    'threshold': '#C62828', 'mean_line': '#212121',
}


def style_ax(ax, xlabel='', ylabel='', fontsize=13):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_visible(True)
    ax.tick_params(width=2.0, length=6, labelsize=11, direction='in', top=True, right=True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    ax.grid(False)


def style_ax3d(ax, tick_labelsize=10):
    ax.tick_params(labelsize=tick_labelsize)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_linewidth(1.5)
    ax.yaxis.pane.set_linewidth(1.5)
    ax.zaxis.pane.set_linewidth(1.5)


def panel_label(ax, label, outside=False):
    if outside:
        ax.text(-0.06, 1.01, label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    else:
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=15,
                fontweight='bold', va='top', ha='left')


def panel_label_3d(ax, label):
    ax.text2D(0.0, 0.94, label, transform=ax.transAxes, fontsize=15,
              fontweight='bold', va='top', ha='left')


def need(data_dir, name):
    """Return Path to a required input file or None if it is absent."""
    p = data_dir / name
    return p if p.exists() else None


def parse_common_args(description):
    """Standard --data-dir / --outdir handling shared by every figure script.

    --data-dir defaults to <repo root>/data/figure_inputs.
    --outdir   defaults to <repo root>/results/regenerated_figures.
    The caller passes its own __file__ so the default data dir is correct.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--data-dir', type=Path, default=None,
                    help="Folder containing the input CSV(s) (default: <repo root>/data/figure_inputs).")
    ap.add_argument('--outdir', type=Path, default=None,
                    help="Output folder (default: <repo root>/results/regenerated_figures).")
    ap.add_argument('--frechet-splits', type=int, default=200,
                    help="Number of split-half draws for the frechet baseline (figure 10 only).")
    return ap.parse_args()


def resolve_dirs(args, script_file):
    repo_root = Path(script_file).resolve().parents[2]   # scripts/figures/<script>.py -> repo root
    data_dir = (args.data_dir or (repo_root / 'data' / 'figure_inputs')).resolve()
    outdir = (args.outdir or (repo_root / 'results' / 'regenerated_figures')).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return data_dir, outdir
