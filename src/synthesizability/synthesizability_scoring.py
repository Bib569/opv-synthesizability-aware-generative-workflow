#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Pipeline stage 3: SAScore / RAScore / combined synthesizability scoring.
# Run from the flat run directory built by `sh hpc/setup_run_dir.sh` (see docs/PIPELINE.md);
# the script expects DMPNN.csv / Guzik_data.csv and its sibling scripts in that directory.
# Launched by hpc/sub_synth_scoring_240.sh. Needs sascorer.py and fpscores.pkl.gz (RDKit Contrib/SA_Score;
# copied by setup_run_dir.sh), the optional `rascore` package, and the stage-1 pooled pair table
# (--pairs-input, about 7.4 GB, NOT in the repository: produced by stage 1 or on request from the authors).
# NOTE: if `rascore` cannot be imported the script silently falls back to a heuristic RAScore; the log line
# 'RAScore backend' (see results/synthesizability/output_metrics_synthesizability.txt) states which was used.
# The scored libraries used in the paper are in data/generated_libraries/*_synthesizability_scored.csv.
# -----------------------------------------------------------------------------
"""
Synthesizability Scoring Pipeline for Generated OPV Molecules
==============================================================

Implements a TANGO-inspired multi-objective scoring framework that combines:
  1. SAScore (Synthetic Accessibility Score) - Ertl & Schuffenhauer complexity estimate
  2. RAScore (Retrosynthetic Accessibility Score) - ML-based retrosynthesis feasibility

Methodology adapted from:
  Genheden et al. "TANGO: direct optimization of constrained synthesizability
  for generative molecular design" Nature Computational Science (2026).
  DOI: 10.1038/s43588-026-00959-1
  Content was rephrased for compliance with licensing restrictions.

Supports distributed execution via MPI (mpi4py) across N ranks:
  - Each rank scores its shard of unique molecules
  - Rank 0 merges all shards and produces plots/reports/metrics

Outputs:
  - Scored CSV with SAScore, RAScore, combined score, and rank
  - Manuscript-quality plots (SAScore & RAScore distributions for donor & acceptor)
  - Plot data CSVs (for manual re-plotting)
  - Analysis reports (donor & acceptor, per-score)
  - Output metrics summary

Usage (serial):
    python synthesizability_scoring.py --pairs-input generated_molecules_with_predictions.csv
    python synthesizability_scoring.py --input donor_preds.csv --mol-type donor

Usage (distributed, 240 processes):
    srun --ntasks=240 python synthesizability_scoring.py --distributed \
        --pairs-input generated_molecules_with_predictions.csv \
        --outdir synthesizability_output

Requirements:
    pip install rascore rdkit-pypi pandas numpy matplotlib mpi4py
    + sascorer.py and fpscores.pkl.gz in the working directory
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Suppress RDKit and general warnings ────────────────────────────────────────
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ── Download sascorer.py if not exists ─────────────────────────────────────────
sascorer_path = "sascorer.py"
# if not os.path.exists(sascorer_path):
#     url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py"
#     urllib.request.urlretrieve(url, sascorer_path)

# ── Download fpscores.pkl.gz if not exists ─────────────────────────────────────
fpscores_path = "fpscores.pkl.gz"
# if not os.path.exists(fpscores_path):
#     url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
#     urllib.request.urlretrieve(url, fpscores_path)


# ══════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path, rank: int = 0, world_size: int = 1) -> logging.Logger:
    """Configure a single unified logger writing to file + console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = output_dir / "synthesizability_scoring_progress.log"

    logger = logging.getLogger("SynthScore")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_prefix = f"%(asctime)s - rank={rank:03d} - %(levelname)s - %(message)s"

    # File handler (DEBUG level — captures everything)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt_prefix, datefmt="%Y-%m-%d %H:%M:%S"))

    # Console handler (INFO level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt_prefix, datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════════════════════════
# SAScore COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════════

_SASCORE_BACKEND = None


def _init_sascore(logger: logging.Logger):
    """Attempt to import SAScore and report backend used."""
    global _SASCORE_BACKEND

    # Try RDKit Contrib first
    try:
        from rdkit.Contrib.SA_Score import sascorer as _sascorer
        _SASCORE_BACKEND = "rdkit_contrib"
        logger.info("SAScore backend: rdkit.Contrib.SA_Score.sascorer (VERIFIED)")
        mol = Chem.MolFromSmiles("c1ccccc1")
        test_score = _sascorer.calculateScore(mol)
        logger.info(f"SAScore self-test (benzene): {test_score:.4f} (expected ~1.0)")
        if 0.5 < test_score < 2.5:
            logger.info("SAScore self-test PASSED.")
        else:
            logger.warning(f"SAScore self-test returned unusual value: {test_score}")
        return
    except ImportError as e:
        logger.debug(f"rdkit.Contrib.SA_Score not available: {e}")

    # Try local sascorer.py
    try:
        sys.path.insert(0, os.getcwd())
        import sascorer as _local_sascorer
        _SASCORE_BACKEND = "local_sascorer"
        logger.info("SAScore backend: local sascorer.py (VERIFIED)")
        mol = Chem.MolFromSmiles("c1ccccc1")
        test_score = _local_sascorer.calculateScore(mol)
        logger.info(f"SAScore self-test (benzene): {test_score:.4f} (expected ~1.0)")
        if 0.5 < test_score < 2.5:
            logger.info("SAScore self-test PASSED.")
        else:
            logger.warning(f"SAScore self-test returned unusual value: {test_score}")
        return
    except ImportError as e:
        logger.debug(f"Local sascorer.py not available: {e}")

    # Fallback heuristic
    _SASCORE_BACKEND = "heuristic"
    logger.warning("SAScore backend: HEURISTIC FALLBACK (no proper SAScore package found!)")
    logger.warning("  Install sascorer.py + fpscores.pkl.gz in working directory for accurate scoring.")


def compute_sascore(mol, logger: logging.Logger = None) -> float:
    """Compute SAScore (1=easy, 10=hard)."""
    global _SASCORE_BACKEND

    if _SASCORE_BACKEND == "rdkit_contrib":
        try:
            from rdkit.Contrib.SA_Score import sascorer
            return sascorer.calculateScore(mol)
        except Exception as e:
            if logger:
                logger.error(f"SAScore rdkit_contrib failed: {e}")
            return np.nan

    elif _SASCORE_BACKEND == "local_sascorer":
        try:
            from sascorer import calculateScore
            return calculateScore(mol)
        except Exception as e:
            if logger:
                logger.error(f"SAScore local_sascorer failed: {e}")
            return np.nan

    else:
        return _simplified_sascore(mol)


def _simplified_sascore(mol) -> float:
    """Simplified SAScore heuristic fallback."""
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_hetero_types = len(set(atoms) - {6, 1})
    complexity = (
        0.5 * n_rings + 0.3 * n_stereo + 0.1 * n_hetero_types +
        0.05 * max(0, n_heavy - 30) + 0.02 * n_rotatable
    )
    return round(1.0 + min(9.0, complexity), 4)


# ══════════════════════════════════════════════════════════════════════════════════
# RAScore COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════════

_RASCORE_BACKEND = None
_RASCORE_SCORER = None


def _init_rascore(logger: logging.Logger):
    """Attempt to import RAScore and report backend used."""
    global _RASCORE_BACKEND, _RASCORE_SCORER

    try:
        from rascore import RAscore_NN
        _RASCORE_SCORER = RAscore_NN.RAScorerNN()
        _RASCORE_BACKEND = "rascore_nn"
        logger.info("RAScore backend: rascore.RAscore_NN.RAScorerNN (VERIFIED)")
        test_score = _RASCORE_SCORER.predict("c1ccccc1")
        logger.info(f"RAScore self-test (benzene): {test_score:.4f} (expected ~0.99)")
        if test_score > 0.8:
            logger.info("RAScore self-test PASSED.")
        else:
            logger.warning(f"RAScore self-test returned unusual value: {test_score}")
        return
    except ImportError as e:
        logger.debug(f"rascore.RAscore_NN not available: {e}")
    except Exception as e:
        logger.debug(f"rascore.RAscore_NN failed to initialize: {e}")

    try:
        from rascore import RAscore_XGB
        _RASCORE_SCORER = RAscore_XGB.RAScorerXGB()
        _RASCORE_BACKEND = "rascore_xgb"
        logger.info("RAScore backend: rascore.RAscore_XGB.RAScorerXGB (VERIFIED)")
        test_score = _RASCORE_SCORER.predict("c1ccccc1")
        logger.info(f"RAScore self-test (benzene): {test_score:.4f} (expected ~0.99)")
        if test_score > 0.8:
            logger.info("RAScore self-test PASSED.")
        else:
            logger.warning(f"RAScore self-test returned unusual value: {test_score}")
        return
    except ImportError as e:
        logger.debug(f"rascore.RAscore_XGB not available: {e}")
    except Exception as e:
        logger.debug(f"rascore.RAscore_XGB failed to initialize: {e}")

    _RASCORE_BACKEND = "heuristic"
    _RASCORE_SCORER = None
    logger.warning("RAScore backend: HEURISTIC FALLBACK (no rascore package found!)")
    logger.warning("  Install rascore for accurate scoring: pip install rascore")


def compute_rascore(smiles: str, logger: logging.Logger = None) -> float:
    """Compute RAScore (0=hard, 1=easy)."""
    global _RASCORE_BACKEND, _RASCORE_SCORER

    if _RASCORE_BACKEND in ("rascore_nn", "rascore_xgb"):
        try:
            return float(_RASCORE_SCORER.predict(smiles))
        except Exception as e:
            if logger:
                logger.error(f"RAScore prediction failed for SMILES={smiles[:60]}: {e}")
            return np.nan
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return _simplified_rascore(mol)


def _simplified_rascore(mol) -> float:
    """Simplified RAScore heuristic."""
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    score = 1.0
    score -= 0.005 * max(0, n_heavy - 20)
    score -= 0.03 * max(0, n_rings - 3)
    score -= 0.1 * n_stereo
    score -= 0.002 * max(0, n_rotatable - 5)
    return max(0.0, min(1.0, round(score, 4)))


# ══════════════════════════════════════════════════════════════════════════════════
# COMBINED SCORING
# ══════════════════════════════════════════════════════════════════════════════════

def combined_synthesizability(sascore: float, rascore: float,
                              w_sa: float = 0.5, w_ra: float = 0.5) -> float:
    """Combined synthesizability score (0-1, higher = more synthesizable)."""
    if np.isnan(sascore) or np.isnan(rascore):
        return np.nan
    sa_norm = max(0.0, 1.0 - (sascore - 1.0) / 9.0)
    combined = w_sa * sa_norm + w_ra * rascore
    return round(combined, 4)


# ══════════════════════════════════════════════════════════════════════════════════
# SMILES COLUMN DETECTION
# ══════════════════════════════════════════════════════════════════════════════════

def detect_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the SMILES column name."""
    candidates = ["smiles", "SMILES", "Smiles", "smi", "SMI",
                  "acceptor(SMILES)", "donor(SMILES)", "canonical_smiles",
                  "molecule", "mol_smiles", "generated_smiles"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    return None


# ══════════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE (per-rank shard)
# ══════════════════════════════════════════════════════════════════════════════════

def score_molecules_shard(smiles_list: list, mol_type: str,
                          w_sa: float, w_ra: float,
                          threshold_sa: float, threshold_ra: float,
                          logger: logging.Logger) -> pd.DataFrame:
    """Score a shard of unique SMILES. Returns a DataFrame with scores."""

    n_total = len(smiles_list)
    logger.info(f"{'='*70}")
    logger.info(f"SCORING {mol_type.upper()} SHARD ({n_total} molecules)")
    logger.info(f"{'='*70}")

    results = []
    n_invalid = 0
    n_sa_errors = 0
    n_ra_errors = 0

    t_start = time.time()
    report_interval = max(1, n_total // 20)

    for i, smiles in enumerate(smiles_list):
        if (i + 1) % report_interval == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Progress: {i+1}/{n_total} ({100*(i+1)/n_total:.1f}%) "
                        f"[{elapsed:.1f}s, {rate:.1f} mol/s]")

        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            results.append({'smiles': smiles, 'SAScore': np.nan, 'RAScore': np.nan,
                            'combined_synthesizability': np.nan, 'valid': False})
            n_invalid += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({'smiles': smiles, 'SAScore': np.nan, 'RAScore': np.nan,
                            'combined_synthesizability': np.nan, 'valid': False})
            n_invalid += 1
            logger.debug(f"  [idx={i}] RDKit parse failed: {smiles[:80]}")
            continue

        sa = compute_sascore(mol, logger)
        if np.isnan(sa):
            n_sa_errors += 1

        ra = compute_rascore(smiles, logger)
        if np.isnan(ra):
            n_ra_errors += 1

        comb = combined_synthesizability(sa, ra, w_sa=w_sa, w_ra=w_ra)

        results.append({
            'smiles': smiles,
            'SAScore': sa,
            'RAScore': ra,
            'combined_synthesizability': comb,
            'valid': True
        })

    elapsed_total = time.time() - t_start
    n_valid = sum(1 for r in results if r['valid'])

    logger.info(f"  Shard scoring complete in {elapsed_total:.1f}s")
    logger.info(f"  Valid: {n_valid}/{n_total} | Invalid: {n_invalid} | "
                f"SA errors: {n_sa_errors} | RA errors: {n_ra_errors}")

    df = pd.DataFrame(results)
    df['mol_type'] = mol_type
    df['synthesizable_SA'] = df['SAScore'] <= threshold_sa
    df['synthesizable_RA'] = df['RAScore'] >= threshold_ra
    df['synthesizable_both'] = df['synthesizable_SA'] & df['synthesizable_RA']

    return df


# ══════════════════════════════════════════════════════════════════════════════════
# MANUSCRIPT-QUALITY PLOTS
# ══════════════════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
    'hist_donor': '#1565C0', 'hist_acc': '#D84315',
}


def _style_ax(ax, xlabel='', ylabel='', fontsize=13):
    """Apply consistent publication styling."""
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


def generate_plots(df: pd.DataFrame, mol_type: str, output_dir: Path,
                   plot_data_dir: Path, threshold_sa: float,
                   threshold_ra: float, logger: logging.Logger):
    """Generate manuscript-quality plots for a single molecule type."""
    import matplotlib.pyplot as plt

    logger.info(f"  Generating plots for {mol_type}...")
    valid_df = df[df["SAScore"].notna() & df["RAScore"].notna()].copy()

    if len(valid_df) == 0:
        logger.warning(f"  No valid scored molecules for {mol_type} — skipping plots.")
        return

    color = COLORS['hist_donor'] if mol_type == 'donor' else COLORS['hist_acc']

    # ── Plot 1: SAScore Distribution ──────────────────────────────────────────
    logger.info(f"    [1/6] SAScore distribution ({mol_type})...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_df["SAScore"], bins=50, color=color, alpha=0.75,
            edgecolor='black', linewidth=0.5)
    ax.axvline(valid_df["SAScore"].mean(), color=COLORS['mean_line'],
               linestyle='--', linewidth=2.0, label=f'Mean = {valid_df["SAScore"].mean():.2f}')
    ax.axvline(valid_df["SAScore"].median(), color=COLORS['combined'],
               linestyle='-.', linewidth=2.0, label=f'Median = {valid_df["SAScore"].median():.2f}')
    ax.axvline(threshold_sa, color=COLORS['threshold'],
               linestyle=':', linewidth=2.5, label=f'Threshold = {threshold_sa}')
    _style_ax(ax, xlabel='SAScore (1=easy, 10=hard)', ylabel='Count')
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    fig.savefig(output_dir / f"sascore_distribution_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: sascore_distribution_{mol_type}.png")
    pd.DataFrame({'SAScore': valid_df["SAScore"].values}).to_csv(
        plot_data_dir / f"sascore_distribution_{mol_type}_data.csv", index=False)

    # ── Plot 2: RAScore Distribution ──────────────────────────────────────────
    logger.info(f"    [2/6] RAScore distribution ({mol_type})...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_df["RAScore"], bins=50, color=color, alpha=0.75,
            edgecolor='black', linewidth=0.5)
    ax.axvline(valid_df["RAScore"].mean(), color=COLORS['mean_line'],
               linestyle='--', linewidth=2.0, label=f'Mean = {valid_df["RAScore"].mean():.3f}')
    ax.axvline(valid_df["RAScore"].median(), color=COLORS['combined'],
               linestyle='-.', linewidth=2.0, label=f'Median = {valid_df["RAScore"].median():.3f}')
    ax.axvline(threshold_ra, color=COLORS['threshold'],
               linestyle=':', linewidth=2.5, label=f'Threshold = {threshold_ra}')
    _style_ax(ax, xlabel='RAScore (0=hard, 1=easy)', ylabel='Count')
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    fig.savefig(output_dir / f"rascore_distribution_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: rascore_distribution_{mol_type}.png")
    pd.DataFrame({'RAScore': valid_df["RAScore"].values}).to_csv(
        plot_data_dir / f"rascore_distribution_{mol_type}_data.csv", index=False)

    # ── Plot 3: Combined Synthesizability Distribution ────────────────────────
    logger.info(f"    [3/6] Combined synthesizability ({mol_type})...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_df["combined_synthesizability"], bins=50,
            color=COLORS['combined'], alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.axvline(valid_df["combined_synthesizability"].mean(), color=COLORS['mean_line'],
               linestyle='--', linewidth=2.0,
               label=f'Mean = {valid_df["combined_synthesizability"].mean():.3f}')
    ax.axvline(valid_df["combined_synthesizability"].median(), color=COLORS['rascore'],
               linestyle='-.', linewidth=2.0,
               label=f'Median = {valid_df["combined_synthesizability"].median():.3f}')
    _style_ax(ax, xlabel='Combined Synthesizability (0=hard, 1=easy)', ylabel='Count')
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    fig.savefig(output_dir / f"combined_synthesizability_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: combined_synthesizability_{mol_type}.png")
    pd.DataFrame({'combined_synthesizability': valid_df["combined_synthesizability"].values}).to_csv(
        plot_data_dir / f"combined_synthesizability_{mol_type}_data.csv", index=False)

    # ── Plot 4: SAScore vs RAScore Scatter ────────────────────────────────────
    logger.info(f"    [4/6] SAScore vs RAScore scatter ({mol_type})...")
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = valid_df if len(valid_df) <= 20000 else valid_df.sample(20000, random_state=42)
    scatter = ax.scatter(plot_df["SAScore"], plot_df["RAScore"],
                         c=plot_df["combined_synthesizability"],
                         cmap='RdYlGn', alpha=0.4, s=8, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label('Combined Score', fontsize=12, fontweight='bold')
    ax.axvline(threshold_sa, color=COLORS['threshold'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'SA threshold={threshold_sa}')
    ax.axhline(threshold_ra, color=COLORS['threshold'], linestyle=':',
               linewidth=1.5, alpha=0.8, label=f'RA threshold={threshold_ra}')
    n_both = valid_df["synthesizable_both"].sum()
    pct = 100 * n_both / len(valid_df)
    ax.text(0.03, 0.97, f'Pass both: {n_both} ({pct:.1f}%)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    _style_ax(ax, xlabel='SAScore (1=easy, 10=hard)', ylabel='RAScore (0=hard, 1=easy)')
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    fig.savefig(output_dir / f"sascore_vs_rascore_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: sascore_vs_rascore_{mol_type}.png")
    pd.DataFrame({'SAScore': plot_df["SAScore"].values, 'RAScore': plot_df["RAScore"].values,
                  'combined_synthesizability': plot_df["combined_synthesizability"].values
    }).to_csv(plot_data_dir / f"sascore_vs_rascore_{mol_type}_data.csv", index=False)

    # ── Plot 5: Box plots ─────────────────────────────────────────────────────
    logger.info(f"    [5/6] Box plots ({mol_type})...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].boxplot(valid_df["SAScore"].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))
    _style_ax(axes[0], xlabel=mol_type.capitalize(), ylabel='SAScore')
    axes[0].set_xticklabels([mol_type.capitalize()])
    axes[1].boxplot(valid_df["RAScore"].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))
    _style_ax(axes[1], xlabel=mol_type.capitalize(), ylabel='RAScore')
    axes[1].set_xticklabels([mol_type.capitalize()])
    plt.tight_layout()
    fig.savefig(output_dir / f"boxplots_synth_scores_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: boxplots_synth_scores_{mol_type}.png")

    # ── Plot 6: CDF ──────────────────────────────────────────────────────────
    logger.info(f"    [6/6] CDF plots ({mol_type})...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sorted_sa = np.sort(valid_df["SAScore"].dropna())
    cdf_sa = np.arange(1, len(sorted_sa) + 1) / len(sorted_sa)
    axes[0].plot(sorted_sa, cdf_sa, color=COLORS['sascore'], linewidth=2.0)
    axes[0].axvline(threshold_sa, color=COLORS['threshold'], linestyle='--',
                    linewidth=1.5, label=f'Threshold={threshold_sa}')
    _style_ax(axes[0], xlabel='SAScore', ylabel='Cumulative Fraction')
    axes[0].legend(fontsize=10)
    sorted_ra = np.sort(valid_df["RAScore"].dropna())
    cdf_ra = np.arange(1, len(sorted_ra) + 1) / len(sorted_ra)
    axes[1].plot(sorted_ra, cdf_ra, color=COLORS['rascore'], linewidth=2.0)
    axes[1].axvline(threshold_ra, color=COLORS['threshold'], linestyle='--',
                    linewidth=1.5, label=f'Threshold={threshold_ra}')
    _style_ax(axes[1], xlabel='RAScore', ylabel='Cumulative Fraction')
    axes[1].legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / f"cdf_synth_scores_{mol_type}.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"    ✓ Saved: cdf_synth_scores_{mol_type}.png")
    pd.DataFrame({'SAScore_sorted': sorted_sa, 'SAScore_CDF': cdf_sa}).to_csv(
        plot_data_dir / f"cdf_sascore_{mol_type}_data.csv", index=False)
    pd.DataFrame({'RAScore_sorted': sorted_ra, 'RAScore_CDF': cdf_ra}).to_csv(
        plot_data_dir / f"cdf_rascore_{mol_type}_data.csv", index=False)

    logger.info(f"  All 6 plots generated for {mol_type}.")


def generate_comparison_plots(donor_df: pd.DataFrame, acceptor_df: pd.DataFrame,
                              output_dir: Path, plot_data_dir: Path,
                              threshold_sa: float, threshold_ra: float,
                              logger: logging.Logger):
    """Generate comparative donor vs acceptor plots."""
    import matplotlib.pyplot as plt

    logger.info("  Generating donor vs acceptor comparison plots...")
    donor_valid = donor_df[donor_df["SAScore"].notna()].copy() if donor_df is not None else pd.DataFrame()
    acc_valid = acceptor_df[acceptor_df["SAScore"].notna()].copy() if acceptor_df is not None else pd.DataFrame()

    if len(donor_valid) == 0 and len(acc_valid) == 0:
        logger.warning("  No valid data for comparison — skipping.")
        return

    # SAScore comparison
    logger.info("    [1/2] SAScore comparison...")
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(donor_valid) > 0:
        ax.hist(donor_valid["SAScore"], bins=50, alpha=0.6, color=COLORS['hist_donor'],
                edgecolor='none', density=True,
                label=f'Donor (n={len(donor_valid)}, mean={donor_valid["SAScore"].mean():.2f})')
    if len(acc_valid) > 0:
        ax.hist(acc_valid["SAScore"], bins=50, alpha=0.6, color=COLORS['hist_acc'],
                edgecolor='none', density=True,
                label=f'Acceptor (n={len(acc_valid)}, mean={acc_valid["SAScore"].mean():.2f})')
    ax.axvline(threshold_sa, color=COLORS['threshold'], linestyle='--', linewidth=2.0,
               label=f'Threshold={threshold_sa}')
    _style_ax(ax, xlabel='SAScore (1=easy, 10=hard)', ylabel='Density')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "sascore_comparison_donor_vs_acceptor.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("    ✓ Saved: sascore_comparison_donor_vs_acceptor.png")

    # RAScore comparison
    logger.info("    [2/2] RAScore comparison...")
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(donor_valid) > 0:
        ax.hist(donor_valid["RAScore"], bins=50, alpha=0.6, color=COLORS['hist_donor'],
                edgecolor='none', density=True,
                label=f'Donor (n={len(donor_valid)}, mean={donor_valid["RAScore"].mean():.3f})')
    if len(acc_valid) > 0:
        ax.hist(acc_valid["RAScore"], bins=50, alpha=0.6, color=COLORS['hist_acc'],
                edgecolor='none', density=True,
                label=f'Acceptor (n={len(acc_valid)}, mean={acc_valid["RAScore"].mean():.3f})')
    ax.axvline(threshold_ra, color=COLORS['threshold'], linestyle='--', linewidth=2.0,
               label=f'Threshold={threshold_ra}')
    _style_ax(ax, xlabel='RAScore (0=hard, 1=easy)', ylabel='Density')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "rascore_comparison_donor_vs_acceptor.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("    ✓ Saved: rascore_comparison_donor_vs_acceptor.png")

    # Save comparison data
    comp_parts = []
    if len(donor_valid) > 0:
        comp_parts.append(pd.DataFrame({'mol_type': 'donor', 'SAScore': donor_valid["SAScore"].values,
                                        'RAScore': donor_valid["RAScore"].values,
                                        'combined': donor_valid["combined_synthesizability"].values}))
    if len(acc_valid) > 0:
        comp_parts.append(pd.DataFrame({'mol_type': 'acceptor', 'SAScore': acc_valid["SAScore"].values,
                                        'RAScore': acc_valid["RAScore"].values,
                                        'combined': acc_valid["combined_synthesizability"].values}))
    if comp_parts:
        pd.concat(comp_parts, ignore_index=True).to_csv(
            plot_data_dir / "synth_scores_comparison_data.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════════
# ANALYSIS REPORT
# ══════════════════════════════════════════════════════════════════════════════════

def write_analysis_report(df: pd.DataFrame, mol_type: str, output_dir: Path,
                          threshold_sa: float, threshold_ra: float,
                          logger: logging.Logger):
    """Write a detailed analysis report for a molecule type."""
    valid_df = df[df["SAScore"].notna() & df["RAScore"].notna()]
    n_valid = len(valid_df)
    if n_valid == 0:
        logger.warning(f"  No valid data for {mol_type} report — skipping.")
        return

    report_path = output_dir / f"{mol_type}_synthesizability_analysis_report.txt"
    logger.info(f"  Writing report: {report_path}")

    sa_mean = valid_df["SAScore"].mean()
    ra_mean = valid_df["RAScore"].mean()
    n_sa_pass = valid_df["synthesizable_SA"].sum()
    n_ra_pass = valid_df["synthesizable_RA"].sum()
    n_both_pass = valid_df["synthesizable_both"].sum()
    pct_sa = 100 * n_sa_pass / n_valid
    pct_ra = 100 * n_ra_pass / n_valid
    pct_both = 100 * n_both_pass / n_valid

    lines = []
    lines.append(f"Detailed Synthesizability Analysis Report for Generated {mol_type.capitalize()} Molecules")
    lines.append("=" * 70)
    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total scored molecules: {n_valid}")
    lines.append(f"- Mean SAScore: {sa_mean:.3f} (lower = easier)")
    lines.append(f"- Mean RAScore: {ra_mean:.3f} (higher = easier)")
    lines.append(f"- Pass SA threshold (<={threshold_sa}): {pct_sa:.1f}%")
    lines.append(f"- Pass RA threshold (>={threshold_ra}): {pct_ra:.1f}%")
    lines.append(f"- Pass BOTH: {pct_both:.1f}%")
    if pct_both > 70:
        lines.append("- Overall: EXCELLENT synthesizability")
    elif pct_both > 40:
        lines.append("- Overall: GOOD synthesizability")
    elif pct_both > 20:
        lines.append("- Overall: MODERATE synthesizability")
    else:
        lines.append("- Overall: LOW synthesizability")
    lines.append("")

    lines.append("SAScore ANALYSIS (Ertl & Schuffenhauer, J. Cheminform. 2009)")
    lines.append("-" * 40)
    lines.append(f"Mean={valid_df['SAScore'].mean():.4f}  Median={valid_df['SAScore'].median():.4f}  "
                 f"Std={valid_df['SAScore'].std():.4f}")
    lines.append(f"Min={valid_df['SAScore'].min():.4f}  Max={valid_df['SAScore'].max():.4f}")
    lines.append(f"25th={valid_df['SAScore'].quantile(0.25):.4f}  75th={valid_df['SAScore'].quantile(0.75):.4f}")
    lines.append(f"Pass (SA<={threshold_sa}): {n_sa_pass}/{n_valid} ({pct_sa:.1f}%)")
    lines.append("")
    for lo, hi, label in [(1,2,"Very easy (1-2)"),(2,3,"Easy (2-3)"),(3,4,"Moderate (3-4)"),
                           (4,5,"Mod-hard (4-5)"),(5,6,"Hard (5-6)"),(6,8,"Very hard (6-8)"),
                           (8,10.01,"Extremely hard (8-10)")]:
        c = ((valid_df["SAScore"]>=lo)&(valid_df["SAScore"]<hi)).sum()
        lines.append(f"  {label:25s}: {c:>6d} ({100*c/n_valid:.1f}%)")
    lines.append("")

    lines.append("RAScore ANALYSIS (Thakkar et al., Chem. Sci. 2021)")
    lines.append("-" * 40)
    lines.append(f"Mean={valid_df['RAScore'].mean():.4f}  Median={valid_df['RAScore'].median():.4f}  "
                 f"Std={valid_df['RAScore'].std():.4f}")
    lines.append(f"Min={valid_df['RAScore'].min():.4f}  Max={valid_df['RAScore'].max():.4f}")
    lines.append(f"25th={valid_df['RAScore'].quantile(0.25):.4f}  75th={valid_df['RAScore'].quantile(0.75):.4f}")
    lines.append(f"Pass (RA>={threshold_ra}): {n_ra_pass}/{n_valid} ({pct_ra:.1f}%)")
    lines.append("")
    for lo, hi, label in [(0,0.2,"Very hard (0-0.2)"),(0.2,0.4,"Hard (0.2-0.4)"),
                           (0.4,0.5,"Mod-hard (0.4-0.5)"),(0.5,0.6,"Moderate (0.5-0.6)"),
                           (0.6,0.8,"Feasible (0.6-0.8)"),(0.8,1.01,"Easy (0.8-1.0)")]:
        c = ((valid_df["RAScore"]>=lo)&(valid_df["RAScore"]<hi)).sum()
        lines.append(f"  {label:25s}: {c:>6d} ({100*c/n_valid:.1f}%)")
    lines.append("")

    lines.append("COMBINED SYNTHESIZABILITY")
    lines.append("-" * 40)
    lines.append(f"Mean={valid_df['combined_synthesizability'].mean():.4f}  "
                 f"Median={valid_df['combined_synthesizability'].median():.4f}")
    lines.append(f"Pass BOTH: {n_both_pass}/{n_valid} ({pct_both:.1f}%)")
    lines.append("")

    lines.append("TOP 20 MOST SYNTHESIZABLE")
    lines.append("-" * 40)
    top20 = valid_df.nlargest(20, "combined_synthesizability")
    for rank_i, (_, row) in enumerate(top20.iterrows(), 1):
        smi = str(row.get('smiles', 'N/A'))[:65]
        lines.append(f"  #{rank_i:>2d}  SA={row['SAScore']:.3f}  RA={row['RAScore']:.3f}  "
                     f"Comb={row['combined_synthesizability']:.3f}  {smi}")
    lines.append("")

    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)
    if pct_both > 50:
        lines.append("- Majority synthesizable. Prioritize by combined score for experiments.")
    elif pct_ra < 30:
        lines.append("- Many fail RAScore. Consider retrosynthesis-aware GAN constraints.")
    elif pct_sa < 30:
        lines.append("- Many fail SAScore. Generated scaffolds may be overly complex.")
    lines.append("- Cross-reference with TANGO pair scores for holistic assessment.")
    lines.append("- Note: RAScore trained on ChEMBL is partially out-of-domain for OPV scaffolds.")
    lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"  ✓ Report: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════════
# OUTPUT METRICS
# ══════════════════════════════════════════════════════════════════════════════════

def write_output_metrics(donor_df: pd.DataFrame, acceptor_df: pd.DataFrame,
                         output_dir: Path, threshold_sa: float,
                         threshold_ra: float, logger: logging.Logger):
    """Write consolidated output metrics."""
    metrics_path = output_dir / "output_metrics_synthesizability.txt"
    logger.info(f"  Writing metrics: {metrics_path}")

    lines = []
    lines.append("SYNTHESIZABILITY SCORING METRICS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"SAScore backend: {_SASCORE_BACKEND}")
    lines.append(f"RAScore backend: {_RASCORE_BACKEND}")
    lines.append(f"Thresholds: SAScore <= {threshold_sa}, RAScore >= {threshold_ra}")
    lines.append("")

    for mol_type, df in [("DONORS", donor_df), ("ACCEPTORS", acceptor_df)]:
        if df is None or len(df) == 0:
            continue
        valid = df[df["SAScore"].notna() & df["RAScore"].notna()]
        n_v = len(valid)
        if n_v == 0:
            continue
        lines.append(f"{mol_type}:")
        lines.append(f"  Total unique molecules: {len(df)}")
        lines.append(f"  Successfully scored: {n_v}")
        lines.append(f"  SAScore: mean={valid['SAScore'].mean():.3f}, "
                     f"std={valid['SAScore'].std():.3f}, "
                     f"median={valid['SAScore'].median():.3f}, "
                     f"range=[{valid['SAScore'].min():.3f}, {valid['SAScore'].max():.3f}]")
        lines.append(f"    pass (<={threshold_sa}): {valid['synthesizable_SA'].sum()}/{n_v} "
                     f"({100*valid['synthesizable_SA'].sum()/n_v:.1f}%)")
        lines.append(f"  RAScore: mean={valid['RAScore'].mean():.4f}, "
                     f"std={valid['RAScore'].std():.4f}, "
                     f"median={valid['RAScore'].median():.4f}, "
                     f"range=[{valid['RAScore'].min():.4f}, {valid['RAScore'].max():.4f}]")
        lines.append(f"    pass (>={threshold_ra}): {valid['synthesizable_RA'].sum()}/{n_v} "
                     f"({100*valid['synthesizable_RA'].sum()/n_v:.1f}%)")
        lines.append(f"  Combined: mean={valid['combined_synthesizability'].mean():.4f}, "
                     f"median={valid['combined_synthesizability'].median():.4f}")
        lines.append(f"  Pass BOTH: {valid['synthesizable_both'].sum()}/{n_v} "
                     f"({100*valid['synthesizable_both'].sum()/n_v:.1f}%)")
        lines.append(f"  Tier 1 (SA<=4 AND RA>=0.7): "
                     f"{((valid['SAScore']<=4)&(valid['RAScore']>=0.7)).sum()}")
        lines.append(f"  Tier 2 (SA<={threshold_sa} AND RA>={threshold_ra}): "
                     f"{valid['synthesizable_both'].sum()}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("NOTE: OPV scaffolds are partially out-of-domain for ChEMBL-trained RAScore.")
    lines.append("SAScore may overestimate difficulty for conjugated systems accessible via")
    lines.append("Suzuki/Stille coupling. Use combined scores + TANGO for robust assessment.")
    lines.append("=" * 60)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"  ✓ Metrics: {metrics_path}")


# ══════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (supports --distributed for MPI across 240 ranks)
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Synthesizability scoring pipeline (SAScore + RAScore), "
                    "supports distributed execution via MPI.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input modes
    ap.add_argument("--input", "-i", type=Path, default=None,
                    help="Single input CSV (use with --mol-type)")
    ap.add_argument("--pairs-input", type=Path, default=None,
                    help="Pairs CSV with donor_smiles + acceptor_smiles columns "
                         "(e.g. generated_molecules_with_predictions.csv)")
    ap.add_argument("--donor-input", type=Path, default=None,
                    help="Donor molecules CSV")
    ap.add_argument("--acceptor-input", type=Path, default=None,
                    help="Acceptor molecules CSV")
    ap.add_argument("--mol-type", type=str, default="donor",
                    choices=["donor", "acceptor"],
                    help="Molecule type when using --input (default: donor)")

    # Column specification
    ap.add_argument("--smiles-col", type=str, default=None,
                    help="Name of SMILES column (auto-detected if not given)")
    ap.add_argument("--donor-smiles-col", type=str, default=None,
                    help="SMILES column for donor (default: donor_smiles)")
    ap.add_argument("--acceptor-smiles-col", type=str, default=None,
                    help="SMILES column for acceptor (default: acceptor_smiles)")

    # Scoring parameters
    ap.add_argument("--w-sa", type=float, default=0.5,
                    help="Weight for SAScore (default: 0.5)")
    ap.add_argument("--w-ra", type=float, default=0.5,
                    help="Weight for RAScore (default: 0.5)")
    ap.add_argument("--threshold-sa", type=float, default=6.0,
                    help="SAScore threshold (default: 6.0)")
    ap.add_argument("--threshold-ra", type=float, default=0.5,
                    help="RAScore threshold (default: 0.5)")

    # Distributed
    ap.add_argument("--distributed", action="store_true",
                    help="Enable MPI-distributed scoring across multiple ranks")

    # Output
    ap.add_argument("--outdir", "-o", type=Path, default=None,
                    help="Output directory (default: ./synthesizability_output)")

    args = ap.parse_args()

    # ── MPI setup ─────────────────────────────────────────────────────────────
    rank = 0
    world_size = 1

    if args.distributed:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            world_size = comm.Get_size()
        except ImportError:
            # Fallback: use SLURM environment variables
            rank = int(os.environ.get("SLURM_PROCID", 0))
            world_size = int(os.environ.get("SLURM_NTASKS", 1))
            comm = None
    else:
        comm = None

    # ── Output directory ──────────────────────────────────────────────────────
    if args.outdir is None:
        args.outdir = Path.cwd() / "synthesizability_output"
    output_dir = args.outdir
    results_dir = output_dir / "results"
    rank_dir = output_dir / f"rank_{rank:03d}"
    plot_dir = output_dir / "plots"
    plot_data_dir = output_dir / "plot_data"

    # All ranks create their rank directory; rank 0 creates shared dirs
    os.makedirs(rank_dir, exist_ok=True)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(plot_data_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir if rank == 0 else rank_dir, rank, world_size)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("SYNTHESIZABILITY SCORING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"  Timestamp     : {datetime.now().isoformat()}")
        logger.info(f"  Python        : {sys.version.split()[0]}")
        logger.info(f"  Working dir   : {os.getcwd()}")
        logger.info(f"  Output dir    : {output_dir}")
        logger.info(f"  Distributed   : {args.distributed} (rank {rank}/{world_size})")
        logger.info(f"  Weights       : SA={args.w_sa}, RA={args.w_ra}")
        logger.info(f"  Thresholds    : SA<={args.threshold_sa}, RA>={args.threshold_ra}")
        logger.info("=" * 70)

    # ── Initialize scoring backends (all ranks) ───────────────────────────────
    logger.info("Initializing scoring backends...")
    if os.path.exists(sascorer_path):
        logger.info(f"  sascorer.py: FOUND at {os.path.abspath(sascorer_path)}")
    else:
        logger.warning(f"  sascorer.py: NOT FOUND in {os.getcwd()}")
    if os.path.exists(fpscores_path):
        logger.info(f"  fpscores.pkl.gz: FOUND at {os.path.abspath(fpscores_path)}")
    else:
        logger.warning(f"  fpscores.pkl.gz: NOT FOUND in {os.getcwd()}")

    _init_sascore(logger)
    _init_rascore(logger)

    # ── Load input and extract unique SMILES (all ranks read) ─────────────────
    logger.info("Loading input data...")

    donor_smiles_all = None
    acceptor_smiles_all = None

    if args.pairs_input:
        if not args.pairs_input.exists():
            logger.error(f"Pairs input not found: {args.pairs_input}")
            return 1
        pairs_df = pd.read_csv(args.pairs_input)
        logger.info(f"  Loaded pairs CSV: {len(pairs_df)} rows")

        d_col = args.donor_smiles_col or "donor_smiles"
        a_col = args.acceptor_smiles_col or "acceptor_smiles"
        # Auto-detect if needed
        if d_col not in pairs_df.columns:
            for c in pairs_df.columns:
                if "donor" in c.lower() and "smiles" in c.lower():
                    d_col = c
                    break
        if a_col not in pairs_df.columns:
            for c in pairs_df.columns:
                if "acceptor" in c.lower() and "smiles" in c.lower():
                    a_col = c
                    break

        if d_col not in pairs_df.columns or a_col not in pairs_df.columns:
            logger.error(f"Cannot find SMILES columns. Available: {list(pairs_df.columns)}")
            return 1

        donor_smiles_all = pairs_df[d_col].dropna().unique().tolist()
        acceptor_smiles_all = pairs_df[a_col].dropna().unique().tolist()
        logger.info(f"  Unique donors: {len(donor_smiles_all)}")
        logger.info(f"  Unique acceptors: {len(acceptor_smiles_all)}")
        del pairs_df  # free memory

    elif args.donor_input and args.acceptor_input:
        if not args.donor_input.exists() or not args.acceptor_input.exists():
            logger.error("Donor or acceptor input file not found.")
            return 1
        d_df = pd.read_csv(args.donor_input)
        a_df = pd.read_csv(args.acceptor_input)
        d_col = args.donor_smiles_col or args.smiles_col or detect_smiles_column(d_df)
        a_col = args.acceptor_smiles_col or args.smiles_col or detect_smiles_column(a_df)
        if d_col is None or a_col is None:
            logger.error("Cannot detect SMILES columns. Use --donor-smiles-col / --acceptor-smiles-col.")
            return 1
        donor_smiles_all = d_df[d_col].dropna().unique().tolist()
        acceptor_smiles_all = a_df[a_col].dropna().unique().tolist()
        logger.info(f"  Unique donors: {len(donor_smiles_all)} | Unique acceptors: {len(acceptor_smiles_all)}")
        del d_df, a_df

    elif args.input:
        if not args.input.exists():
            logger.error(f"Input not found: {args.input}")
            return 1
        df = pd.read_csv(args.input)
        smiles_col = args.smiles_col or detect_smiles_column(df)
        if smiles_col is None or smiles_col not in df.columns:
            logger.error(f"Cannot find SMILES column. Available: {list(df.columns)}")
            return 1
        all_smiles = df[smiles_col].dropna().unique().tolist()
        if args.mol_type == "donor":
            donor_smiles_all = all_smiles
        else:
            acceptor_smiles_all = all_smiles
        del df
    else:
        logger.error("No input. Use --pairs-input, --input, or --donor-input + --acceptor-input.")
        return 1

    # ── Distribute work across ranks ──────────────────────────────────────────
    t_start = time.time()

    # Combine all molecules into a single work pool for maximum efficiency
    # Tag each molecule with its type so we can split after scoring
    work_pool = []
    if donor_smiles_all:
        for s in donor_smiles_all:
            work_pool.append((s, "donor"))
    if acceptor_smiles_all:
        for s in acceptor_smiles_all:
            work_pool.append((s, "acceptor"))

    total_molecules = len(work_pool)
    logger.info(f"  Total molecules to score: {total_molecules}")

    # Shard assignment: each rank gets a contiguous chunk
    chunk_size = total_molecules // world_size
    remainder = total_molecules % world_size
    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
    my_shard = work_pool[start_idx:end_idx]

    logger.info(f"  Rank {rank}: scoring indices [{start_idx}, {end_idx}) "
                f"= {len(my_shard)} molecules")

    # Score this rank's shard
    my_donor_smiles = [s for s, t in my_shard if t == "donor"]
    my_acceptor_smiles = [s for s, t in my_shard if t == "acceptor"]

    donor_shard_df = None
    acceptor_shard_df = None

    if my_donor_smiles:
        logger.info(f"  Scoring {len(my_donor_smiles)} donor molecules...")
        donor_shard_df = score_molecules_shard(
            my_donor_smiles, "donor", args.w_sa, args.w_ra,
            args.threshold_sa, args.threshold_ra, logger)

    if my_acceptor_smiles:
        logger.info(f"  Scoring {len(my_acceptor_smiles)} acceptor molecules...")
        acceptor_shard_df = score_molecules_shard(
            my_acceptor_smiles, "acceptor", args.w_sa, args.w_ra,
            args.threshold_sa, args.threshold_ra, logger)

    # Save rank shard outputs
    if donor_shard_df is not None and len(donor_shard_df) > 0:
        donor_shard_df.to_csv(rank_dir / "donor_scores.csv", index=False)
        logger.info(f"  ✓ Wrote: {rank_dir / 'donor_scores.csv'} ({len(donor_shard_df)} rows)")
    if acceptor_shard_df is not None and len(acceptor_shard_df) > 0:
        acceptor_shard_df.to_csv(rank_dir / "acceptor_scores.csv", index=False)
        logger.info(f"  ✓ Wrote: {rank_dir / 'acceptor_scores.csv'} ({len(acceptor_shard_df)} rows)")

    elapsed_scoring = time.time() - t_start
    logger.info(f"  Rank {rank} scoring complete in {elapsed_scoring:.1f}s")

    # ── Barrier: wait for all ranks ───────────────────────────────────────────
    if comm is not None:
        logger.info(f"  Rank {rank} waiting at barrier...")
        comm.Barrier()
        logger.info(f"  Rank {rank} passed barrier.")
    elif args.distributed and world_size > 1:
        # Without mpi4py, we use a file-based sentinel approach
        sentinel_dir = output_dir / "sentinels"
        os.makedirs(sentinel_dir, exist_ok=True)
        # Write my sentinel
        with open(sentinel_dir / f"rank_{rank:03d}_done", 'w') as f:
            f.write("done")
        # Rank 0 waits for all sentinels
        if rank == 0:
            logger.info("  Rank 0 waiting for all ranks to finish scoring...")
            while True:
                done_count = len(list(sentinel_dir.glob("rank_*_done")))
                if done_count >= world_size:
                    break
                time.sleep(2)
            logger.info(f"  All {world_size} ranks complete.")
        else:
            # Non-zero ranks exit after writing their shard
            logger.info(f"  Rank {rank} complete. Exiting (rank 0 handles merge).")
            return 0

    # ══════════════════════════════════════════════════════════════════════════════
    # RANK 0 ONLY: MERGE + PLOTS + REPORTS
    # ══════════════════════════════════════════════════════════════════════════════
    if rank != 0:
        return 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("RANK 0: MERGING ALL SHARD OUTPUTS")
    logger.info("=" * 70)

    # Collect all rank outputs
    donor_parts = []
    acceptor_parts = []

    for r in range(world_size):
        r_dir = output_dir / f"rank_{r:03d}"
        donor_file = r_dir / "donor_scores.csv"
        acceptor_file = r_dir / "acceptor_scores.csv"
        if donor_file.exists():
            donor_parts.append(pd.read_csv(donor_file))
        if acceptor_file.exists():
            acceptor_parts.append(pd.read_csv(acceptor_file))

    donor_merged = pd.concat(donor_parts, ignore_index=True) if donor_parts else None
    acceptor_merged = pd.concat(acceptor_parts, ignore_index=True) if acceptor_parts else None

    if donor_merged is not None:
        # Add global rank
        donor_merged["synth_rank"] = donor_merged["combined_synthesizability"].rank(
            ascending=False, method="min", na_option="bottom").astype(int)
        donor_merged.to_csv(results_dir / "donor_synthesizability_scored.csv", index=False)
        logger.info(f"  ✓ Merged donor scores: {len(donor_merged)} molecules → "
                    f"{results_dir / 'donor_synthesizability_scored.csv'}")

    if acceptor_merged is not None:
        acceptor_merged["synth_rank"] = acceptor_merged["combined_synthesizability"].rank(
            ascending=False, method="min", na_option="bottom").astype(int)
        acceptor_merged.to_csv(results_dir / "acceptor_synthesizability_scored.csv", index=False)
        logger.info(f"  ✓ Merged acceptor scores: {len(acceptor_merged)} molecules → "
                    f"{results_dir / 'acceptor_synthesizability_scored.csv'}")

    # ── Generate plots ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING MANUSCRIPT-QUALITY PLOTS")
    logger.info("=" * 70)

    setup_plot_style()

    if donor_merged is not None:
        generate_plots(donor_merged, "donor", plot_dir, plot_data_dir,
                       args.threshold_sa, args.threshold_ra, logger)
    if acceptor_merged is not None:
        generate_plots(acceptor_merged, "acceptor", plot_dir, plot_data_dir,
                       args.threshold_sa, args.threshold_ra, logger)
    if donor_merged is not None and acceptor_merged is not None:
        generate_comparison_plots(donor_merged, acceptor_merged, plot_dir, plot_data_dir,
                                  args.threshold_sa, args.threshold_ra, logger)

    # ── Write reports ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("WRITING ANALYSIS REPORTS")
    logger.info("=" * 70)

    if donor_merged is not None:
        write_analysis_report(donor_merged, "donor", output_dir,
                              args.threshold_sa, args.threshold_ra, logger)
    if acceptor_merged is not None:
        write_analysis_report(acceptor_merged, "acceptor", output_dir,
                              args.threshold_sa, args.threshold_ra, logger)

    # ── Write metrics ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("WRITING OUTPUT METRICS")
    logger.info("=" * 70)

    write_output_metrics(donor_merged, acceptor_merged, output_dir,
                         args.threshold_sa, args.threshold_ra, logger)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    logger.info(f"  Ranks used: {world_size}")
    logger.info(f"  SAScore backend: {_SASCORE_BACKEND}")
    logger.info(f"  RAScore backend: {_RASCORE_BACKEND}")
    n_d = len(donor_merged) if donor_merged is not None else 0
    n_a = len(acceptor_merged) if acceptor_merged is not None else 0
    logger.info(f"  Donors scored: {n_d}")
    logger.info(f"  Acceptors scored: {n_a}")
    logger.info(f"  Throughput: {(n_d + n_a) / elapsed_total:.1f} mol/s total "
                f"({(n_d + n_a) / elapsed_total / world_size:.1f} mol/s/rank)")
    logger.info("")
    logger.info(f"  Output:")
    logger.info(f"    {output_dir}/synthesizability_scoring_progress.log")
    logger.info(f"    {output_dir}/output_metrics_synthesizability.txt")
    logger.info(f"    {output_dir}/donor_synthesizability_analysis_report.txt")
    logger.info(f"    {output_dir}/acceptor_synthesizability_analysis_report.txt")
    logger.info(f"    {results_dir}/donor_synthesizability_scored.csv")
    logger.info(f"    {results_dir}/acceptor_synthesizability_scored.csv")
    logger.info(f"    {plot_dir}/*.png (14 plots)")
    logger.info(f"    {plot_data_dir}/*.csv (plot data for manual re-plotting)")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
