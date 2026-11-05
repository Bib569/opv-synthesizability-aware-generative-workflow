#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected E_loss vs PCE plot
=============================

WHY THIS SCRIPT EXISTS
-----------------------
The original `create_eloss_vs_pce_plot()` in manuscript_plots.py draws the
"generated" scatter from:

    merged/plot_data/pair_predictions_data.csv  ->  column 'PCE'

That pooled 'PCE' column is the FINAL blended device PCE written by
GAN_240_final.py. Empirically (verified on 28-Jul/9-Aug run data)
`generated_molecules_with_predictions.csv` and the postgen `pair_predictions.csv`
both show PCE == legacy Method-2 (physics) PCE in ~99.8% of rows, i.e. the
Method-1-only fix that now lives in GAN_240_final.py's *source code* was not
yet in effect when this specific batch of results was generated. The result:
eloss_vs_pce.png's "generated" cloud (mean PCE ~19-21%, hard-capped at 21%)
is a Method-2 artefact, not a Method-1 (ScreeningSurrogatePipeline) prediction.

THE FIX (v2 -- corrects a v1 bug)
----------------------------------
v1 of this script used the `dev_PCE` column already present in
`pair_predictions.csv` as a same-file, physics-blend-free stand-in for
Method 1. That was WRONG: `dev_PCE` is a *different*, independently trained
model (fit inside postgen_pipeline.py, DMPNN-grouped CV, held-out R^2=0.660,
max observed value only ~11.4%) -- not the same "Method 1" screening
surrogate (`surrogate_PCE`, fit inside GAN_240_final.py's
ScreeningSurrogatePipeline) that every other Method-1 number in the
manuscript (Figure 5, Figure 11, Table 1: mean 5.87%, max 18.22%) is built
from. Plotting `dev_PCE` silently swapped in a more conservative model and
made the high-PCE tail (7.5-18.22%) disappear from this figure alone.

The correct fix joins the TRUE Method-1 value (`surrogate_PCE`, from
`generated_molecules_with_predictions.csv`) to the already-correct,
physics-gated E_loss value (`eloss_pred_physical`, from
`pair_predictions.csv`) by molecule identity (donor_smiles, acceptor_smiles).
The two files contain the same ~12M-pair pool but in different row order
(rank-sharded I/O reorders rows between the generation and post-generation
stages), so a positional column swap is not valid -- a key-based join is
required.

Given machine memory constraints, the join is done as a streamed
sample-and-lookup:
  1. Draw a reproducible systematic sample of N_SAMPLE pairs from
     `generated_molecules_with_predictions.csv`, keeping only
     (donor_smiles, acceptor_smiles, surrogate_PCE).
  2. Stream `pair_predictions.csv` in chunks, keeping only rows whose
     (donor_smiles, acceptor_smiles) key is in the sampled set, and record
     their `eloss_pred_physical`.
  3. Join the two in memory (both are now small) and plot.

This script re-reads only the columns needed, draws the corrected figure,
and writes the joined plot data back out to CSV so it can be re-plotted by
hand later.

USAGE
-----
    python src/analysis/create_eloss_vs_pce_corrected.py
Adjust GEN_POOL_CSV / PAIR_PRED_CSV / DMPNN_CSV / OUT_DIR below if paths differ.

INPUTS NOT SHIPPED IN THIS REPOSITORY (HPC outputs, too large for git)
----------------------------------------------------------------------
  data/large_inputs/generated_molecules_with_predictions.csv   (about 7.4 GB;
      all 11,988,459 pairs with surrogate_* columns; written by stage 1,
      hpc/sub_140.sh -> GAN_240_final.py + merge_rank_outputs_final.py --stage generation)
  data/large_inputs/pair_predictions.csv   (about 5.2 GB; written by stage 2,
      hpc/sub_postgen_full_140.sh -> postgen_pipeline.py + merge --stage postgen)
Obtain them by running those stages (see docs/PIPELINE.md) or request them
from the authors. data/input/DMPNN.csv (training overlay) ships in the repo.
The 20,000-point plot data that this script produces for manuscript Fig. 9 is
already committed in data/figure_inputs/eloss_vs_pce_*_data.csv, so Fig. 9 can be
redrawn without these large files via scripts/figures/06_eloss_vs_pce.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG - adjust if your paths differ
# ---------------------------------------------------------------------------
_base = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_base, '..', '..'))   # src/analysis/ -> repo root
_large = os.path.join(_repo, 'data', 'large_inputs')         # not shipped (see USAGE above)
DMPNN_CSV = os.path.join(_repo, 'data', 'input', 'DMPNN.csv')
GEN_POOL_CSV = os.path.join(_large, 'generated_molecules_with_predictions.csv')
PAIR_PRED_CSV = os.path.join(_large, 'pair_predictions.csv')
OUT_DIR = os.path.join(_repo, 'results', 'eloss_vs_pce')
PLOT_DATA_DIR = os.path.join(_repo, 'results', 'eloss_vs_pce', 'plot_data')
DPI = 300
N_SAMPLE = 20000
SEED = 42
GEN_CHUNKSIZE = 500_000      # rows per chunk when sampling the 7 GB pool file
PAIR_CHUNKSIZE = 1_000_000   # rows per chunk when scanning the 5 GB pair-prediction file

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DATA_DIR, exist_ok=True)

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False

COLOR_TRAIN = '#1B7F79'
COLOR_GEN = '#D4602E'


def _style_ax(ax, xlabel='', ylabel='', fontsize=13):
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


def main():
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel=r'$E_{\mathrm{loss}}$ (eV)', ylabel='PCE (%)')
    plotted = False
    train_out = None
    gen_out = None

    # ---------------- training side (unchanged from original script) -------
    if os.path.exists(DMPNN_CSV):
        tdf = pd.read_csv(DMPNN_CSV)
        eloss_col = 'E_loss' if 'E_loss' in tdf.columns else None
        pce_col = 'PCE_max(%)' if 'PCE_max(%)' in tdf.columns else (
            'PCE_max' if 'PCE_max' in tdf.columns else None)
        if eloss_col and pce_col:
            e = pd.to_numeric(tdf[eloss_col], errors='coerce')
            p = pd.to_numeric(tdf[pce_col], errors='coerce')
            ok = e.notna() & p.notna()
            unphys = ok & (e < 0.25)
            phys = ok & ~unphys
            ax.scatter(e[phys], p[phys], s=30, color=COLOR_TRAIN, alpha=0.7,
                       label='Training data, physical', edgecolor='black', linewidth=0.3)
            ax.scatter(e[unphys], p[unphys], s=30, color='red',
                       marker='x', linewidth=1.2,
                       label=r'Training data, excluded (unphysical, $E_{loss}$<0.25 eV)')
            plotted = True
            train_out = pd.DataFrame({'E_loss': e[ok], 'PCE_max_pct': p[ok], 'excluded_unphysical': unphys[ok]})
        else:
            print("[warn] DMPNN.csv missing E_loss/PCE_max_pct columns for training overlay")
    else:
        print("[warn] DMPNN.csv not found at %s" % DMPNN_CSV)

    # ---------------- generated side (CORRECTED v2) -------------------------
    # Method-1 PCE (surrogate_PCE) lives in GEN_POOL_CSV; the physics-gated
    # E_loss (eloss_pred_physical) lives in PAIR_PRED_CSV. The two files are
    # NOT row-aligned (rank-sharded I/O reorders rows between pipeline
    # stages), so we join by (donor_smiles, acceptor_smiles) instead of by
    # position. To keep this within available memory, we sample first and
    # look up second, rather than loading either 5-7 GB file whole.
    if os.path.exists(GEN_POOL_CSV) and os.path.exists(PAIR_PRED_CSV):
        print("Step 1/2: drawing a systematic sample of %d pairs from %s ..."
              % (N_SAMPLE, os.path.basename(GEN_POOL_CSV)))
        sample_chunks = []
        n_seen = 0
        step = None
        # First pass: count total rows cheaply via the header-less line count
        # is expensive for 7GB, so instead take a fixed stride estimated from
        # the known pool size range for this pipeline (~12M pairs), then
        # correct at the end if we under/over-shoot.
        rng = np.random.RandomState(SEED)
        for chunk in pd.read_csv(GEN_POOL_CSV,
                                  usecols=['donor_smiles', 'acceptor_smiles', 'surrogate_PCE'],
                                  chunksize=GEN_CHUNKSIZE):
            chunk = chunk.dropna(subset=['surrogate_PCE'])
            if len(chunk) == 0:
                n_seen += GEN_CHUNKSIZE
                continue
            # reservoir-style per-chunk random subsample proportional to chunk size
            frac = min(1.0, (N_SAMPLE * 3) / 11_988_459)  # oversample ~3x for join loss
            take = max(1, int(len(chunk) * frac))
            take = min(take, len(chunk))
            idx = rng.choice(len(chunk), size=take, replace=False)
            sample_chunks.append(chunk.iloc[idx])
            n_seen += GEN_CHUNKSIZE

        gen_sample = pd.concat(sample_chunks, ignore_index=True)
        print("  drew %d candidate pairs with surrogate_PCE" % len(gen_sample))

        # build a lookup set/dict keyed on (donor_smiles, acceptor_smiles)
        gen_sample['_key'] = gen_sample['donor_smiles'].astype(str) + '||' + gen_sample['acceptor_smiles'].astype(str)
        pce_lookup = dict(zip(gen_sample['_key'], gen_sample['surrogate_PCE']))
        wanted_keys = set(pce_lookup.keys())

        print("Step 2/2: scanning %s for matching pairs (this reads the full file once)..."
              % os.path.basename(PAIR_PRED_CSV))
        matched_e = []
        matched_p = []
        n_scanned = 0
        for chunk in pd.read_csv(PAIR_PRED_CSV,
                                  usecols=['donor_smiles', 'acceptor_smiles', 'eloss_pred_physical'],
                                  chunksize=PAIR_CHUNKSIZE):
            n_scanned += len(chunk)
            chunk_key = chunk['donor_smiles'].astype(str) + '||' + chunk['acceptor_smiles'].astype(str)
            hit_mask = chunk_key.isin(wanted_keys)
            if not hit_mask.any():
                continue
            hits = chunk.loc[hit_mask]
            hit_keys = chunk_key.loc[hit_mask]
            for k, eloss in zip(hit_keys, hits['eloss_pred_physical']):
                pce = pce_lookup.get(k)
                if pce is not None and pd.notna(eloss):
                    matched_e.append(eloss)
                    matched_p.append(pce)
            if len(matched_e) >= N_SAMPLE:
                break
        print("  scanned %d rows, matched %d pairs with both surrogate_PCE and eloss_pred_physical"
              % (n_scanned, len(matched_e)))

        if len(matched_e) > N_SAMPLE:
            sel = np.random.RandomState(SEED).choice(len(matched_e), N_SAMPLE, replace=False)
            e_s = np.array(matched_e)[sel]
            p_s = np.array(matched_p)[sel]
        else:
            e_s = np.array(matched_e)
            p_s = np.array(matched_p)

        ax.scatter(e_s, p_s, s=8, color=COLOR_GEN, alpha=0.35,
                   label=r'Generated (predicted PCE)')
        plotted = True
        gen_out = pd.DataFrame({'E_loss_pred_physical': e_s, 'surrogate_PCE': p_s})
    else:
        missing = [p for p in (GEN_POOL_CSV, PAIR_PRED_CSV) if not os.path.exists(p)]
        print("[warn] required file(s) not found: %s" % missing)

    ax.axvline(0.25, color='grey', linestyle=':', linewidth=1.2,
               label='Detailed-balance floor (0.25 eV)')

    if plotted:
        # The corrected (surrogate_PCE) data now densely populates the FULL
        # PCE range (0-18.22%), so no in-axes corner is free of points.
        # Place the legend below the plot instead of on top of the data.
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13),
                   ncol=2, frameon=True)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, 'eloss_vs_pce_corrected.png')
        plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        print("Saved: %s" % out_path)

        if train_out is not None:
            train_out.to_csv(os.path.join(PLOT_DATA_DIR, 'eloss_vs_pce_training_data.csv'), index=False)
            print("Saved: %s" % os.path.join(PLOT_DATA_DIR, 'eloss_vs_pce_training_data.csv'))
        if gen_out is not None:
            gen_out.to_csv(os.path.join(PLOT_DATA_DIR, 'eloss_vs_pce_generated_corrected_data.csv'), index=False)
            print("Saved: %s" % os.path.join(PLOT_DATA_DIR, 'eloss_vs_pce_generated_corrected_data.csv'))
    else:
        plt.close(fig)
        print("Nothing plotted - check input paths.")


if __name__ == '__main__':
    main()
