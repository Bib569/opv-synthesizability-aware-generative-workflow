#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frechet_baseline_analysis.py
=============================

Answers reviewer point 1.4 ("Frechet Distances Lack Reference Baselines") by
computing a train-vs-train (split-half) baseline for the fingerprint-space
Frechet distance (FP-PCA-FD) using the IDENTICAL algorithm already used for
the train-vs-generated value reported in the manuscript
(postgen_pipeline.py::metric_fp_pca_frechet / _frechet_from_features).

Method
------
For each of the donor and acceptor reference libraries (157 and 483 unique
canonical training molecules respectively, pooled from DMPNN.csv + Guzik_data.csv
exactly as in postgen_pipeline.py's Stage C4), the reference set is repeatedly
split into two random, non-overlapping halves and the FP-PCA-FD is computed
BETWEEN THE TWO HALVES using the same radius-2/2048-bit Morgan fingerprint +
2-component PCA + Frechet-distance-of-Gaussians procedure used for the
train-vs-generated number. This gives an honest "distance expected from
finite-sample noise alone, with zero systematic distributional shift" floor
against which the train-vs-generated distance (25.2 donor / 17.0 acceptor) can
be judged.

Because the reference libraries are small (157 / 483 molecules), 200
independent random half/half splits are drawn and the resulting distribution
(mean, std, percentiles) is reported rather than a single point estimate, to
avoid over-interpreting a single split's sampling noise.

The saved property-embedding model used for the manuscript's PFD (444.3 /
428.0) was not persisted to disk in this run's artifact tree, so an
apples-to-apples PFD baseline cannot be recomputed without retraining that
model; this script only rebuilds the FP-PCA-FD baseline, for which the
computation requires nothing beyond RDKit fingerprints and is therefore fully
reproducible from raw SMILES alone. This limitation is stated explicitly in
the written summary and in the manuscript text drafted around this analysis.

Run from a fresh clone (inputs ship in data/input/):
    python src/analysis/frechet_baseline_analysis.py

Outputs (written to results/tier2_analysis/ of this repository):
  - frechet_baseline_summary.txt
  - frechet_baseline_donor.csv / frechet_baseline_acceptor.csv  (per-split values)
  - frechet_baseline_plot.png  (train-vs-train null distribution with the
    reported train-vs-generated value marked)
"""

from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

_base = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_base, "..", ".."))   # src/analysis/ -> repo root
DMPNN_CSV = os.path.join(_repo, "data", "input", "DMPNN.csv")
GUZIK_CSV = os.path.join(_repo, "data", "input", "Guzik_data.csv")
OUT_DIR = os.path.join(_repo, "results", "tier2_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 200
SEED = 42
N_COMPONENTS = 2

# Manuscript-reported train-vs-generated FP-PCA-FD values (for reference line)
REPORTED_FP_PCA_FD = {"donor": 25.2, "acceptor": 17.0}


def morgan_fp(mol, radius=2, n_bits=2048):
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float64)
        ConvertToNumpyArray(bv, arr)
        return arr
    except Exception:
        return None


def _frechet_from_features(feat_real, feat_gen):
    """Identical to postgen_pipeline.py::_frechet_from_features."""
    feat_real = np.asarray(feat_real, dtype=np.float64)
    feat_gen = np.asarray(feat_gen, dtype=np.float64)
    if feat_real.ndim != 2 or feat_gen.ndim != 2:
        return float("nan")
    if len(feat_real) < 2 or len(feat_gen) < 2:
        return float("nan")
    mu_r, mu_g = feat_real.mean(axis=0), feat_gen.mean(axis=0)
    sig_r = np.atleast_2d(np.cov(feat_real, rowvar=False))
    sig_g = np.atleast_2d(np.cov(feat_gen, rowvar=False))
    diff = mu_r - mu_g
    try:
        covmean = sqrtm(sig_r.dot(sig_g))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fd = diff.dot(diff) + np.trace(sig_r + sig_g - 2.0 * covmean)
        return float(fd)
    except Exception:
        return float("nan")


def fp_pca_frechet(mols_a, mols_b, n_components=N_COMPONENTS, seed=SEED):
    """Identical algorithm to postgen_pipeline.py::metric_fp_pca_frechet,
    but applied between two arbitrary molecule sets (here: two random halves
    of the SAME reference library) rather than train-vs-generated."""
    fa = [f for f in (morgan_fp(m) for m in mols_a) if f is not None]
    fb = [f for f in (morgan_fp(m) for m in mols_b) if f is not None]
    if not fa or not fb:
        return float("nan")
    A, B = np.array(fa), np.array(fb)
    allfp = np.vstack([A, B])
    k = max(2, min(n_components, allfp.shape[0] - 1, allfp.shape[1]))
    pca = PCA(n_components=k, random_state=seed)
    proj = pca.fit_transform(allfp)
    return _frechet_from_features(proj[:len(A)], proj[len(A):])


def load_reference_library(side):
    dmpnn = pd.read_csv(DMPNN_CSV)
    guzik = pd.read_csv(GUZIK_CSV)
    if side == "donor":
        raw = list(dmpnn["donor(SMILES)"].dropna().astype(str)) + \
              list(guzik["Donor SMILES"].dropna().astype(str))
    else:
        raw = list(dmpnn["acceptor(SMILES)"].dropna().astype(str)) + \
              list(guzik["Acceptor SMILES"].dropna().astype(str))
    mols, canon_seen = [], set()
    for s in raw:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        cs = Chem.MolToSmiles(m, canonical=True)
        if cs in canon_seen:
            continue
        canon_seen.add(cs)
        mols.append(m)
    return mols


def split_half_baseline(mols, side, n_splits=N_SPLITS, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(mols)
    vals = []
    for i in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        a_idx, b_idx = idx[:half], idx[half:2 * half]
        mols_a = [mols[j] for j in a_idx]
        mols_b = [mols[j] for j in b_idx]
        v = fp_pca_frechet(mols_a, mols_b, seed=seed + i)
        vals.append(v)
    return np.array(vals, dtype=np.float64)


def make_plot(donor_vals, acc_vals):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    for ax, vals, side, color in [
        (axes[0], donor_vals, "donor", "#1B7F79"),
        (axes[1], acc_vals, "acceptor", "#D4602E"),
    ]:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(width=2.0, length=6, labelsize=11, direction="in", top=True, right=True)
        ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor="black", linewidth=0.5,
                label=f"train-vs-train split-half\n(n={len(vals)} random splits)")
        ax.axvline(np.nanmean(vals), color="black", linestyle="-", linewidth=1.8,
                   label=f"split-half mean = {np.nanmean(vals):.2f}")
        ax.axvline(REPORTED_FP_PCA_FD[side], color="#C62828", linestyle="--", linewidth=2.2,
                   label=f"train-vs-generated = {REPORTED_FP_PCA_FD[side]:.1f}\n(manuscript-reported)")
        ax.set_xlabel("FP-PCA-FD", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count (of 200 random splits)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right")
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "frechet_baseline_plot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def main():
    summary_lines = []
    summary_lines.append("FP-PCA-FD TRAIN-VS-TRAIN (SPLIT-HALF) BASELINE\n" + "=" * 70 + "\n")
    all_vals = {}
    for side in ("donor", "acceptor"):
        mols = load_reference_library(side)
        n_ref = len(mols)
        print(f"{side}: {n_ref} unique canonical reference molecules")
        vals = split_half_baseline(mols, side)
        vals = vals[np.isfinite(vals)]
        all_vals[side] = vals
        reported = REPORTED_FP_PCA_FD[side]
        pct_rank = 100.0 * np.mean(vals < reported)
        summary_lines.append(f"{side.upper()} (n_ref = {n_ref} unique canonical molecules)\n")
        summary_lines.append(f"  Split-half FP-PCA-FD over {len(vals)} random 50/50 splits:\n")
        summary_lines.append(f"    mean   = {np.mean(vals):.3f}\n")
        summary_lines.append(f"    std    = {np.std(vals):.3f}\n")
        summary_lines.append(f"    median = {np.median(vals):.3f}\n")
        summary_lines.append(f"    5th-95th pct = [{np.percentile(vals,5):.3f}, {np.percentile(vals,95):.3f}]\n")
        summary_lines.append(f"    min/max = [{np.min(vals):.3f}, {np.max(vals):.3f}]\n")
        summary_lines.append(f"  Manuscript-reported train-vs-generated FP-PCA-FD = {reported:.2f}\n")
        summary_lines.append(f"  -> {reported:.2f} exceeds {pct_rank:.1f}% of the {len(vals)} "
                             f"train-vs-train split-half values "
                             f"(i.e. is at or above the {pct_rank:.1f}th percentile of the pure-noise floor).\n")
        summary_lines.append(f"  -> ratio train-vs-generated / split-half mean = {reported/np.mean(vals):.2f}x\n\n")

        pd.DataFrame({"split_half_fp_pca_fd": vals}).to_csv(
            os.path.join(OUT_DIR, f"frechet_baseline_{side}.csv"), index=False)

    make_plot(all_vals["donor"], all_vals["acceptor"])

    summary_lines.append(
        "NOTE ON PFD (property-embedding Frechet distance, manuscript values 444.3 / 428.0):\n"
        "The trained property-embedding model used for this metric was not persisted to disk\n"
        "in the available run artefacts, so a matched train-vs-train PFD baseline could not be\n"
        "recomputed without retraining that model from scratch. Only the FP-PCA-FD baseline\n"
        "above is reproduced here; the manuscript text explicitly flags PFD as lacking an\n"
        "equivalent baseline and recommends one be computed in a future revision once the\n"
        "embedding-model checkpoint is retained.\n"
    )

    with open(os.path.join(OUT_DIR, "frechet_baseline_summary.txt"), "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print("".join(summary_lines))
    print("Wrote frechet_baseline_summary.txt")


if __name__ == "__main__":
    main()
