#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
10_frechet_baseline.py  ->  frechet_baseline_plot.png
=============================================================================
Train--train split-half null distribution of the fingerprint-space Frechet
distance (FP-PCA-FD). REQUIRES RDKit + scikit-learn + SciPy.

Required input CSVs (in --data-dir):
    DMPNN.csv        cols: donor(SMILES), acceptor(SMILES)
    Guzik_data.csv   cols: Donor SMILES, Acceptor SMILES

Optional:
    --frechet-splits N   (default 200)

Usage:
    python 10_frechet_baseline.py --data-dir data/figure_inputs
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, need, parse_common_args, resolve_dirs)

REPORTED_FP_PCA_FD = {'donor': 25.20, 'acceptor': 17.00}


def fig_frechet(data_dir, outdir, n_splits=200, seed=42):
    name = 'frechet_baseline_plot.png'
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        from sklearn.decomposition import PCA
        RDLogger.DisableLog('rdApp.*')
    except Exception as e:
        print(f"  [SKIP] {name}: RDKit/scikit-learn not available ({e})")
        return False

    dmpnn_csv = need(data_dir, 'DMPNN.csv')
    guzik_csv = need(data_dir, 'Guzik_data.csv')
    if dmpnn_csv is None or guzik_csv is None:
        miss = ['DMPNN.csv'] if dmpnn_csv is None else []
        miss += ['Guzik_data.csv'] if guzik_csv is None else []
        print(f"  [SKIP] {name}: missing {', '.join(miss)}")
        return False
    dmpnn = pd.read_csv(dmpnn_csv)
    guzik = pd.read_csv(guzik_csv)

    def morgan(mol, radius=2, n_bits=2048):
        try:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        except Exception:
            return None
        arr = np.zeros((n_bits,), dtype=np.float64)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(bv, arr)
        return arr

    def frechet_feat(fa, fb):
        fa = np.asarray(fa, dtype=np.float64)
        fb = np.asarray(fb, dtype=np.float64)
        mu1, mu2 = fa.mean(0), fb.mean(0)
        s1 = np.cov(fa, rowvar=False)
        s2 = np.cov(fb, rowvar=False)
        diff = mu1 - mu2
        from scipy.linalg import sqrtm
        cov = sqrtm(s1.dot(s2))
        if np.iscomplexobj(cov):
            cov = cov.real
        return float(diff.dot(diff) + np.trace(s1 + s2 - 2.0 * cov))

    def fp_pca_frechet(mols_a, mols_b, n_components=2):
        fa = [f for f in (morgan(m) for m in mols_a) if f is not None]
        fb = [f for f in (morgan(m) for m in mols_b) if f is not None]
        if len(fa) < 3 or len(fb) < 3:
            return np.nan
        X = np.vstack([np.array(fa), np.array(fb)])
        k = min(n_components, X.shape[1], X.shape[0] - 1)
        Z = PCA(n_components=k, random_state=seed).fit_transform(X)
        return frechet_feat(Z[:len(fa)], Z[len(fa):])

    def load_side(side):
        if side == 'donor':
            raw = list(dmpnn['donor(SMILES)'].dropna().astype(str)) + \
                  list(guzik['Donor SMILES'].dropna().astype(str))
        else:
            raw = list(dmpnn['acceptor(SMILES)'].dropna().astype(str)) + \
                  list(guzik['Acceptor SMILES'].dropna().astype(str))
        mols, seen = [], set()
        for s in raw:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            cs = Chem.MolToSmiles(m, canonical=True)
            if cs in seen:
                continue
            seen.add(cs)
            mols.append(m)
        return mols

    def split_half(mols):
        rng = np.random.RandomState(seed)
        n = len(mols)
        vals = []
        for _ in range(n_splits):
            idx = rng.permutation(n)
            half = n // 2
            a = [mols[j] for j in idx[:half]]
            b = [mols[j] for j in idx[half:2 * half]]
            vals.append(fp_pca_frechet(a, b))
        v = np.array(vals, dtype=np.float64)
        return v[np.isfinite(v)]

    print("  Computing frechet split-half baseline (RDKit, 200 splits/side)...")
    all_vals = {s: split_half(load_side(s)) for s in ('donor', 'acceptor')}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')
    for ax, vals, side, color in [
        (axes[0], all_vals['donor'], 'donor', '#1B7F79'),
        (axes[1], all_vals['acceptor'], 'acceptor', '#D4602E'),
    ]:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(width=2.0, length=6, labelsize=11, direction='in', top=True, right=True)
        ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor='black', linewidth=0.5,
                label=f"train-vs-train split-half\n(n={len(vals)} random splits)")
        ax.axvline(np.nanmean(vals), color='black', linestyle='-', linewidth=1.8,
                   label=f"split-half mean = {np.nanmean(vals):.2f}")
        ax.axvline(REPORTED_FP_PCA_FD[side], color='#C62828', linestyle='--', linewidth=2.2,
                   label=f"train-vs-generated = {REPORTED_FP_PCA_FD[side]:.1f}\n(manuscript-reported)")
        ax.set_xlabel('FP-PCA-FD', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count (of 200 random splits)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper right')
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 10: FP-PCA-FD split-half baseline (needs RDKit+sklearn+SciPy).")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_frechet(data_dir, outdir, n_splits=args.frechet_splits)


if __name__ == "__main__":
    main()
