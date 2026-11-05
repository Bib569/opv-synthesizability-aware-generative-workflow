#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
08_tier2_scaffold_family_bars.py  ->  tier2_scaffold_family_bars.png
=============================================================================
Most frequent Bemis--Murcko scaffolds among Tier-2 donor/acceptor molecules.
REQUIRES RDKit.

Required input CSV (in --data-dir):
    high_performance_pairs.csv    cols: donor_smiles, acceptor_smiles

Usage:
    python 08_tier2_scaffold_family_bars.py --data-dir data/figure_inputs
=============================================================================
"""
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, need, parse_common_args, resolve_dirs)


def fig_scaffold_family_bars(data_dir, outdir):
    name = 'tier2_scaffold_family_bars.png'
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog('rdApp.*')
    except Exception as e:
        print(f"  [SKIP] {name}: RDKit not available ({e})")
        return False

    hp_csv = need(data_dir, 'high_performance_pairs.csv')
    if hp_csv is None:
        print(f"  [SKIP] {name}: missing high_performance_pairs.csv")
        return False
    hp = pd.read_csv(hp_csv)

    def murcko(smi):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s, canonical=True) if s is not None else None

    donors = hp['donor_smiles'].dropna().unique().tolist()
    accs = hp['acceptor_smiles'].dropna().unique().tolist()
    print("  Parsing Tier-2 molecules with RDKit for scaffolds (slow step)...")
    d_scaf = Counter([x for x in (murcko(s) for s in donors) if x])
    a_scaf = Counter([x for x in (murcko(s) for s in accs) if x])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    for ax, top, color, n_total in [
        (axes[0], d_scaf.most_common(8), COLORS['donor'], len(donors)),
        (axes[1], a_scaf.most_common(8), COLORS['acceptor'], len(accs)),
    ]:
        style_ax(ax, xlabel='', ylabel='Number of Tier-2 molecules')
        labels = [f"scaffold {i+1}" for i in range(len(top))]
        counts = [c for _, c in top]
        ax.bar(labels, counts, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for i, c in enumerate(counts):
            ax.annotate(f'{c}\n({100*c/n_total:.1f}%)', xy=(i, c), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 8: Tier-2 scaffold-family bars (needs RDKit).")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_scaffold_family_bars(data_dir, outdir)


if __name__ == "__main__":
    main()
