#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tier2_chemical_analysis.py
============================

One-off analysis script (not part of the regular pipeline) that answers the
reviewer's three "results lack chemical interpretation" points:

  1. Illustrate several representative donor and acceptor candidates.
  2. Characterise the scaffold families dominating the 2,832-pair Tier-2
     shortlist and assess whether they are chemically sensible OPV materials
     (thiophene/selenophene/furan donor cores; cyanoacrylate/indanedione-type
     acceptor end groups; A-D-A-type acceptor topology).
  3. Highlight specific candidates from the energy-loss vs. PCE analysis that
     combine low predicted E_loss with high predicted PCE ("sweet spot").

Inputs (all already produced by the existing pipeline; no new predictions
are made -- this script only re-reads and chemically annotates them):
  - data/figure_inputs/high_performance_pairs.csv        (Tier-2 shortlist, 2832 rows; ships in repo)
  - data/large_inputs/pair_predictions.csv               (pool-wide predictions,
                                                            joined by SMILES key
                                                            to attach eloss_pred_physical)
                                                          *** NOT SHIPPED (about 5.2 GB, HPC output). ***
                                                          Obtain it by running stage 2 of the pipeline
                                                          (hpc/sub_postgen_full_140.sh, then
                                                          merge_rank_outputs_final.py --stage postgen;
                                                          see docs/PIPELINE.md) or request it from the
                                                          authors, and place it at the path above.
                                                          Without it this script cannot run from a clone;
                                                          its outputs are already committed in
                                                          results/tier2_analysis/.
  - data/generated_libraries/{donor,acceptor}_molecule_predictions.csv
                                                          (full 52,995 / 53,719 -molecule
                                                            libraries, used as the
                                                            background population for
                                                            scaffold-enrichment comparison)

Outputs (written to results/tier2_analysis/ of this repository):
  - tier2_donor_scaffold_counts.csv / tier2_acceptor_scaffold_counts.csv
  - tier2_substructure_enrichment.csv
  - tier2_sweet_spot_candidates.csv     (low-Eloss / high-PCE shortlist)
  - tier2_representative_grid_donors.png / _acceptors.png  (2D structure grids)
  - tier2_eloss_pce_sweet_spot.png       (scatter highlighting the sweet-spot set)
  - tier2_scaffold_family_bars.png       (donor/acceptor scaffold-family bar chart)
  - tier2_chemical_summary.txt           (plain-text summary of every number below)

Usage:
    python src/analysis/tier2_chemical_analysis.py
"""

from __future__ import annotations

import os
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Nimbus Roman No9 L',
                               'Liberation Serif', 'STIXGeneral']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = False

DPI = 300

COLORS = {
    'donor': '#1B7F79', 'acceptor': '#D4602E',
    'threshold': '#C62828', 'mean_line': '#212121',
    'sweet': '#8E24AA',
}


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


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_base = Path(__file__).resolve().parent
_repo = _base.parents[1]                       # src/analysis/ -> repo root
HP_CSV = _repo / "data" / "figure_inputs" / "high_performance_pairs.csv"
PAIR_PRED_CSV = _repo / "data" / "large_inputs" / "pair_predictions.csv"   # not shipped (HPC output)
DONOR_LIB_CSV = _repo / "data" / "generated_libraries" / "donor_molecule_predictions.csv"
ACCEPTOR_LIB_CSV = _repo / "data" / "generated_libraries" / "acceptor_molecule_predictions.csv"
OUT_DIR = _repo / "results" / "tier2_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SWEET_SPOT = 12
ELOSS_SWEET_MAX = 0.55       # eV; "low" energy loss within the physically screened Tier-2 set
PCE_SWEET_MIN = 17.0         # %; near the top of the Tier-2 PCE range (14.0-18.22%)

SUBSTRUCTURE_PATTERNS = {
    'thiophene':            'c1ccsc1',
    'selenophene':          'c1cc[se]c1',
    'furan':                'c1ccoc1',
    'pyridine':             'c1ccncc1',
    'benzothiophene':       'c1ccc2sccc2c1',
    'nitrile (-C#N)':       'C#N',
    'cyclic imide (5-ring)': 'O=C1NC(=O)CC1',
    'phthalimide-type (6-ring fused)': 'O=C1NC(=O)c2ccccc21',
    'benzothiadiazole-type (c1nsnc1)': 'c1nsnc1',
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# STEP 1: load Tier-2 shortlist and attach eloss_pred_physical by SMILES key
# ---------------------------------------------------------------------------
def load_tier2_with_eloss():
    log(f"Loading Tier-2 shortlist: {HP_CSV}")
    hp = pd.read_csv(HP_CSV)
    hp['_key'] = hp['donor_smiles'].astype(str) + '||' + hp['acceptor_smiles'].astype(str)
    wanted = set(hp['_key'])

    log(f"Streaming {PAIR_PRED_CSV.name} to attach eloss_pred_physical "
        f"for {len(wanted):,} Tier-2 pairs (11.99M-row file; single pass)...")
    matches = []
    for chunk in pd.read_csv(PAIR_PRED_CSV,
                              usecols=['donor_smiles', 'acceptor_smiles',
                                       'eloss_pred_physical', 'eloss_pred_negative',
                                       'eloss_pred_unphysical'],
                              chunksize=2_000_000):
        key = chunk['donor_smiles'].astype(str) + '||' + chunk['acceptor_smiles'].astype(str)
        hit = key.isin(wanted)
        if hit.any():
            sub = chunk.loc[hit].copy()
            sub['_key'] = key.loc[hit]
            matches.append(sub)
    res = pd.concat(matches, ignore_index=True).drop_duplicates(subset='_key')
    log(f"  matched {len(res):,} / {len(wanted):,} Tier-2 pairs with an E_loss estimate")

    merged = hp.merge(
        res[['_key', 'eloss_pred_physical', 'eloss_pred_negative', 'eloss_pred_unphysical']],
        on='_key', how='left')
    return merged


# ---------------------------------------------------------------------------
# STEP 2: scaffold and substructure analysis
# ---------------------------------------------------------------------------
def murcko_scaffold_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(s, canonical=True) if s is not None else None


def generic_scaffold_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    if s is None:
        return None
    g = MurckoScaffold.MakeScaffoldGeneric(s)
    return Chem.MolToSmiles(g, canonical=True)


def substructure_fraction(mols, smarts):
    patt = Chem.MolFromSmarts(smarts)
    hits = sum(1 for m in mols if m is not None and m.HasSubstructMatch(patt))
    total = sum(1 for m in mols if m is not None)
    return hits, total


def scaffold_analysis(hp):
    donors = hp['donor_smiles'].unique().tolist()
    accs = hp['acceptor_smiles'].unique().tolist()
    log(f"Tier-2 unique donors: {len(donors)}; unique acceptors: {len(accs)}")

    log("Loading full donor/acceptor libraries for background comparison...")
    d_full = pd.read_csv(DONOR_LIB_CSV, usecols=['smiles'])
    a_full = pd.read_csv(ACCEPTOR_LIB_CSV, usecols=['smiles'])
    donors_full = d_full['smiles'].tolist()
    accs_full = a_full['smiles'].tolist()
    log(f"Full donor library: {len(donors_full)}; full acceptor library: {len(accs_full)}")

    log("Parsing molecules with RDKit (this is the slow step, ~1-2 min)...")
    donor_mols_t2 = [Chem.MolFromSmiles(s) for s in donors]
    acc_mols_t2 = [Chem.MolFromSmiles(s) for s in accs]
    donor_mols_full = [Chem.MolFromSmiles(s) for s in donors_full]
    acc_mols_full = [Chem.MolFromSmiles(s) for s in accs_full]
    log("  parsing complete")

    # ---- Bemis-Murcko scaffold family counts (Tier-2 only) ----------------
    donor_scaf = [murcko_scaffold_smiles(s) for s in donors]
    acc_scaf = [murcko_scaffold_smiles(s) for s in accs]
    donor_scaf_counts = Counter(donor_scaf)
    acc_scaf_counts = Counter(acc_scaf)

    donor_scaf_df = pd.DataFrame(
        [(s, c, 100 * c / len(donors)) for s, c in donor_scaf_counts.most_common()],
        columns=['bemis_murcko_scaffold', 'n_donors', 'pct_of_tier2_donors'])
    acc_scaf_df = pd.DataFrame(
        [(s, c, 100 * c / len(accs)) for s, c in acc_scaf_counts.most_common()],
        columns=['bemis_murcko_scaffold', 'n_acceptors', 'pct_of_tier2_acceptors'])
    donor_scaf_df.to_csv(OUT_DIR / "tier2_donor_scaffold_counts.csv", index=False)
    acc_scaf_df.to_csv(OUT_DIR / "tier2_acceptor_scaffold_counts.csv", index=False)
    log(f"  Tier-2 donors span {len(donor_scaf_counts)} distinct Bemis-Murcko scaffolds "
        f"(of {len(donors)} unique donors)")
    log(f"  Tier-2 acceptors span {len(acc_scaf_counts)} distinct Bemis-Murcko scaffolds "
        f"(of {len(accs)} unique acceptors)")

    # ---- Substructure enrichment: Tier-2 vs full library ------------------
    rows = []
    for name, smarts in SUBSTRUCTURE_PATTERNS.items():
        dh, dt = substructure_fraction(donor_mols_t2, smarts)
        dhf, dtf = substructure_fraction(donor_mols_full, smarts)
        ah, at = substructure_fraction(acc_mols_t2, smarts)
        ahf, atf = substructure_fraction(acc_mols_full, smarts)
        rows.append({
            'motif': name,
            'donor_tier2_pct': 100 * dh / dt if dt else np.nan,
            'donor_full_lib_pct': 100 * dhf / dtf if dtf else np.nan,
            'donor_tier2_n': dh, 'donor_tier2_total': dt,
            'acceptor_tier2_pct': 100 * ah / at if at else np.nan,
            'acceptor_full_lib_pct': 100 * ahf / atf if atf else np.nan,
            'acceptor_tier2_n': ah, 'acceptor_tier2_total': at,
        })
    enrich_df = pd.DataFrame(rows)
    enrich_df.to_csv(OUT_DIR / "tier2_substructure_enrichment.csv", index=False)
    log("  substructure enrichment table written")

    return {
        'donors': donors, 'accs': accs,
        'donor_mols_t2': donor_mols_t2, 'acc_mols_t2': acc_mols_t2,
        'donor_scaf_counts': donor_scaf_counts, 'acc_scaf_counts': acc_scaf_counts,
        'enrich_df': enrich_df,
        'n_donor_scaffolds': len(donor_scaf_counts), 'n_acc_scaffolds': len(acc_scaf_counts),
    }


# ---------------------------------------------------------------------------
# STEP 3: energy-loss / PCE "sweet spot" candidates
# ---------------------------------------------------------------------------
def sweet_spot_analysis(hp_eloss):
    valid = hp_eloss.dropna(subset=['eloss_pred_physical']).copy()
    log(f"Tier-2 pairs with a defined (non-excluded) E_loss estimate: {len(valid)} / {len(hp_eloss)}")

    sweet = valid[(valid['eloss_pred_physical'] <= ELOSS_SWEET_MAX) &
                  (valid['surrogate_PCE'] >= PCE_SWEET_MIN)].copy()
    sweet = sweet.sort_values(['surrogate_PCE', 'eloss_pred_physical'],
                               ascending=[False, True])
    log(f"Sweet-spot candidates (E_loss <= {ELOSS_SWEET_MAX} eV and PCE >= {PCE_SWEET_MIN}%): "
        f"{len(sweet)} of {len(valid)} Tier-2 pairs with defined E_loss "
        f"({100 * len(sweet) / len(valid):.2f}%)")

    top = sweet.head(N_SWEET_SPOT).copy()
    keep_cols = ['donor_smiles', 'acceptor_smiles', 'surrogate_PCE', 'surrogate_Jsc',
                 'surrogate_FF', 'surrogate_Voc', 'eloss_pred_physical', 'pair_tango',
                 'donor_tango', 'acceptor_tango']
    top[keep_cols].to_csv(OUT_DIR / "tier2_sweet_spot_candidates.csv", index=False)
    return valid, sweet, top


# ---------------------------------------------------------------------------
# STEP 4: figures
# ---------------------------------------------------------------------------
def make_sweet_spot_scatter(valid, sweet):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    _style_ax(ax, xlabel=r'$E_{\mathrm{loss}}$ (eV)', ylabel='Predicted PCE (%)')

    ax.scatter(valid['eloss_pred_physical'], valid['surrogate_PCE'],
               s=14, color='#9E9E9E', alpha=0.45, edgecolor='none',
               label=f'Tier-2 shortlist (n={len(valid):,})')
    ax.scatter(sweet['eloss_pred_physical'], sweet['surrogate_PCE'],
               s=32, color=COLORS['sweet'], alpha=0.9, edgecolor='black', linewidth=0.4,
               label=f'Sweet-spot subset (n={len(sweet):,})')
    ax.axvline(ELOSS_SWEET_MAX, color=COLORS['threshold'], linestyle='--', linewidth=1.6,
               label=fr'$E_{{\mathrm{{loss}}}} \leq {ELOSS_SWEET_MAX}$ eV')
    ax.axhline(PCE_SWEET_MIN, color='#1565C0', linestyle=':', linewidth=1.6,
               label=fr'PCE $\geq {PCE_SWEET_MIN}\%$')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95)
    plt.tight_layout()
    out = OUT_DIR / "tier2_eloss_pce_sweet_spot.png"
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f"  Saved: {out}")


def make_scaffold_family_bars(scaf_result):
    donor_top = scaf_result['donor_scaf_counts'].most_common(8)
    acc_top = scaf_result['acc_scaf_counts'].most_common(8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')

    for ax, top, color, label, n_total in [
        (axes[0], donor_top, COLORS['donor'], 'Donor', len(scaf_result['donors'])),
        (axes[1], acc_top, COLORS['acceptor'], 'Acceptor', len(scaf_result['accs'])),
    ]:
        _style_ax(ax, xlabel='', ylabel='Number of Tier-2 molecules')
        labels = [f"scaffold {i+1}" for i in range(len(top))]
        counts = [c for _, c in top]
        ax.bar(labels, counts, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for i, c in enumerate(counts):
            ax.annotate(f'{c}\n({100*c/n_total:.1f}%)', xy=(i, c), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    out = OUT_DIR / "tier2_scaffold_family_bars.png"
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f"  Saved: {out}")


def make_structure_grid(smiles_list, legends, out_path, mols_per_row=3, sub_size=(320, 280)):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_size,
                                legends=legends, returnPNG=False)
    img.save(str(out_path))
    log(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# STEP 5: representative candidates for structure grids
# ---------------------------------------------------------------------------
def pick_representative(hp, top_n_by_pce, scaf_result):
    """Pick a chemically diverse, high-performing representative set:
    the sweet-spot top pairs, but de-duplicated so the same donor or
    acceptor is not shown twice, and spanning distinct scaffold families
    where possible."""
    seen_donor_scaf, seen_acc_scaf = set(), set()
    donor_reps, acc_reps = [], []
    for _, row in top_n_by_pce.iterrows():
        d_s = murcko_scaffold_smiles(row['donor_smiles'])
        a_s = murcko_scaffold_smiles(row['acceptor_smiles'])
        if d_s not in seen_donor_scaf and len(donor_reps) < 6:
            donor_reps.append(row['donor_smiles'])
            seen_donor_scaf.add(d_s)
        if a_s not in seen_acc_scaf and len(acc_reps) < 6:
            acc_reps.append(row['acceptor_smiles'])
            seen_acc_scaf.add(a_s)
    return donor_reps, acc_reps


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    hp_eloss = load_tier2_with_eloss()
    scaf_result = scaffold_analysis(hp_eloss)
    valid, sweet, top = sweet_spot_analysis(hp_eloss)

    make_sweet_spot_scatter(valid, sweet)
    make_scaffold_family_bars(scaf_result)

    donor_reps, acc_reps = pick_representative(hp_eloss, top, scaf_result)
    if donor_reps:
        legends = []
        for s in donor_reps:
            row = hp_eloss[hp_eloss['donor_smiles'] == s].iloc[0]
            legends.append(f"PCE={row['surrogate_PCE']:.1f}%, SA-scaf")
        make_structure_grid(donor_reps, legends,
                             OUT_DIR / "tier2_representative_grid_donors.png", mols_per_row=3)
    if acc_reps:
        legends = []
        for s in acc_reps:
            row = hp_eloss[hp_eloss['acceptor_smiles'] == s].iloc[0]
            legends.append(f"PCE={row['surrogate_PCE']:.1f}%")
        make_structure_grid(acc_reps, legends,
                             OUT_DIR / "tier2_representative_grid_acceptors.png", mols_per_row=3)

    # also draw the single best sweet-spot pair (donor + acceptor together)
    if len(top) > 0:
        best = top.iloc[0]
        make_structure_grid(
            [best['donor_smiles'], best['acceptor_smiles']],
            [f"Best donor (PCE={best['surrogate_PCE']:.2f}%)",
             f"Best acceptor (E_loss={best['eloss_pred_physical']:.2f} eV)"],
            OUT_DIR / "tier2_best_pair.png", mols_per_row=2, sub_size=(400, 350))

    # ---- text summary -------------------------------------------------------
    enrich_df = scaf_result['enrich_df']
    with open(OUT_DIR / "tier2_chemical_summary.txt", "w", encoding="utf-8") as f:
        f.write("TIER-2 SHORTLIST CHEMICAL-INTERPRETATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tier-2 pairs: {len(hp_eloss):,}\n")
        f.write(f"Unique donors in Tier-2: {len(scaf_result['donors']):,} "
                f"(of 52,995 full donor library)\n")
        f.write(f"Unique acceptors in Tier-2: {len(scaf_result['accs']):,} "
                f"(of 53,719 full acceptor library)\n")
        f.write(f"Distinct Bemis-Murcko donor scaffolds in Tier-2: {scaf_result['n_donor_scaffolds']}\n")
        f.write(f"Distinct Bemis-Murcko acceptor scaffolds in Tier-2: {scaf_result['n_acc_scaffolds']}\n\n")

        f.write("Top donor scaffolds (Tier-2):\n")
        for s, c in scaf_result['donor_scaf_counts'].most_common(5):
            f.write(f"  {c:4d} ({100*c/len(scaf_result['donors']):.1f}%)  {s}\n")
        f.write("\nTop acceptor scaffolds (Tier-2):\n")
        for s, c in scaf_result['acc_scaf_counts'].most_common(5):
            f.write(f"  {c:4d} ({100*c/len(scaf_result['accs']):.1f}%)  {s}\n")

        f.write("\nSubstructure enrichment (Tier-2 vs full generated library):\n")
        f.write(enrich_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"Sweet-spot definition: E_loss <= {ELOSS_SWEET_MAX} eV and PCE >= {PCE_SWEET_MIN}%\n")
        f.write(f"Tier-2 pairs with defined E_loss: {len(valid):,}\n")
        f.write(f"Sweet-spot pairs: {len(sweet):,} ({100*len(sweet)/len(valid):.2f}% of those)\n\n")
        f.write("Top sweet-spot candidates:\n")
        f.write(top[['donor_smiles', 'acceptor_smiles', 'surrogate_PCE', 'surrogate_Jsc',
                      'surrogate_FF', 'surrogate_Voc', 'eloss_pred_physical',
                      'pair_tango']].to_string(index=False))
        f.write("\n")

    log(f"All outputs written to {OUT_DIR}")
    log(f"Total wall time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ADDENDUM (run separately via --addendum): recurrence + saturation checks
# ---------------------------------------------------------------------------
def addendum_recurrence_and_saturation():
    """Quantifies two follow-up chemical-interpretation questions that a
    raw Bemis-Murcko scaffold count does not answer on its own:

      (a) Is Tier-2 performance driven by a handful of highly-reused donor/
          acceptor building blocks, or is it spread across the shortlist?
      (b) Are the very-top single-pair PCE structures representative of the
          bulk Tier-2 chemistry, or are they comparatively small / weakly
          conjugated outliers favoured by the descriptor-based surrogate?
    """
    from rdkit.Chem import rdMolDescriptors

    hp = pd.read_csv(HP_CSV)
    donor_counts = hp['donor_smiles'].value_counts()
    acc_counts = hp['acceptor_smiles'].value_counts()

    donors_t2 = hp['donor_smiles'].unique().tolist()
    accs_t2 = hp['acceptor_smiles'].unique().tolist()
    d_full = pd.read_csv(DONOR_LIB_CSV, usecols=['smiles'])['smiles'].tolist()
    a_full = pd.read_csv(ACCEPTOR_LIB_CSV, usecols=['smiles'])['smiles'].tolist()

    def csp3_and_arom(smis):
        csp3, arom = [], []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            csp3.append(rdMolDescriptors.CalcFractionCSP3(m))
            arom.append(rdMolDescriptors.CalcNumAromaticRings(m))
        return np.array(csp3), np.array(arom)

    c_d_t2, ar_d_t2 = csp3_and_arom(donors_t2)
    c_a_t2, ar_a_t2 = csp3_and_arom(accs_t2)
    c_d_full, ar_d_full = csp3_and_arom(d_full)
    c_a_full, ar_a_full = csp3_and_arom(a_full)

    top10_donor_cov = 100 * donor_counts.head(10).sum() / len(hp)
    top10_acc_cov = 100 * acc_counts.head(10).sum() / len(hp)

    # -- recurrence bar chart -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    for ax, counts, color, label in [
        (axes[0], donor_counts.head(10), COLORS['donor'], 'Donor'),
        (axes[1], acc_counts.head(10), COLORS['acceptor'], 'Acceptor'),
    ]:
        _style_ax(ax, xlabel='', ylabel='Number of Tier-2 pairs containing this molecule')
        x = np.arange(len(counts))
        ax.bar(x, counts.values, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for i, c in enumerate(counts.values):
            ax.annotate(str(c), xy=(i, c), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(counts))])
        _plabel = 'a)' if ax is axes[0] else 'b)'
        ax.text(-0.06, 1.01, _plabel, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='bottom', ha='left')
    plt.tight_layout()
    out = OUT_DIR / "tier2_building_block_recurrence.png"
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f"  Saved: {out}")

    donor_counts.head(20).rename('n_tier2_pairs').to_csv(
        OUT_DIR / "tier2_top_recurring_donors.csv", header=True)
    acc_counts.head(20).rename('n_tier2_pairs').to_csv(
        OUT_DIR / "tier2_top_recurring_acceptors.csv", header=True)

    with open(OUT_DIR / "tier2_recurrence_and_saturation_summary.txt", "w", encoding="utf-8") as f:
        f.write("TIER-2 BUILDING-BLOCK RECURRENCE AND SATURATION CHECK\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Donor multiplicity per Tier-2 pair: mean={donor_counts.mean():.2f}, "
                f"median={donor_counts.median():.0f}, max={donor_counts.max()}\n")
        f.write(f"Acceptor multiplicity per Tier-2 pair: mean={acc_counts.mean():.2f}, "
                f"median={acc_counts.median():.0f}, max={acc_counts.max()}\n")
        f.write(f"Top-10 donors cover {top10_donor_cov:.1f}% of the 2,832 Tier-2 pairs.\n")
        f.write(f"Top-10 acceptors cover {top10_acc_cov:.1f}% of the 2,832 Tier-2 pairs.\n\n")

        f.write("Fraction-Csp3 (0=fully aromatic/conjugated backbone, 1=fully saturated):\n")
        f.write(f"  Tier-2 donors:      mean={c_d_t2.mean():.3f}  median={np.median(c_d_t2):.3f}\n")
        f.write(f"  Full donor library: mean={c_d_full.mean():.3f}  median={np.median(c_d_full):.3f}\n")
        f.write(f"  Tier-2 acceptors:      mean={c_a_t2.mean():.3f}  median={np.median(c_a_t2):.3f}\n")
        f.write(f"  Full acceptor library: mean={c_a_full.mean():.3f}  median={np.median(c_a_full):.3f}\n\n")

        f.write("Mean number of aromatic rings per molecule:\n")
        f.write(f"  Tier-2 donors:      {ar_d_t2.mean():.2f}\n")
        f.write(f"  Full donor library: {ar_d_full.mean():.2f}\n")
        f.write(f"  Tier-2 acceptors:      {ar_a_t2.mean():.2f}\n")
        f.write(f"  Full acceptor library: {ar_a_full.mean():.2f}\n\n")

        f.write("Top 10 recurring donors (SMILES, n Tier-2 pairs):\n")
        for smi, c in donor_counts.head(10).items():
            f.write(f"  {c:3d}  {smi}\n")
        f.write("\nTop 10 recurring acceptors (SMILES, n Tier-2 pairs):\n")
        for smi, c in acc_counts.head(10).items():
            f.write(f"  {c:3d}  {smi}\n")

    log("Recurrence/saturation addendum complete.")


if __name__ == "__main__" and os.environ.get("TIER2_ADDENDUM") == "1":
    addendum_recurrence_and_saturation()
