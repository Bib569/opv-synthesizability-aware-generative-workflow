#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_representative_figure.py
=============================

Builds tier2_representative_pair.png : a three-panel figure of representative
high-performing Tier-2 donor-acceptor pairs, with visible vertical separators
between panels and high-resolution structure rendering (side chains fully
visible).

Panels:
  (a) Headline / best all-around pair -- highest predicted PCE that
      simultaneously has high FF and low E_loss (the "sweet-spot" top row).
  (b) Most structurally distinct high-PCE pair -- the rank-1-by-PCE pair whose
      acceptor differs from panel (a); a tetracyanovinylidene/pyridyl acceptor.
  (c) Rediscovered real-NFA archetype -- a Y6/ITIC-type fused-ladder acceptor
      (indanone-dicyanovinylidene end caps on a dithienothiophene-type core),
      also predicted at PCE = 18.2%.

Every molecule is rendered individually with rdMolDraw2D (cairo) at high
resolution so that alkyl / cyanoalkyl solubilising side chains render clearly,
then the panels are composited with matplotlib so a labelled separator bar can
be drawn between them.

Reads only existing pipeline output; makes no new predictions.

Run from a fresh clone (inputs ship in data/figure_inputs/ and results/tier2_analysis/):
    python src/analysis/make_representative_figure.py
Needs RDKit and Pillow. Outputs go to results/tier2_analysis/.
(The manuscript Fig. 8 PNG in results/reference_figures_png/ is produced by
scripts/figures/12_tier2_representative_pair.py, not by this script.)
"""

from __future__ import annotations
import os
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
try:
    from rdkit.Chem import rdCoordGen
    _HAVE_COORDGEN = True
except Exception:
    _HAVE_COORDGEN = False
RDLogger.DisableLog("rdApp.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from io import BytesIO
from PIL import Image

_base = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_base, "..", ".."))   # src/analysis/ -> repo root
TIER2_DIR = os.path.join(_repo, "results", "tier2_analysis")
HP_CSV = os.path.join(_repo, "data", "figure_inputs", "high_performance_pairs.csv")
OUT_PNG = os.path.join(TIER2_DIR, "tier2_representative_pair.png")
OUT_PNG_LOCAL = os.path.join(TIER2_DIR, "tier2_representative_pair_final.png")

Y6_ACCEPTOR = ('CCC=C1C(=O)c2ccccc2C1=C(C#N)CCc1cc2c(s1)-c1c(ccc3c4c(ccc13)'
               'C(CCCC#N)(CCCC(C#N)=C1C(=CCCCC)C(=O)c3ccccc31)c1cc(C#N)sc1-4)'
               'C2(CCCC)CCCCCC')


def draw_mol_png(smiles, width=1100, height=760):
    """Render one molecule to a high-res PNG (numpy RGBA) with clear side chains.

    Uses CoordGen (Schrodinger-style) 2D coordinates when available, which
    spreads congested substituents -- in particular multiple nitrile / cyano
    groups on a cross-conjugated core -- far better than the default RDKit
    layout, eliminating the visual overlap where a C#N triple bond and its N
    label were drawn on top of a neighbouring carbon. Falls back to the
    standard CalcCoords layout if CoordGen is unavailable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if _HAVE_COORDGEN:
        try:
            rdCoordGen.AddCoords(mol)
        except Exception:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = d.drawOptions()
    opts.bondLineWidth = 2
    opts.minFontSize = 14
    opts.maxFontSize = 22
    opts.padding = 0.10
    # additionalPadding / multipleBondOffset help splay parallel triple-bond
    # strokes so stacked nitriles stay legible
    opts.multipleBondOffset = 0.18
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png = d.GetDrawingText()
    img = Image.open(BytesIO(png)).convert("RGBA")
    return np.asarray(img)


def main():
    hp = pd.read_csv(HP_CSV)

    # ----- panel (a): headline best all-around (sweet-spot top row) ----------
    sweet = pd.read_csv(os.path.join(TIER2_DIR,
                                     "tier2_sweet_spot_candidates.csv"))
    sweet = sweet.sort_values(["surrogate_PCE", "surrogate_FF", "eloss_pred_physical"],
                              ascending=[False, False, True])
    a_row = sweet.iloc[0]

    # ----- panel (b): highest-PCE pair whose acceptor differs from (a) -------
    other = hp[hp["acceptor_smiles"] != a_row["acceptor_smiles"]]
    b_row = other.sort_values("surrogate_PCE", ascending=False).iloc[0]

    # ----- panel (c): Y6-type ladder acceptor + its best donor partner -------
    ysub = hp[hp["acceptor_smiles"] == Y6_ACCEPTOR].sort_values(
        "surrogate_PCE", ascending=False)
    c_row = ysub.iloc[0]
    n_y6 = len(ysub)

    panels = [
        ("(a) Best all-around pair",
         a_row["donor_smiles"], a_row["acceptor_smiles"],
         a_row["surrogate_PCE"], a_row["surrogate_Jsc"], a_row["surrogate_FF"],
         a_row["surrogate_Voc"], a_row.get("eloss_pred_physical", float("nan")),
         "dicyanovinylidene / thienylpyridine acceptor"),
        ("(b) Highest-PCE pair (distinct acceptor)",
         b_row["donor_smiles"], b_row["acceptor_smiles"],
         b_row["surrogate_PCE"], b_row["surrogate_Jsc"], b_row["surrogate_FF"],
         b_row["surrogate_Voc"], float("nan"),
         "tetracyanovinylidene / pyridyl acceptor"),
        ("(c) Rediscovered Y6/ITIC-type ladder NFA",
         c_row["donor_smiles"], c_row["acceptor_smiles"],
         c_row["surrogate_PCE"], c_row["surrogate_Jsc"], c_row["surrogate_FF"],
         c_row["surrogate_Voc"], float("nan"),
         f"fused-ladder acceptor, indanone-dicyanovinylidene end caps (n={n_y6} Tier-2 pairs)"),
    ]

    # composite: 3 rows, each row = [donor img | + | acceptor img], with a
    # horizontal separator bar drawn between rows.
    # Each of the 3 panels is a stacked block: a text header on top, then the
    # donor|acceptor structure row beneath it. Using one gridspec row per PANEL
    # (with an internal 2-column structure row placed in its lower portion)
    # keeps the header text well clear of the "Donor"/"Acceptor" subplot titles,
    # which previously overlapped.
    fig = plt.figure(figsize=(14, 16.5))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(3, 1, hspace=0.10,
                             left=0.03, right=0.97, top=0.985, bottom=0.015)

    for i, (title, dsmi, asmi, pce, jsc, ff, voc, eloss, motif) in enumerate(panels):
        d_img = draw_mol_png(dsmi)
        a_img = draw_mol_png(asmi)

        # inner grid: row0 = header strip, row1 = structures (large).
        # A larger header:structure ratio + more vertical gap keeps the 3-line
        # header clear of the "Donor"/"Acceptor" subplot titles below it.
        inner = outer[i].subgridspec(2, 2, height_ratios=[1, 5],
                                     hspace=0.16, wspace=0.05)

        # header spanning both columns
        ax_hdr = fig.add_subplot(inner[0, :]); ax_hdr.axis("off")
        eloss_txt = "" if np.isnan(eloss) else fr", $E_{{\mathrm{{loss}}}}={eloss:.2f}$ eV"
        line1 = f"{title}"
        line2 = (f"PCE = {pce:.2f}%, $J_{{sc}}$ = {jsc:.1f} mA cm$^{{-2}}$, "
                 f"FF = {ff:.2f}, $V_{{oc}}$ = {voc:.2f} V{eloss_txt}")
        line3 = motif
        ax_hdr.text(0.0, 1.0, line1, fontsize=12.5, family="serif",
                    fontweight="bold", va="top", ha="left",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(0.0, 0.60, line2, fontsize=11, family="serif",
                    fontweight="bold", va="top", ha="left",
                    transform=ax_hdr.transAxes)
        ax_hdr.text(0.0, 0.22, line3, fontsize=10, family="serif",
                    style="italic", va="top", ha="left",
                    transform=ax_hdr.transAxes)

        ax_d = fig.add_subplot(inner[1, 0])
        ax_a = fig.add_subplot(inner[1, 1])
        ax_d.imshow(d_img); ax_d.axis("off")
        ax_a.imshow(a_img); ax_a.axis("off")
        ax_d.set_title("Donor", fontsize=12, fontweight="bold", family="serif")
        ax_a.set_title("Acceptor", fontsize=12, fontweight="bold", family="serif")

        # horizontal separator bar between panels
        if i < 2:
            sep_y = 1.0 - (i + 1) / 3.0 * (0.985 - 0.015) - 0.015
            fig.add_artist(plt.Line2D([0.02, 0.98], [sep_y, sep_y],
                                      color="black", linewidth=1.4,
                                      transform=fig.transFigure))

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PNG_LOCAL, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", OUT_PNG)
    print("Saved:", OUT_PNG_LOCAL)
    print()
    print("Panel (a) donor   :", a_row["donor_smiles"])
    print("Panel (a) acceptor:", a_row["acceptor_smiles"])
    print("Panel (b) donor   :", b_row["donor_smiles"])
    print("Panel (b) acceptor:", b_row["acceptor_smiles"])
    print("Panel (c) donor   :", c_row["donor_smiles"])
    print("Panel (c) acceptor:", c_row["acceptor_smiles"])


if __name__ == "__main__":
    main()
