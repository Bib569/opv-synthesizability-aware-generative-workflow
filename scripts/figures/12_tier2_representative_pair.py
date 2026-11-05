#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
12_tier2_representative_pair.py  ->  tier2_representative_pair.png
=============================================================================
Representative Tier-2 donor--acceptor structure panels. REQUIRES RDKit + Pillow.

Required input CSV (in --data-dir):
    tier2_representative_pair_data.csv
        cols: panel, title, motif, donor_smiles, acceptor_smiles,
              PCE, Jsc, FF, Voc, E_loss

Usage:
    python 12_tier2_representative_pair.py --data-dir data/figure_inputs
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, need, parse_common_args, resolve_dirs)


def fig_representative_pair(data_dir, outdir):
    name = 'tier2_representative_pair.png'
    csv = need(data_dir, 'tier2_representative_pair_data.csv')
    if csv is None:
        print(f"  [SKIP] {name}: missing tier2_representative_pair_data.csv")
        return False
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        RDLogger.DisableLog('rdApp.*')
        try:
            from rdkit.Chem import rdCoordGen
            have_coordgen = True
        except Exception:
            have_coordgen = False
    except Exception as e:
        print(f"  [SKIP] {name}: RDKit not available ({e})")
        return False
    from io import BytesIO
    from PIL import Image
    import matplotlib.image as _mpimg  # noqa: F401

    def draw_mol(smiles, width=1100, height=760):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.ones((height, width, 4), dtype=np.uint8) * 255
        if have_coordgen:
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
        opts.padding = 0.02
        opts.multipleBondOffset = 0.18
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return np.asarray(Image.open(BytesIO(d.GetDrawingText())).convert('RGBA'))

    panels = pd.read_csv(csv)
    fig = plt.figure(figsize=(14, 16.5))
    fig.patch.set_facecolor('white')
    outer = fig.add_gridspec(3, 1, hspace=0.10, left=0.03, right=0.97,
                             top=0.985, bottom=0.015)
    for i, row in panels.reset_index(drop=True).iterrows():
        d_img = draw_mol(row['donor_smiles'])
        a_img = draw_mol(row['acceptor_smiles'])
        inner = outer[i].subgridspec(2, 2, height_ratios=[1, 5], hspace=0.16, wspace=-0.15)
        ax_hdr = fig.add_subplot(inner[0, :]); ax_hdr.axis('off')
        eloss = row.get('E_loss', float('nan'))
        eloss_txt = '' if pd.isna(eloss) else fr", $E_{{\mathrm{{loss}}}}={eloss:.2f}$ eV"
        line1 = str(row['title'])
        line2 = (f"PCE = {row['PCE']:.2f}%, $J_{{sc}}$ = {row['Jsc']:.1f} mA cm$^{{-2}}$, "
                 f"FF = {row['FF']:.2f}, $V_{{oc}}$ = {row['Voc']:.2f} V{eloss_txt}")
        line3 = str(row['motif'])
        ax_hdr.text(0.0, 1.0, line1, fontsize=12.5, family='serif', fontweight='bold',
                    va='top', ha='left', transform=ax_hdr.transAxes)
        ax_hdr.text(0.0, 0.60, line2, fontsize=11, family='serif', fontweight='bold',
                    va='top', ha='left', transform=ax_hdr.transAxes)
        ax_hdr.text(0.0, 0.22, line3, fontsize=10, family='serif', style='italic',
                    va='top', ha='left', transform=ax_hdr.transAxes)
        ax_d = fig.add_subplot(inner[1, 0]); ax_a = fig.add_subplot(inner[1, 1])
        ax_d.imshow(d_img); ax_d.axis('off')
        ax_a.imshow(a_img); ax_a.axis('off')
        ax_d.set_title('Donor', fontsize=12, fontweight='bold', family='serif')
        ax_a.set_title('Acceptor', fontsize=12, fontweight='bold', family='serif')
        if i < 2:
            sep_y = 1.0 - (i + 1) / 3.0 * (0.985 - 0.015) - 0.015
            fig.add_artist(plt.Line2D([0.02, 0.98], [sep_y, sep_y], color='black',
                                      linewidth=1.4, transform=fig.transFigure))
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 12: Tier-2 representative pair (needs RDKit+Pillow).")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_representative_pair(data_dir, outdir)


if __name__ == "__main__":
    main()
