#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
03_combined_space_analysis.py  ->  combined_space_analysis.png  (+ .tex table)
=============================================================================
Includes the a)/b) panel labels and the enlarged (fontsize 12) histogram
"Mean:" text, matching the current manuscript figure exactly.

Required input CSVs (in --data-dir):
    donor_space4d_data.csv     cols: similarities, prop_scores, intdiv, combined_synthesizability
    acceptor_space4d_data.csv  cols: similarities, prop_scores, intdiv, combined_synthesizability

Usage:
    python 03_combined_space_analysis.py --data-dir data/figure_inputs
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from _common_style import (DPI, COLORS, style_ax, style_ax3d, panel_label_3d,
                           need, parse_common_args, resolve_dirs)


def _load_space4d_data(csv_path):
    """TRUE-4D per-molecule data:
        x=similarities, y=prop_scores, z=intdiv (per molecule),
        color=combined_synthesizability.
    """
    df = pd.read_csv(csv_path)
    sims = df['similarities'].values
    props = df['prop_scores'].values
    intdiv = df['intdiv'].values
    synth = df['combined_synthesizability'].values
    stats = {
        'n_plotted': len(sims), 'n_original': len(sims),
        'avg_similarity': float(np.mean(sims)),
        'avg_property_match': float(np.mean(props)),
        'avg_intdiv': float(np.mean(intdiv)),
        'avg_synth': float(np.mean(synth)),
        'global_fp_pca_fd': float('nan'),
    }
    return sims, props, intdiv, synth, stats


def _draw_space_analysis_block(fig, gs_row0, gs_l, gs_r, sims, props, zvals, cvals,
                                mol_type, block_label, panel_label=None):
    ax3d = fig.add_subplot(gs_row0, projection='3d')
    style_ax3d(ax3d, tick_labelsize=12)
    scatter = ax3d.scatter(sims, props, zvals, c=cvals, cmap='viridis',
                           s=10, alpha=0.6, depthshade=False)
    ax3d.set_xlabel('Similarity to\nTraining Set', fontsize=13, fontweight='bold', labelpad=12)
    ax3d.set_ylabel('Property Match\nScore', fontsize=13, fontweight='bold', labelpad=12)
    ax3d.set_zlabel('Internal Diversity, IntDiv\n(1 \u2212 max NN Tanimoto vs library)',
                    fontsize=12, fontweight='bold', labelpad=12)
    ax3d.set_title(block_label, fontsize=14, fontweight='bold', pad=2)
    if panel_label is not None:
        panel_label_3d(ax3d, panel_label)
    cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.55, pad=0.10)
    cbar.set_label('Combined Synthesizability', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    color = COLORS['donor'] if mol_type == 'donor' else COLORS['acceptor']
    ax_sim = fig.add_subplot(gs_l)
    style_ax(ax_sim, xlabel='Similarity Score', ylabel='Frequency', fontsize=13)
    ax_sim.hist(sims, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.5)
    ax_sim.axvline(np.mean(sims), color=COLORS['mean_line'], linestyle='-', linewidth=1.6,
                   label=f'Mean: {np.mean(sims):.3f}')
    ax_sim.legend(fontsize=12, loc='upper right')

    ax_prop = fig.add_subplot(gs_r)
    style_ax(ax_prop, xlabel='Property Match Score', ylabel='Frequency', fontsize=13)
    ax_prop.hist(props, bins=50, color=color, alpha=0.75, edgecolor='black', linewidth=0.5)
    ax_prop.axvline(np.mean(props), color=COLORS['mean_line'], linestyle='-', linewidth=1.6,
                    label=f'Mean: {np.mean(props):.3f}')
    ax_prop.legend(fontsize=12, loc='upper left')


def _write_space_table(stats, outdir):
    out = outdir / 'combined_space_analysis_table.tex'
    if stats is None:
        d = a = {'n_plotted': 'N/A', 'n_original': 'N/A', 'avg_similarity': float('nan'),
                 'avg_property_match': float('nan'), 'avg_intdiv': float('nan'),
                 'avg_synth': float('nan')}
        note = "% NOTE: space-analysis CSVs not found; placeholders written.\n"
    else:
        d, a, note = stats['donor'], stats['acceptor'], ""

    def cell(dv, av, prec=3):
        def one(v):
            if isinstance(v, str):
                return v
            if isinstance(v, float) and np.isnan(v):
                return 'N/A'
            return f'{v:,}' if isinstance(v, int) else f'{v:.{prec}f}'
        return one(dv), one(av)
    ns_d, ns_a = cell(d['n_plotted'], a['n_plotted'])
    si_d, si_a = cell(d['avg_similarity'], a['avg_similarity'])
    pm_d, pm_a = cell(d['avg_property_match'], a['avg_property_match'])
    iv_d, iv_a = cell(d['avg_intdiv'], a['avg_intdiv'])
    sy_d, sy_a = cell(d.get('avg_synth', float('nan')), a.get('avg_synth', float('nan')))
    with open(out, 'w', encoding='utf-8') as f:
        f.write(note)
        f.write("\\begin{table}[!htb]\n\\centering\n")
        f.write("\\caption{Chemical-space analysis summary statistics for the sampled donor "
                "and acceptor populations shown in Figure~\\ref{fig:combined_space_analysis}.}\n")
        f.write("\\label{tab:combined_space_analysis_stats}\n\\scriptsize\n"
                "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcc}\n\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Donor} & \\textbf{Acceptor} \\\\\n\\hline\n")
        f.write(f"Sample size, $n$ & {ns_d} & {ns_a} \\\\\n")
        f.write(f"Avg.\\ similarity to training set & {si_d} & {si_a} \\\\\n")
        f.write(f"Avg.\\ property match score & {pm_d} & {pm_a} \\\\\n")
        f.write(f"Avg.\\ internal diversity (IntDiv) & {iv_d} & {iv_a} \\\\\n")
        f.write(f"Avg.\\ combined synthesizability & {sy_d} & {sy_a} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  [OK]   combined_space_analysis_table.tex")


def fig_space_analysis(data_dir, outdir):
    d_csv = need(data_dir, 'donor_space4d_data.csv')
    a_csv = need(data_dir, 'acceptor_space4d_data.csv')
    if d_csv is None or a_csv is None:
        miss = ['donor_space4d_data.csv'] if d_csv is None else []
        miss += ['acceptor_space4d_data.csv'] if a_csv is None else []
        print(f"  [SKIP] combined_space_analysis.png: missing {', '.join(miss)}")
        _write_space_table(None, outdir)
        return False
    d_sim, d_prop, d_z, d_col, d_stats = _load_space4d_data(d_csv)
    a_sim, a_prop, a_z, a_col, a_stats = _load_space4d_data(a_csv)
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    outer = gridspec.GridSpec(2, 4, height_ratios=[3, 1.1], hspace=0.32, wspace=0.38,
                              left=0.045, right=0.98, top=0.93, bottom=0.07)
    _draw_space_analysis_block(fig, outer[0, 0:2], outer[1, 0], outer[1, 1],
                               d_sim, d_prop, d_z, d_col, 'donor',
                               f"Donor (n = {d_stats['n_plotted']:,})", panel_label='a)')
    _draw_space_analysis_block(fig, outer[0, 2:4], outer[1, 2], outer[1, 3],
                               a_sim, a_prop, a_z, a_col, 'acceptor',
                               f"Acceptor (n = {a_stats['n_plotted']:,})", panel_label='b)')
    fig.savefig(outdir / 'combined_space_analysis.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  [OK]   combined_space_analysis.png")
    _write_space_table({'donor': d_stats, 'acceptor': a_stats}, outdir)
    return True


def main():
    args = parse_common_args("Figure 3: combined donor+acceptor 4D chemical-space analysis.")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_space_analysis(data_dir, outdir)


if __name__ == "__main__":
    main()
