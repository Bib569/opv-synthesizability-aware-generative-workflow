#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
01_gan_training_donor.py  ->  gan_training_donor_optimized.png
=============================================================================
Required input CSVs (in --data-dir):
    gan_losses_donor_optimized.csv          cols: epoch, discriminator_loss, generator_loss
    discriminator_accuracy_donor_optimized.csv  cols: epoch, discriminator_accuracy_percent

Usage:
    python 01_gan_training_donor.py --data-dir data/figure_inputs
    #   -> default: reads data/figure_inputs/, writes results/regenerated_figures/
    python 01_gan_training_donor.py --data-dir DATA --outdir OUT
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common_style import (DPI, COLORS, style_ax, panel_label, need,
                           parse_common_args, resolve_dirs)

TAG = 'donor_optimized'


def _plot_gan_losses_on_ax(ax, df):
    epochs = df['epoch'].values
    d_loss = df['discriminator_loss'].values
    g_loss = df['generator_loss'].values
    n_epochs = len(epochs)

    style_ax(ax, xlabel='Epoch', ylabel='Loss')
    ax.fill_between(epochs, d_loss, g_loss, alpha=0.08, color='#9E9E9E', label='D\u2013G gap')
    ax.plot(epochs, d_loss, 'o-', color=COLORS['d_loss'], linewidth=2.8,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['d_loss'], label='Discriminator Loss', zorder=5)
    ax.plot(epochs, g_loss, 's-', color=COLORS['g_loss'], linewidth=2.8,
            markersize=9, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['g_loss'], label='Generator Loss', zorder=5)

    eq_val = np.log(2)
    ax.axhline(y=eq_val, color=COLORS['equilibrium'], linestyle='--',
               linewidth=2.2, alpha=0.85, zorder=3)
    eq_x = epochs[int(n_epochs * 0.6)]
    ax.annotate(r'Equilibrium ($\ln 2 \approx 0.693$)',
                xy=(eq_x, eq_val), xytext=(eq_x, eq_val - 0.004),
                fontsize=9.5, fontweight='bold', color=COLORS['equilibrium'],
                ha='center', va='top')

    all_loss = np.concatenate([d_loss, g_loss, [eq_val]])
    gap = abs(d_loss[-1] - g_loss[-1])
    y_offset = 0.002
    x_frac = 0.55 if gap < 0.005 else 0.70
    ann_x = epochs[0] + (epochs[-1] - epochs[0]) * x_frac
    ax.annotate(f'D = {d_loss[-1]:.4f}', xy=(epochs[-1], d_loss[-1]),
                xytext=(ann_x, d_loss[-1] + y_offset),
                fontsize=9, color=COLORS['d_loss'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['d_loss'], lw=1.2), ha='right')
    ax.annotate(f'G = {g_loss[-1]:.4f}', xy=(epochs[-1], g_loss[-1]),
                xytext=(ann_x, g_loss[-1] - y_offset),
                fontsize=9, color=COLORS['g_loss'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['g_loss'], lw=1.2), ha='right')

    ax.set_ylim(min(all_loss) - 0.005, max(all_loss) + 0.005)
    ax.set_xticks(epochs)
    ax.set_xticklabels([str(int(e)) for e in epochs])
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor='black',
              fancybox=False, ncol=1)
    panel_label(ax, 'a)', outside=True)


def _plot_discriminator_accuracy_on_ax(ax, df):
    epochs = df['epoch'].values
    accuracy = df['discriminator_accuracy_percent'].values
    n_epochs = len(epochs)

    style_ax(ax, xlabel='Epoch', ylabel='Accuracy (%)')
    ax.fill_between(epochs, 0, accuracy, alpha=0.12, color=COLORS['accuracy'])
    ax.plot(epochs, accuracy, 'o-', color=COLORS['accuracy'], linewidth=2.8,
            markersize=10, markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=COLORS['accuracy'], label='Discriminator Accuracy', zorder=5)

    ax.axhline(y=50.0, color=COLORS['target'], linestyle='--', linewidth=2.2,
               alpha=0.85, zorder=3)
    ax.annotate('Target Equilibrium (50%)',
                xy=(epochs[int(n_epochs * 0.3)], 50.0),
                xytext=(epochs[int(n_epochs * 0.3)], 55),
                fontsize=9.5, fontweight='bold', color=COLORS['target'],
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=COLORS['target'], lw=1.5))

    closest_idx = int(np.argmin(np.abs(accuracy - 50.0)))
    ax.plot(epochs[closest_idx], accuracy[closest_idx], '*',
            color='#FFD600', markersize=16, markeredgecolor='black',
            markeredgewidth=1.5, zorder=6)
    ax.annotate(f'Closest to 50%\n({accuracy[closest_idx]:.1f}%)',
                xy=(epochs[closest_idx], accuracy[closest_idx]),
                xytext=(epochs[closest_idx] + 0.5, accuracy[closest_idx] - 8),
                fontsize=8.5, fontweight='bold', color='#424242',
                arrowprops=dict(arrowstyle='->', color='#424242', lw=1.2))
    ax.annotate(f'{accuracy[-1]:.1f}%', xy=(epochs[-1], accuracy[-1]),
                xytext=(epochs[-1] - 0.8, accuracy[-1] + 5),
                fontsize=9.5, color=COLORS['accuracy'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['accuracy'], lw=1.2))

    ax.set_ylim(0, max(accuracy) + 10)
    ax.set_xticks(epochs)
    ax.set_xticklabels([str(int(e)) for e in epochs])
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=False)
    panel_label(ax, 'b)', outside=True)


def fig_gan_training(data_dir, outdir, tag):
    gan_csv = need(data_dir, f'gan_losses_{tag}.csv')
    disc_csv = need(data_dir, f'discriminator_accuracy_{tag}.csv')
    name = f'gan_training_{tag}.png'
    if gan_csv is None or disc_csv is None:
        miss = [f'gan_losses_{tag}.csv'] if gan_csv is None else []
        miss += [f'discriminator_accuracy_{tag}.csv'] if disc_csv is None else []
        print(f"  [SKIP] {name}: missing {', '.join(miss)}")
        return False
    gan_df = pd.read_csv(gan_csv)
    disc_df = pd.read_csv(disc_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
    fig.patch.set_facecolor('white')
    _plot_gan_losses_on_ax(ax1, gan_df)
    _plot_discriminator_accuracy_on_ax(ax2, disc_df)
    plt.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK]   {name}")
    return True


def main():
    args = parse_common_args("Figure 1: donor GAN training (losses + discriminator accuracy).")
    data_dir, outdir = resolve_dirs(args, __file__)
    print(f"data directory   : {data_dir}")
    print(f"output directory : {outdir}")
    fig_gan_training(data_dir, outdir, TAG)


if __name__ == "__main__":
    main()
