#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
run_all.py
=============================================================================
Convenience runner: executes all 12 single-figure scripts in this folder
against ONE data directory, writing every figure into ONE output directory.

This does NOT replace the 12 standalone scripts; each of them still runs on
its own. It only saves you from typing 12 commands.

Usage:
    python run_all.py --data-dir data/figure_inputs
    python run_all.py --data-dir DATA --outdir OUT
    python run_all.py --data-dir DATA --frechet-splits 200

Each figure script is self-contained; if an optional dependency (RDKit,
scikit-learn, SciPy, Pillow) is missing, only the figures that need it are
skipped with a clear message and the rest are still produced.
=============================================================================
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_gan_training_donor.py",
    "02_gan_training_acceptor.py",
    "03_combined_space_analysis.py",
    "04_device_properties.py",
    "05_disjoint_cv_r2.py",
    "06_eloss_vs_pce.py",
    "07_sascore_rascore_tango_4d.py",
    "08_tier2_scaffold_family_bars.py",
    "09_tier2_building_block_recurrence.py",
    "10_frechet_baseline.py",
    "11_tier2_eloss_pce_sweet_spot.py",
    "12_tier2_representative_pair.py",
]


def main():
    ap = argparse.ArgumentParser(description="Run all 12 single-figure scripts.")
    ap.add_argument('--data-dir', type=Path, default=None)
    ap.add_argument('--outdir', type=Path, default=None)
    ap.add_argument('--frechet-splits', type=int, default=200)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    data_dir = (args.data_dir or (repo_root / 'data' / 'figure_inputs')).resolve()
    common = ['--data-dir', str(data_dir)]
    if args.outdir is not None:
        common += ['--outdir', str(args.outdir.resolve())]

    print("=" * 78)
    print("RUN ALL 12 SINGLE-FIGURE SCRIPTS")
    print("=" * 78)
    for s in SCRIPTS:
        extra = ['--frechet-splits', str(args.frechet_splits)] if s.startswith('10_') else []
        print(f"\n>>> {s}")
        subprocess.run([sys.executable, str(here / s)] + common + extra, check=False)
    print("\n" + "=" * 78)
    print("Done. Check the output directory for the 12 figures.")
    print("=" * 78)


if __name__ == "__main__":
    main()
