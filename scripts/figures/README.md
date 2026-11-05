# 12 single-figure scripts

This folder contains one standalone script per manuscript figure, plus a shared
style module. Together with the CSVs in `data/figure_inputs/` of this repository they
reproduce all 12 figures in the exact current publication
format (including the `a)`/`b)` panel labels and the enlarged histogram mean
text on the chemical-space figure).

## Files

| Script | Output figure | Input CSV(s) | Extra deps |
|--------|---------------|--------------|-----------|
| `01_gan_training_donor.py` | `gan_training_donor_optimized.png` | `gan_losses_donor_optimized.csv`, `discriminator_accuracy_donor_optimized.csv` | — |
| `02_gan_training_acceptor.py` | `gan_training_acceptor_optimized.png` | `gan_losses_acceptor_optimized.csv`, `discriminator_accuracy_acceptor_optimized.csv` | — |
| `03_combined_space_analysis.py` | `combined_space_analysis.png` (+ `.tex` table) | `donor_space4d_data.csv`, `acceptor_space4d_data.csv` | — |
| `04_device_properties.py` | `device_properties_distribution_method1_ridge_gp.png` | `device_properties_distribution_method1_ridge_gp_data.csv` | — |
| `05_disjoint_cv_r2.py` | `method1_disjoint_cv_r2.png` | `method1_disjoint_cv_report.csv` | — |
| `06_eloss_vs_pce.py` | `eloss_vs_pce_corrected.png` | `eloss_vs_pce_training_data.csv`, `eloss_vs_pce_generated_corrected_data.csv` | — |
| `07_sascore_rascore_tango_4d.py` | `sascore_rascore_tango_4d_combined.png` | `sascore_rascore_tango_4d_donor_data.csv`, `sascore_rascore_tango_4d_acceptor_data.csv` | — |
| `08_tier2_scaffold_family_bars.py` | `tier2_scaffold_family_bars.png` | `high_performance_pairs.csv` | RDKit |
| `09_tier2_building_block_recurrence.py` | `tier2_building_block_recurrence.png` | `high_performance_pairs.csv` | — |
| `10_frechet_baseline.py` | `frechet_baseline_plot.png` | `DMPNN.csv`, `Guzik_data.csv` | RDKit, scikit-learn, SciPy |
| `11_tier2_eloss_pce_sweet_spot.py` | `tier2_eloss_pce_sweet_spot.png` | `tier2_eloss_pce_sweet_spot_data.csv` | — |
| `12_tier2_representative_pair.py` | `tier2_representative_pair.png` | `tier2_representative_pair_data.csv` | RDKit, Pillow |

`_common_style.py` — shared style/colours/helpers. Keep it in this folder; every
figure script imports it. It is not a figure script.

`run_all.py` — optional convenience runner that calls all 12 scripts in turn.

## Requirements

Baseline (all 12 scripts import these):
```
pip install numpy pandas matplotlib
```
For the full 12/12, also install:
```
pip install rdkit scikit-learn scipy pillow
```
Scripts 8, 10, and 12 skip themselves cleanly with a message if their extra
dependency is missing; the other 9 figures still render.

## How to run

By default every script reads `data/figure_inputs/` and writes into
`results/regenerated_figures/` (both relative to the repository root, resolved
from the script location, so the working directory does not matter). Override
with `--data-dir` / `--outdir`. Run one figure:
```
python scripts/figures/03_combined_space_analysis.py
```
Run every figure (single entry point for this stage):
```
python scripts/figures/run_all.py
python scripts/figures/run_all.py --data-dir data/figure_inputs --outdir figures_out
```

The scripts do not modify the input CSVs. The PNGs they produce can be compared
with the committed reference outputs in `results/reference_figures_png/`.
