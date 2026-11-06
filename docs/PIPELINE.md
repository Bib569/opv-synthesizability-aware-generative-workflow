# Pipeline, layout and entry points

This file maps each stage of the workflow described in the associated paper to its code, its driver, its inputs and the
files that are committed. Paths are relative to the repository root.

## Layout

```
src/
  generation/          GAN_240_final.py            stage 1: 13-approach generation (12 routes + Training_Set_Augmentation), pairing, Ridge+isotonic+GP surrogate, tiering, disjoint CV
  postprocessing/      merge_rank_outputs_final.py merges the 240 per-rank outputs (--stage generation | postgen)
  postgen/             postgen_pipeline.py         stage 2: metrics, property models, E_loss gate, retrosynthesis audit
  synthesizability/    synthesizability_scoring.py stage 3: SAScore / RAScore / combined score
  plotting/            manuscript_plots.py, manuscript_plots_revised.py   exploratory plots; producers of some figure-input CSVs
  analysis/            tier2_chemical_analysis.py, frechet_baseline_analysis.py,
                       create_eloss_vs_pce_corrected.py, make_representative_figure.py       stage 4
scripts/figures/       01..12_*.py, run_all.py, _common_style.py                                stage 5 (manuscript figures)
hpc/                   setup_run_dir.sh, sub_140.sh, sub_postgen_full_140.sh, sub_synth_scoring_240.sh   SLURM drivers
data/
  input/               DMPNN.csv (300 device records), Guzik_data.csv        training / reference data
  figure_inputs/       CSVs that scripts/figures/ reads (a copy of DMPNN.csv and Guzik_data.csv is kept here for script 10)
  generated_libraries/ donor/acceptor libraries with predictions and synthesizability scores
  large_inputs/        NOT in git: multi-GB HPC outputs (see below); the scripts that need them look here
results/
  summary/ postgen/ synthesizability/ tier2_analysis/   committed output tables and reports quoted in the manuscript
  reference_figures_png/                                PNG outputs of scripts/figures/ (byte-identical to a fresh run)
  regenerated_figures/                                  default output folder of scripts/figures/ (git-ignored)
docs/                  this file
CHANGELOG.md           every change made to the original code
```

## Entry points

| Stage | What it does | Entry point | Runs from a fresh clone? |
|---|---|---|---|
| 5. Manuscript figures | Redraws the 12 figure PNGs from `data/figure_inputs/` | `python scripts/figures/run_all.py` (or one `scripts/figures/NN_*.py`) | Yes. Needs numpy, pandas, matplotlib; RDKit for 08, 10, 12; scikit-learn and SciPy for 10; Pillow for 12. |
| 4. Analysis | Fig. S1 baseline data and plot | `python src/analysis/frechet_baseline_analysis.py` | Yes (inputs in `data/input/`) |
| 4. Analysis | Tier-2 representative-pair panels | `python src/analysis/make_representative_figure.py` | Yes (inputs in `data/figure_inputs/`, `results/tier2_analysis/`) |
| 4. Analysis | Tier-2 chemistry, scaffold tables, sweet spot | `python src/analysis/tier2_chemical_analysis.py` | No: needs `data/large_inputs/pair_predictions.csv` |
| 4. Analysis | E_loss vs PCE data for Fig. 9 | `python src/analysis/create_eloss_vs_pce_corrected.py` | No: needs two multi-GB files in `data/large_inputs/` (Fig. 9 itself redraws from `data/figure_inputs/` with script 06) |
| 1. Generation (+2, plots) | 240-rank generation, surrogate, tiering, then stage 2 and plots | `sh hpc/setup_run_dir.sh && cd run && sbatch sub_140.sh` | HPC only |
| 2. Post-generation | Re-run of stage 2 on an existing stage-1 pool | `cd run && sbatch sub_postgen_full_140.sh` | HPC only |
| 3. Synthesizability | SAScore / RAScore scoring of all unique molecules | `cd run && sbatch sub_synth_scoring_240.sh` | HPC only |

`hpc/setup_run_dir.sh` is needed because the stage 1 to 3 scripts were written for one flat working directory
(they read `DMPNN.csv`, `Guzik_data.csv`, `sascorer.py` and `fpscores.pkl.gz` from the working directory and
`manuscript_plots.py` reads `<script dir>/merged`). The helper copies the scripts and input data into `run/`
so that none of them had to be edited. Set `CONDA_ACTIVATE` to the `bin/activate` script of your conda
installation (default `$HOME/miniconda3/bin/activate`); the environment must be called `RAScore`.
`runs/22051` in the drivers is the SLURM job ID of the original stage-1 run; replace it with your own.

## Inputs that are not in the repository

| File | Size | Produced by | Used by |
|---|---|---|---|
| `generated_molecules_with_predictions.csv` | about 7.4 GB (11,988,459 pairs, `surrogate_*`, `pair_tango`) | stage 1 (`hpc/sub_140.sh`) | `create_eloss_vs_pce_corrected.py`, `postgen_pipeline.py`, `synthesizability_scoring.py`; also source of the Tier counts and of the 133,943 / 1,584 sensitivity numbers |
| `pair_predictions.csv` | about 5.2 GB (`eloss_pred_physical`, `dev_*`) | stage 2 (`hpc/sub_postgen_full_140.sh` + `merge_rank_outputs_final.py --stage postgen`) | `tier2_chemical_analysis.py`, `create_eloss_vs_pce_corrected.py` |
| full Method 1 surrogate table for Fig. 6 | about 850 MB | stage 1 | `data/figure_inputs/device_properties_distribution_method1_ridge_gp_data.csv` is a 500,000-row subsample of it |
| trained models, `models/fcd`, `models/aizynthfinder` | not retained | cluster run | stage 2 |

Place the first two under `data/large_inputs/` if you regenerate or obtain them from the authors.

## Figures: PNG from the scripts

The scripts in `scripts/figures/` write PNGs, and the PNGs are byte-identical to `results/reference_figures_png/`.
The manuscript sources and the vector PDF versions of the figures are not distributed in this repository, and the
figure numbering follows the associated paper. Fig. 1 (the workflow scheme) is a drawn schematic with no source in
this repository.

## Known inconsistencies carried over from the source material (not changed)

- `src/analysis/frechet_baseline_analysis.py` quotes 483 acceptor reference molecules in its docstring and
  summary, while `results/postgen/generative_metrics.csv` and SI Table S1 give 481.
- `data/input/Guzik_data.csv` is a third-party-derived literature dataset; confirm its licence before redistribution.
- `synthesizability_scoring.py` falls back to a heuristic RAScore when the `rascore` package is missing. The
  committed run (`results/synthesizability/output_metrics_synthesizability.txt`) reports `RAScore backend: heuristic`.
