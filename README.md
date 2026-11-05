# opv-synthesizability-aware-generative-workflow

**Synthesizability-Aware Hybrid Generative Workflow for Donor-Acceptor Discovery in Organic Photovoltaics**

Bibhas Das and Anirban Mondal, Department of Chemistry, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India.

Repository name: `opv-synthesizability-aware-generative-workflow`. This repository holds the manuscript sources
(`manuscript/main.tex`, `manuscript/si.tex`), the code of the workflow, the processed data behind every data figure, the
generated donor and acceptor libraries, and the scripts that redraw the figures.

## Abstract

Molecular discovery in organic photovoltaics (OPVs) requires simultaneous optimization of donor and acceptor
structures, molecular novelty, synthetic practicality, and device-level performance. We present a
synthesizability-aware hybrid molecular-generation and screening workflow for OPV donor-acceptor discovery. Twelve
complementary proposal and augmentation routes produced class-specific molecular libraries that, after structural
validation, canonicalization, deduplication, and adaptive novelty filtering, yielded 52,995 unique donor candidates and
53,719 unique acceptor candidates with no exact matches to the reference structures. The libraries exhibited substantial
scaffold and fingerprint diversity while retaining a measurable relationship to the OPV reference domain. Synthetic
practicality was assessed using SAScore, RAScore, and a TANGO-inspired building-block compatibility heuristic; 85.0% of
donors and 99.3% of acceptors satisfied the operational accessibility screen. Pairing the filtered libraries produced
11,988,459 donor-acceptor combinations, ranked using a ridge-regression surrogate with isotonic recalibration and
Gaussian-process residual correction trained on 300 curated device records. A total of 2,832 Tier-2 candidate pairs
satisfied joint predicted performance and compatibility criteria, including a small subset with predicted high
efficiency and low energy loss. Structure-disjoint validation revealed target-dependent transferability, with
particularly limited quantitative reliability for fill factor and open-circuit voltage. The reported device properties
are therefore interpreted as screening-based prioritization signals rather than experimentally realized performance. The
workflow provides a chemically interpretable route for expanding OPV chemical space and prioritizing donor-acceptor
pairs for experimental or higher-fidelity computational follow-up.

(Abstract text from `manuscript/main.tex`, with LaTeX markup removed.)

## Table of contents

1. [What can and cannot be reproduced from this repository](#what-can-and-cannot-be-reproduced-from-this-repository)
2. [Pipeline overview](#pipeline-overview)
3. [Repository layout](#repository-layout)
4. [Figure and table to script to data mapping](#figure-and-table-to-script-to-data-mapping)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [Full reproduction steps](#full-reproduction-steps)
8. [Expected outputs](#expected-outputs)
9. [Compute notes for the HPC / SLURM stages](#compute-notes-for-the-hpc--slurm-stages)
10. [Data dictionary](#data-dictionary)
11. [Limitations / not reproducible from provided files](#limitations--not-reproducible-from-provided-files)
12. [Data availability, licence and citation](#data-availability-licence-and-citation)

## What can and cannot be reproduced from this repository

| Part | Reproducible from a fresh clone? |
|---|---|
| Redrawing 12 of the 13 figures (Figs. 2 to 9 and S1 to S4) from the CSV files in `data/figure_inputs/` | Yes, on a laptop, in a few minutes (see [Quick start](#quick-start)) |
| The Fig. S1 baseline (`src/analysis/frechet_baseline_analysis.py`) and the Fig. 8 panels (`src/analysis/make_representative_figure.py`) | Yes |
| Fig. 1 (`manuscript/figures/Scheme.pdf`) | No. It is a drawn schematic and its source is not in the repository |
| Stage 1 (generation, 240 ranks), stage 2 (post-generation metrics) and stage 3 (synthesizability scoring) | Not locally. They were run on a SLURM cluster; the scripts and drivers are included, but the multi-GB outputs and trained models are not |
| Typography of the manuscript PDFs | No. The scripts write PNG files with a different text setup than the PDFs in `manuscript/figures/` (see [Limitations](#limitations--not-reproducible-from-provided-files)) |

## Pipeline overview

```
 DMPNN.csv (300 device records) + Guzik_data.csv            data/input/
                       |
                       v
 Stage 1  src/generation/GAN_240_final.py              driver hpc/sub_140.sh (240 SLURM ranks)
          12 generation routes (+ training-set augmentation), validation, canonicalisation,
          deduplication, adaptive novelty filter, pairing of donors x acceptors,
          Ridge + isotonic recalibration + Gaussian-process residual surrogate,
          Tier-1 / Tier-2 selection, structure-disjoint cross-validation
          merge: src/postprocessing/merge_rank_outputs_final.py --stage generation
                       |
                       v
 Stage 2  src/postgen/postgen_pipeline.py              driver hpc/sub_postgen_full_140.sh
          generative metrics, property models and model-selection horse race, E_loss gate,
          retrosynthesis audit
          merge: src/postprocessing/merge_rank_outputs_final.py --stage postgen
                       |
                       v
 Stage 3  src/synthesizability/synthesizability_scoring.py   driver hpc/sub_synth_scoring_240.sh
          SAScore, RAScore, combined synthesizability for every unique molecule
                       |
                       v
 Stage 4  src/analysis/ (tier2_chemical_analysis.py, frechet_baseline_analysis.py,
          create_eloss_vs_pce_corrected.py, make_representative_figure.py)
          Tier-2 chemistry, split-half baseline, E_loss vs PCE data, representative pairs
                       |
                       v
 Stage 5  scripts/figures/01..12_*.py (run_all.py)     reads data/figure_inputs/  ->  PNG figures
```

More detail on entry points, working-directory conventions and the inputs that are not in the repository is in
[`docs/PIPELINE.md`](docs/PIPELINE.md).

## Repository layout

```
manuscript/            main.tex, si.tex, references.bib, figures/ (13 PDFs included by the .tex files)
src/                   pipeline code (generation, postprocessing, postgen, synthesizability, plotting, analysis)
scripts/figures/       one script per figure (01..12), run_all.py, _common_style.py
hpc/                   SLURM drivers and setup_run_dir.sh
data/input/            DMPNN.csv, Guzik_data.csv
data/figure_inputs/    CSV files read by scripts/figures/
data/generated_libraries/   donor and acceptor libraries with predictions and synthesizability scores
data/large_inputs/     placeholder for multi-GB files that are not distributed
results/               tables and reports written by the pipeline and quoted in the manuscript
results/reference_figures_png/   PNG outputs of scripts/figures/ for comparison
docs/                  PIPELINE.md, DATA_AVAILABILITY.md
CHANGELOG.md           every change made to the original code when this repository was assembled
```

## Figure and table to script to data mapping

All scripts are in `scripts/figures/`. Input CSV files are in `data/figure_inputs/`. Each script writes a PNG with the
same base name as the PDF that the manuscript includes from `manuscript/figures/`.

| Item | Manuscript file (`manuscript/figures/`) | Script | Input data | Upstream producer of the data |
|---|---|---|---|---|
| Fig. 1 workflow scheme | `Scheme.pdf` | none | none | not reproducible (no source) |
| Fig. 2 donor GAN training | `gan_training_donor_optimized.pdf` | `01_gan_training_donor.py` | `gan_losses_donor_optimized.csv`, `discriminator_accuracy_donor_optimized.csv` | `src/generation/GAN_240_final.py` and `src/plotting/manuscript_plots.py` |
| Fig. 3 acceptor GAN training | `gan_training_acceptor_optimized.pdf` | `02_gan_training_acceptor.py` | `gan_losses_acceptor_optimized.csv`, `discriminator_accuracy_acceptor_optimized.csv` | same |
| Fig. 4 chemical-space analysis | `combined_space_analysis.pdf` | `03_combined_space_analysis.py` | `donor_space4d_data.csv`, `acceptor_space4d_data.csv` | producer script not in the repository |
| Fig. 5 SAScore x RAScore x TANGO-inspired | `sascore_rascore_tango_4d_combined.pdf` | `07_sascore_rascore_tango_4d.py` | `sascore_rascore_tango_4d_donor_data.csv`, `sascore_rascore_tango_4d_acceptor_data.csv` | `src/plotting/manuscript_plots_revised.py` |
| Fig. 6 predicted PCE, Jsc, FF, Voc distributions | `device_properties_distribution_method1_ridge_gp.pdf` | `04_device_properties.py` | `device_properties_distribution_method1_ridge_gp_data.csv` (500,000-row subsample) | stage 1; subsampling script not in the repository |
| Fig. 7 structure-disjoint held-out R2 | `method1_disjoint_cv_r2.pdf` | `05_disjoint_cv_r2.py` | `method1_disjoint_cv_report.csv` | stage 1 (`GAN_240_final.py`) |
| Fig. 8 representative Tier-2 pairs | `tier2_representative_pair.pdf` | `12_tier2_representative_pair.py` | `tier2_representative_pair_data.csv` | `src/analysis/make_representative_figure.py` |
| Fig. 9 E_loss vs PCE | `eloss_vs_pce_corrected.pdf` | `06_eloss_vs_pce.py` | `eloss_vs_pce_training_data.csv`, `eloss_vs_pce_generated_corrected_data.csv` | `src/analysis/create_eloss_vs_pce_corrected.py` (needs the multi-GB inputs) |
| Fig. S1 split-half baseline | `frechet_baseline_plot.pdf` | `10_frechet_baseline.py` | `DMPNN.csv`, `Guzik_data.csv` (recomputes 200 split-half draws) | `src/analysis/frechet_baseline_analysis.py` |
| Fig. S2 Tier-2 scaffold families | `tier2_scaffold_family_bars.pdf` | `08_tier2_scaffold_family_bars.py` | `high_performance_pairs.csv` | stage 1 Tier-2 filter; `src/analysis/tier2_chemical_analysis.py` |
| Fig. S3 building-block recurrence | `tier2_building_block_recurrence.pdf` | `09_tier2_building_block_recurrence.py` | `high_performance_pairs.csv` | same |
| Fig. S4 E_loss vs PCE sweet spot | `tier2_eloss_pce_sweet_spot.pdf` | `11_tier2_eloss_pce_sweet_spot.py` | `tier2_eloss_pce_sweet_spot_data.csv` | `src/analysis/tier2_chemical_analysis.py` (needs the multi-GB inputs) |
| Table S1 consolidated metrics | typed into `manuscript/si.tex` | none | values in `results/postgen/generative_metrics.csv`, `results/synthesizability/output_metrics_synthesizability.txt`, `results/postgen/output_metrics_postgen.txt`, `results/summary/output_metrics_generation.txt`, `results/tier2_analysis/frechet_baseline_summary.txt` | stages 1 to 4 |

`scripts/figures/03_combined_space_analysis.py` also writes `combined_space_analysis_table.tex`, a LaTeX table of the
Fig. 4 summary statistics. `main.tex` contains no tables.

Where the manuscript quotes a number, `results/` holds the file that contains it, except for the numbers listed under
[Limitations](#limitations--not-reproducible-from-provided-files).

## Installation

Python 3.12 was used. The pinned versions in `requirements.txt` are those reported by `py -3.12 -m pip list`.

```
python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

Conda users can run `conda env create -f environment.yml` instead (same pins, installed through pip inside the
environment).

`requirements.txt` covers `scripts/figures/` and the local scripts in `src/analysis/`. The HPC stages need more packages
and are not pinned; see [Compute notes](#compute-notes-for-the-hpc--slurm-stages).

## Quick start

From the repository root, with the environment active:

```
python scripts/figures/run_all.py
```

This reads `data/figure_inputs/` and writes the 12 PNG figures and `combined_space_analysis_table.tex` to
`results/regenerated_figures/`. The working directory does not matter. To draw one figure:

```
python scripts/figures/03_combined_space_analysis.py
```

Each script accepts `--data-dir` and `--outdir`. The PNG files in `results/reference_figures_png/` are the outputs
that the same scripts and CSV files produced when this repository was assembled.

## Full reproduction steps

1. **Figures (local).** Follow [Installation](#installation) and [Quick start](#quick-start). Compare the new PNG files
   with `results/reference_figures_png/`.
2. **Fig. S1 baseline data (local).** `python src/analysis/frechet_baseline_analysis.py` reads `data/input/DMPNN.csv` and
   `data/input/Guzik_data.csv` and writes to `results/tier2_analysis/`.
3. **Fig. 8 panels (local).** `python src/analysis/make_representative_figure.py` reads
   `data/figure_inputs/high_performance_pairs.csv` and `results/tier2_analysis/tier2_sweet_spot_candidates.csv` and
   writes to `results/tier2_analysis/`.
4. **Tier-2 chemistry and E_loss data (needs HPC outputs).** `src/analysis/tier2_chemical_analysis.py` and
   `src/analysis/create_eloss_vs_pce_corrected.py` need the multi-GB files described in
   [`data/large_inputs/README.md`](data/large_inputs/README.md). The corresponding results and figure inputs are
   already in `results/tier2_analysis/` and `data/figure_inputs/`.
5. **Generation, post-generation metrics and synthesizability scoring (HPC).** On a SLURM cluster:
   ```
   sh hpc/setup_run_dir.sh          # builds run/ with the flat layout the stage 1-3 scripts expect
   cd run
   export CONDA_ACTIVATE=/path/to/conda/bin/activate    # environment named RAScore
   sbatch sub_140.sh                # stage 1 (+ stage 2 and plots)
   sbatch sub_postgen_full_140.sh   # stage 2 alone, on an existing stage 1 pool
   sbatch sub_synth_scoring_240.sh  # stage 3
   ```
   `runs/22051` in the drivers is the SLURM job ID of the original stage 1 run; replace it with your own. Details are in
   [`docs/PIPELINE.md`](docs/PIPELINE.md).
6. **Manuscript.** `manuscript/main.tex` and `manuscript/si.tex` include figures from `manuscript/figures/` and use
   `references.bib` (BibTeX, `achemso` style). See the compile note under
   [Limitations](#limitations--not-reproducible-from-provided-files).

Steps 5 and 6 have not been run as part of assembling this repository.

## Expected outputs

| Command | Output |
|---|---|
| `python scripts/figures/run_all.py` | In `results/regenerated_figures/`: `gan_training_donor_optimized.png`, `gan_training_acceptor_optimized.png`, `combined_space_analysis.png`, `combined_space_analysis_table.tex`, `device_properties_distribution_method1_ridge_gp.png`, `method1_disjoint_cv_r2.png`, `eloss_vs_pce_corrected.png`, `sascore_rascore_tango_4d_combined.png`, `tier2_scaffold_family_bars.png`, `tier2_building_block_recurrence.png`, `frechet_baseline_plot.png`, `tier2_eloss_pce_sweet_spot.png`, `tier2_representative_pair.png` |
| `python src/analysis/frechet_baseline_analysis.py` | `results/tier2_analysis/frechet_baseline_donor.csv`, `frechet_baseline_acceptor.csv`, `frechet_baseline_summary.txt` |
| `python src/analysis/make_representative_figure.py` | `results/tier2_analysis/tier2_representative_pair.png` and `tier2_representative_pair_final.png` (these two files are not byte-identical to the Fig. 8 PNG written by `scripts/figures/12_tier2_representative_pair.py`; they were not compared further) |

## Compute notes for the HPC / SLURM stages

These stages cannot be rerun locally. What the repository records about the original runs:

* Each driver in `hpc/` requests 5 nodes, 48 tasks per node, 240 tasks in total with 1 CPU each, and the partition
  `highmemory`. Time limits are 240 hours for `sub_140.sh` and `sub_postgen_full_140.sh` and 24 hours for
  `sub_synth_scoring_240.sh`. Partition names and limits are specific to the original cluster; adapt them.
* The scripts are launched with `srun --ntasks=240` and `--distributed`. Each of the 240 ranks writes its own output; the
  merge script `src/postprocessing/merge_rank_outputs_final.py` pools them (`--stage generation` or `--stage postgen`).
* The comments in `src/generation/GAN_240_final.py` state that the original generation run took more than 22 hours
  end to end, with some ranks finishing in about 1 hour and others taking more than 20 hours.
* Peak output sizes: about 7.4 GB for the stage 1 pair table and about 5.2 GB for the stage 2 pair table. Memory
  figures per rank were not recorded.
* The conda environment on the cluster was called `RAScore`. Its package versions were not recorded, so the HPC
  environment is not pinned. The packages that the stage 1 to 3 scripts import are: numpy, pandas, scipy, scikit-learn,
  matplotlib, seaborn, RDKit, Pillow, joblib, optuna, mpi4py, and (optionally, with fallbacks in the code) tensorflow,
  chemprop, fcd, selfies, rascore, aizynthfinder, syba, scscore, plus RDKit's contributed `sascorer.py` and
  `fpscores.pkl.gz`. `hpc/setup_run_dir.sh` copies the RDKit files when it finds them.
* Trained models for stage 2 (`models/postgen`, `models/fcd`, `models/aizynthfinder`) were not retained.

## Data dictionary

Paths are relative to the repository root. Column descriptions follow the column names and the code that writes them; a
description is given only where it could be confirmed from the code or the manuscript. "Surrogate" means the Ridge +
isotonic + Gaussian-process residual model. Units: PCE in percent, Jsc in mA cm^-2, FF as a fraction, Voc and energies in
V and eV.

### `data/input/` (and identical copies in `data/figure_inputs/`)

| File | Rows x columns | Content |
|---|---|---|
| `DMPNN.csv` | 300 x 61 | Training set of 300 curated device records. Columns: `Ref`; acceptor descriptors (`Acceptor_Compounds`, `HOMO (eV)`, `HOMO-1 (eV)`, `LUMO (eV)`, `LUMO+1 (eV)`, `H-L Gap (eV)`, dipole moment, wavelength, `E_S`, `Osc. strength`, `E_b`, `Q20`, `A_EA` and related differences and inverses); the same descriptors for the donor with prefix `D_`; `ΔHOMO`, `ΔLUMO`, `ΔG_CR`, `ΔG_CS`, `ΔG_CT`; device values `Jsc(mA/cm2)`, `Voc(V)`, `FF`, `PCE_max(%)`; `V_OC_sim`, `E_loss`; `acceptor(SMILES)`, `donor(SMILES)`. Some header characters (for example the dipole and wavelength symbols) are stored in a legacy encoding and are kept byte-identical because the scripts look the columns up by name. |
| `Guzik_data.csv` | 558 x 45 | Literature device and material dataset: reference and DOI, donor and acceptor names and frontier-orbital energies, processing and device-stack parameters, mobilities, `Voc (V)`, `Jsc (mA cm^-2)`, `FF (%)`, `PCE (%)`, `calculated PCE (%)`, and donor and acceptor SMILES and SELFIES. Source and licence not confirmed (see [Data availability](#data-availability-licence-and-citation)). |

### `data/figure_inputs/` (the rest)

| File | Rows x columns | Columns |
|---|---|---|
| `gan_losses_donor_optimized.csv`, `gan_losses_acceptor_optimized.csv` | 5 x 3 | `epoch`, `discriminator_loss`, `generator_loss` |
| `discriminator_accuracy_donor_optimized.csv`, `discriminator_accuracy_acceptor_optimized.csv` | 5 x 2 | `epoch`, `discriminator_accuracy_percent` |
| `donor_space4d_data.csv`, `acceptor_space4d_data.csv` | 20,000 x 4 each | Per-molecule values for a 20,000-molecule sample: `similarities` (similarity to the training set, x axis of Fig. 4), `prop_scores` (property-match score, y axis), `intdiv` (internal diversity IntDiv, 1 minus the maximum nearest-neighbour Tanimoto similarity within the library, z axis), `combined_synthesizability` (colour) |
| `sascore_rascore_tango_4d_donor_data.csv`, `sascore_rascore_tango_4d_acceptor_data.csv` | 20,000 x 5 each | `smiles`, `SAScore`, `RAScore`, `tango_inspired_screen` (TANGO-inspired building-block compatibility score), `combined_synthesizability` |
| `device_properties_distribution_method1_ridge_gp_data.csv` | 500,000 x 4 | `surrogate_PCE`, `surrogate_Jsc`, `surrogate_FF`, `surrogate_Voc`: surrogate predictions for 500,000 of the 11,988,459 pairs |
| `method1_disjoint_cv_report.csv` | 24 x 9 | `property` (HOMO, LUMO, Eg of donor and acceptor; PCE, Jsc, FF, Voc), `split_kind` (`donor_disjoint`, `acceptor_disjoint`, `scaffold_disjoint`), `n_samples`, `n_groups`, `n_splits_used`, `heldout_r2`, `heldout_mae`, `heldout_rmse`, `status` |
| `eloss_vs_pce_training_data.csv` | 300 x 3 | `E_loss`, `PCE_max_pct`, `excluded_unphysical` (True when `E_loss` is below the 0.25 eV floor) for the 300 reference records |
| `eloss_vs_pce_generated_corrected_data.csv` | 20,000 x 2 | `E_loss_pred_physical`, `surrogate_PCE` for a 20,000-pair sample of the generated pairs |
| `high_performance_pairs.csv` | 2,832 x 32 | The Tier-2 pairs. `donor_smiles`, `acceptor_smiles`; predicted `donor_HOMO_donor`, `donor_LUMO_donor`, `donor_Eg_donor`, `acceptor_HOMO_acceptor`, `acceptor_LUMO_acceptor`, `acceptor_Eg_acceptor` (eV); `donor_M_gmol`, `acceptor_M_gmol` (g/mol); `PCE`, `Jsc`, `FF`, `Voc` (equal to the `surrogate_*` columns in every row); `surrogate_PCE`, `surrogate_Jsc`, `surrogate_FF`, `surrogate_Voc`; `donor_tango`, `acceptor_tango`, `pair_tango` (equal to the smaller of the two in every row); `tier` (`Tier2_Promising` in every row). The 10 `diag_legacy_*` columns are diagnostics from earlier physics-based and SISSO++ models; they are not used for any reported result. |
| `tier2_eloss_pce_sweet_spot_data.csv` | 2,832 x 3 | `E_loss`, `PCE`, `is_sweet` (1 for the 33 pairs with `E_loss` <= 0.55 eV and PCE >= 17.0%) |
| `tier2_representative_pair_data.csv` | 3 x 10 | `panel`, `title`, `motif`, `donor_smiles`, `acceptor_smiles`, `PCE`, `Jsc`, `FF`, `Voc`, `E_loss` (empty for two panels) |

### `data/generated_libraries/`

| File | Rows x columns | Columns |
|---|---|---|
| `donor_molecule_predictions.csv` | 52,995 x 21 | `smiles`; predicted donor descriptors `don_HOMO`, `don_LUMO`, `don_HL_gap`, `don_Diff_L_Lp1`, `don_IP`, `don_E_S`, `don_E_b`, `don_lambda_nm`, `don_osc_strength`, `don_dipole_debye`; predicted literature-set properties `guz_HOMO_D`, `guz_LUMO_D`, `guz_Eg_D`, `guz_Ehl_D`, `guz_donor_Mn_kDa`, `guz_donor_Mw_kDa`, `guz_donor_PDI`; `sascore`, `sascore_reward`, `tango_inspired_screen` |
| `acceptor_molecule_predictions.csv` | 53,719 x 19 | `smiles`; `acc_HOMO`, `acc_LUMO`, `acc_HL_gap`, `acc_Diff_L_Lp1`, `acc_EA`, `acc_E_S`, `acc_E_b`, `acc_Q20`, `acc_lambda_nm`, `acc_osc_strength`, `acc_dipole_debye`; `guz_HOMO_A`, `guz_LUMO_A`, `guz_Eg_A`, `guz_Ehl_A`; `sascore`, `sascore_reward`, `tango_inspired_screen` |
| `donor_synthesizability_scored.csv`, `acceptor_synthesizability_scored.csv` | 52,995 x 10 and 53,719 x 10 | `smiles`, `SAScore`, `RAScore`, `combined_synthesizability` (0.5 x normalised SAScore + 0.5 x RAScore, 0 to 1), `valid`, `mol_type`, `synthesizable_SA` (SAScore <= 6.0), `synthesizable_RA` (RAScore >= 0.5), `synthesizable_both`, `synth_rank` |

### `results/`

| File | Content |
|---|---|
| `results/summary/approach_statistics.csv` (13 x 4) | `approach`, `donor_count`, `acceptor_count`, `total`: molecules per generation route before deduplication. It has 13 labels because `Training_Set_Augmentation` is listed in addition to the 12 routes. |
| `results/summary/dmpnn_prediction_stats.csv` (9 x 5) | Mean, std, min, max, median, physical bounds, and counts at the lower and upper bound for the four surrogate outputs (`PCE`, `Jsc`, `FF`, `Voc`). The first column has no header. |
| `results/summary/output_metrics_generation.txt` | Pooled stage 1 metrics (pair count, library counts, surrogate means and maxima). |
| `results/postgen/generative_metrics.csv` (2 x 76) | One row per set (donor, acceptor): validity, uniqueness, novelty, internal diversity, scaffold diversity, nearest-neighbour similarity, FP-PCA Frechet distance, property-embedding Frechet distance, FCD, property-match, Wasserstein and Mahalanobis distances, SAScore summary. Source of most of SI Table S1. |
| `results/postgen/model_selection_horse_race.csv`, `results/postgen/property_model_cv_report.csv` (41 x 21 and 41 x 17) | Cross-validated comparison of classical models and D-MPNN for each descriptor and device target (`target`, `source`, `column`, `n_train`, `group_kind`, `winner`, `classical_*`, `dmpnn_*`, `notes`). |
| `results/postgen/retrosynthesis_audit.csv` (864 x 10) | Retrosynthesis audit of 864 molecules: `audit_backend`, `route_found`, `n_route_steps`, `route_score`, `rascore`, `sascore`, `sascore_reward`, `tango_inspired`, `audit_error`. |
| `results/postgen/stage_a_dataset_diagnostics.csv` (44 x 2) | Counts of raw, canonical and unparseable training molecules (first column has no header, then `value`). |
| `results/postgen/*.txt` | Donor and acceptor post-generation reports and `output_metrics_postgen.txt`. |
| `results/synthesizability/*.txt` | Donor and acceptor synthesizability reports and `output_metrics_synthesizability.txt` (means, pass rates, backends used). |
| `results/tier2_analysis/frechet_baseline_donor.csv`, `frechet_baseline_acceptor.csv` (200 x 1 each) | `split_half_fp_pca_fd`: FP-PCA Frechet distance of 200 split-half draws of the reference set. |
| `results/tier2_analysis/tier2_donor_scaffold_counts.csv`, `tier2_acceptor_scaffold_counts.csv` | `bemis_murcko_scaffold`, `n_donors` or `n_acceptors`, percentage of the Tier-2 donors or acceptors. |
| `results/tier2_analysis/tier2_substructure_enrichment.csv` (9 x 9) | Motif frequency among Tier-2 donors and acceptors versus the full libraries. |
| `results/tier2_analysis/tier2_top_recurring_donors.csv`, `tier2_top_recurring_acceptors.csv` (20 x 2) | The 20 most frequent donor or acceptor SMILES in the Tier-2 set and their pair counts. |
| `results/tier2_analysis/tier2_sweet_spot_candidates.csv` (12 x 10) | Selected sweet-spot pairs with surrogate values, `eloss_pred_physical` and TANGO scores. |
| `results/tier2_analysis/*.txt` | Chemical summary, recurrence and saturation summary, and Frechet baseline summary. |
| `reference_figures_png/*.png` | The 12 PNG outputs of `scripts/figures/`. |

## Limitations / not reproducible from provided files

Statements about the scientific content of the manuscript are limited to what the files in this repository show.

**Scope of the model.** The Ridge + isotonic + Gaussian-process surrogate is a screening tool. The manuscript reports
negative held-out R2 values for FF under all three structure-disjoint schemes and for Voc under donor-disjoint and
scaffold-disjoint evaluation (`data/figure_inputs/method1_disjoint_cv_report.csv`). Predicted values are for
candidate prioritisation, not for quantitative device prediction.

**Numbers that cannot be traced to a file in this repository**

1. **Fig. 1** (`Scheme.pdf`): no source or generating file.
2. **Sensitivity analysis** (2,832 to 133,943 pairs; 1,584 nominal Tier-1 pairs): no script or output file stores these
   values. They can only be recomputed from the 7.4 GB stage 1 pair table, which is not distributed.
3. **Fig. 4 inputs** (`donor_space4d_data.csv`, `acceptor_space4d_data.csv`; IntDiv_i of 0.485 and 0.491): the CSV files are
   included, the script that computed them is not.
4. **Fig. 6 input**: the 500,000-row subsample is included, the script that drew it is not. The full table has 11,988,459
   rows.
5. **Reference-set statement "46 entries with E_loss < 0.25 eV"**: `eloss_vs_pce_training_data.csv` flags 81 entries with
   `E_loss` below 0.25 eV (46 of them negative). The text and the data disagree.
6. **Acceptor reference counts**: `results/postgen/generative_metrics.csv` and SI Table S1 give 481 reference acceptors,
   while `src/analysis/frechet_baseline_analysis.py` and `results/tier2_analysis/frechet_baseline_summary.txt` give 483.
7. **"299 valid coordinate instances" for the donor GAN**: this count appears only in a job log that is not
   included. (`DMPNN.csv` has 300 rows and 299 distinct acceptor SMILES strings, consistent with the manuscript's
   "299 acceptor entries"; `results/postgen/stage_a_dataset_diagnostics.csv` reports 299 unique raw and 289 unique
   canonical acceptors.)
8. **Number of generation routes**: the manuscript says 12; `results/summary/approach_statistics.csv` lists 13 labels
   (the extra one is `Training_Set_Augmentation`).
9. **Sample sizes behind "complete library" wording**: nearest-neighbour similarity statistics use 20,000 molecules, MOSES
   IntDiv uses 5,000 molecules, and FCD uses 10,000 generated molecules (`results/postgen/generative_metrics.csv`).
10. **RAScore backend**: `results/synthesizability/output_metrics_synthesizability.txt` reports `RAScore backend: heuristic`.
    `synthesizability_scoring.py` falls back to a simplified heuristic when the `rascore` package is missing, so the
    RAScore values and the 85.0% / 99.3% pass rates in that run come from the fallback. The manuscript describes the
    published RAscore classifier. The 864-molecule retrosynthesis audit reports backend `rascore`; whether the real model
    was loaded for it is not documented.
11. **PFD** (property-embedding Frechet distance): the embedding model was not retained, so the value cannot be recomputed.
    The manuscript states this.
12. **Source-database counts** (1318 devices, 667 acceptors, 279 donors) and other literature statements are external.
13. **Everything that needs cluster-only assets**: the trained models, the `RAScore` conda environment, and the 240-rank
    run itself.

**Figures: PDF versus PNG.** `manuscript/main.tex` and `si.tex` include vector PDFs. The scripts write PNG files. The PDFs
use Computer Modern text and, for `eloss_vs_pce_corrected`, a different layout (wide axes, legend below the plot, axis
label `E_loss [eV]`). The PNGs use STIX mathtext and DejaVu Serif. The plotted data and annotations agree where they
were compared, but the regenerated PNGs are not typographically identical to the PDFs.

**Manuscript sources (checked by inspection, not compiled).** `main.tex` uses `\Eloss` 13 times and defines it only in
`si.tex`. The reference `\ref{sec:tier2composition}` has no matching `\label`. Nine `\adref` placeholders and one `{\com ...}`
span remain visible in the text. The Fig. 6 dashed FF threshold is drawn at 0.8, while the Tier-1 criterion in the text is
FF >= 0.75. These are left unchanged in this repository.

**Other.**
* `postgen_pipeline.py` is the run-time copy (`_MAX_CHEMNET_SMILES_LEN = 350`); the newer working copy differs in that
  one constant (349).
* `data/input/Guzik_data.csv` is not cited in `main.tex` or `references.bib`; its provenance and licence are unconfirmed.
* The `hpc/` drivers contain site-specific settings (partition name, job ID, environment name).

## Data availability, licence and citation

**Data availability.** The manuscript states that the data "are available from the corresponding authors upon
reasonable request". This repository provides the processed data behind the figures and the generated libraries. The
multi-GB intermediate files are not included. See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

**Licence.** The code is released under the MIT licence ([`LICENSE`](LICENSE)). The manuscript states no licence, so MIT
is a permissive default chosen when assembling this repository, and the authors have not confirmed it. The licence of
`Guzik_data.csv` and the redistribution terms of `DMPNN.csv` have not been confirmed, and the manuscript text and
figures follow the publisher's terms.

**Citation.** Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). The manuscript has no DOI yet.

```
B. Das and A. Mondal, "Synthesizability-Aware Hybrid Generative Workflow for Donor-Acceptor Discovery in
Organic Photovoltaics", manuscript in preparation. Code and data: opv-synthesizability-aware-generative-workflow.
```

**Contact.** The corresponding authors' addresses are printed on the title page of `manuscript/main.tex`.
