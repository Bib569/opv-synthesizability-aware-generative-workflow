# Synthesizability-Aware Hybrid Generative Workflow for Donor-Acceptor Discovery in Organic Photovoltaics

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A synthesizability-aware hybrid molecular-generation and screening workflow that proposes new organic-photovoltaic (OPV) donor and acceptor molecules, filters them for validity, novelty and synthetic practicality, and ranks donor-acceptor pairs with a screening surrogate for experimental or higher-fidelity computational follow-up.

## Overview

This repository (`opv-synthesizability-aware-generative-workflow`) holds the code of the workflow, the processed data behind every data figure, the generated donor and acceptor libraries, and the scripts that redraw the figures. The workflow generates class-specific donor and acceptor libraries, scores them for synthetic practicality, pairs them, and shortlists pairs by predicted performance and building-block compatibility. The device-property model is a screening tool for ranking, not a predictor of experimentally realized performance.

- **Chemically controlled generation**: 12 complementary proposal and augmentation routes, followed by common structural validation, canonicalization, deduplication and an adaptive nearest-neighbor novelty filter, give 52,995 unique donors and 53,719 unique acceptors with no exact matches to the reference structures
- **Synthetic practicality as a prioritization attribute**: SAScore, RAScore and a TANGO-inspired building-block compatibility heuristic rank molecules without claiming a confirmed synthesis route
- **Pair-level screening with stated limits**: 11,988,459 donor-acceptor pairs are ranked by a Ridge + isotonic + Gaussian-process surrogate trained on 300 curated device records, and structure-disjoint validation defines where its predictions can and cannot be trusted

## Paper

This repository contains code and data for the paper:

**Synthesizability-Aware Hybrid Generative Workflow for Donor-Acceptor Discovery in Organic Photovoltaics**

### Authors

Bibhas Das<sup>1,*</sup> and Anirban Mondal<sup>1,*</sup>

<sup>1</sup>Department of Chemistry, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India

*Corresponding authors: Bibhas Das (dasbibhas@iitgn.ac.in) and Anirban Mondal (amondal@iitgn.ac.in)

## Table of Contents

- [Abstract](#abstract)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Figure-to-Script-to-Data Mapping](#figure-to-script-to-data-mapping)
- [Model Summary](#model-summary)
- [Results](#results)
- [Datasets](#datasets)
- [Customization](#customization)
- [Limitations and Reproducibility Notes](#limitations-and-reproducibility-notes)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Abstract

Molecular discovery in organic photovoltaics (OPVs) requires simultaneous optimization of donor and acceptor structures, molecular novelty, synthetic practicality, and device-level performance. We present a synthesizability-aware hybrid molecular-generation and screening workflow for OPV donor-acceptor discovery. Twelve complementary proposal and augmentation routes produced class-specific molecular libraries that, after structural validation, canonicalization, deduplication, and adaptive novelty filtering, yielded 52,995 unique donor candidates and 53,719 unique acceptor candidates with no exact matches to the reference structures. The libraries exhibited substantial scaffold and fingerprint diversity while retaining a measurable relationship to the OPV reference domain. Synthetic practicality was assessed using SAScore, RAScore, and a TANGO-inspired building-block compatibility heuristic; 85.0% of donors and 99.3% of acceptors satisfied the operational accessibility screen. Pairing the filtered libraries produced 11,988,459 donor-acceptor combinations, ranked using a ridge-regression surrogate with isotonic recalibration and Gaussian-process residual correction trained on 300 curated device records. A total of 2,832 Tier-2 candidate pairs satisfied joint predicted performance and compatibility criteria, including a small subset with predicted high efficiency and low energy loss. Structure-disjoint validation revealed target-dependent transferability, with particularly limited quantitative reliability for fill factor and open-circuit voltage. The reported device properties are therefore interpreted as screening-based prioritization signals rather than experimentally realized performance. The workflow provides a chemically interpretable route for expanding OPV chemical space and prioritizing donor-acceptor pairs for experimental or higher-fidelity computational follow-up.

(Abstract of the associated paper, with LaTeX markup removed.)

## Workflow

The framework consists of an integrated five-stage pipeline. Stages 1 to 3 were run on a SLURM cluster, stages 4 and 5 run locally (stage 4 partly, see [Usage](#usage)). Entry points, working-directory conventions and the inputs that are not in the repository are described in [`docs/PIPELINE.md`](docs/PIPELINE.md).

### Stage 1: Generation, Pairing and Surrogate Screening

12 generation routes (plus training-set augmentation) produce donor and acceptor libraries, which are paired and ranked (`src/generation/GAN_240_final.py`, driver `hpc/sub_140.sh`, 240 SLURM ranks):

```
DMPNN.csv (300 device records) + Guzik_data.csv        data/input/
        |
        v
12 generation routes -> validation -> canonicalization -> deduplication
        -> adaptive novelty filter -> donor library | acceptor library
        |
        v
pairing (donors x acceptors) -> Ridge + isotonic + GP surrogate
        -> Tier-1 / Tier-2 selection -> structure-disjoint cross-validation
        |
        v
merge of the 240 rank outputs: src/postprocessing/merge_rank_outputs_final.py --stage generation
```

### Stage 2: Post-Generation Metrics

Library metrics, property models, E<sub>loss</sub> gate and retrosynthesis audit (`src/postgen/postgen_pipeline.py`, driver `hpc/sub_postgen_full_140.sh`):

```
generated pair pool -> generative metrics (validity, uniqueness, novelty, diversity, FCD, ...)
                    -> property models and model-selection horse race
                    -> E_loss gate
                    -> retrosynthesis audit
merge: src/postprocessing/merge_rank_outputs_final.py --stage postgen
```

### Stage 3: Synthesizability Scoring

SAScore, RAScore and a combined score for every unique molecule (`src/synthesizability/synthesizability_scoring.py`, driver `hpc/sub_synth_scoring_240.sh`):

```
unique donors / acceptors -> SAScore (<= 6.0) + RAScore (>= 0.5)
                          -> combined synthesizability = 0.5 x normalized SAScore + 0.5 x RAScore
```

### Stage 4: Tier-2 and Baseline Analysis

```
high-performance pairs -> Tier-2 chemistry, scaffold tables, sweet spot   (tier2_chemical_analysis.py)
reference set          -> split-half baseline (Fig. S1)                   (frechet_baseline_analysis.py)
generated pool         -> E_loss vs PCE data (Fig. 9)                     (create_eloss_vs_pce_corrected.py)
Tier-2 sweet spot      -> representative pairs (Fig. 8)                   (make_representative_figure.py)
```

### Stage 5: Manuscript Figures

```
data/figure_inputs/*.csv -> scripts/figures/01..12_*.py (run_all.py) -> PNG figures
```

### Key Framework Features

1. **Separate Donor and Acceptor Treatment**: donors and acceptors are standardized, generated and quality-controlled as distinct classes and are combined only after molecular filtering
2. **Common Validity and Novelty Controls**: every candidate must give a sanitized, single-component, non-radical RDKit molecule, be unique, and pass an adaptive Morgan-Tanimoto nearest-neighbor novelty criterion (radius-2, 2048-bit fingerprints, cutoff that depends on heavy-atom count)
3. **Synthetic Practicality as Prioritization**: SAScore <= 6.0 and RAScore >= 0.5 define an operational screen; the TANGO-inspired score S = 0.5 FMS + 0.5 BB-Tanimoto measures building-block compatibility. None of these establishes a synthesis route
4. **Pair-Level Surrogate and Tiering**: concatenated donor and acceptor fingerprints plus descriptors feed a Ridge + isotonic + Gaussian-process residual surrogate; two nested tiers (see [Model Summary](#model-summary)) shortlist pairs
5. **Explicit Validation Limits**: donor-disjoint, acceptor-disjoint and Bemis-Murcko scaffold-disjoint splits show which targets transfer and which do not

## System Requirements

### Hardware Requirements
- **Figures and local analysis (stages 4 and 5, partly)**: an ordinary desktop or laptop is enough. No GPU is needed. The figure scripts redraw all 12 figures from the CSV files in `data/figure_inputs/` in a few minutes
- **Storage**: about 120 MB for the repository. The two multi-GB files described in [`data/large_inputs/README.md`](data/large_inputs/README.md) (about 7.4 GB and 5.2 GB) are not included
- **HPC cluster (stages 1 to 3, and the stage 4 scripts that need the large files)**: the SLURM drivers in `hpc/` request 5 nodes, 48 tasks per node (240 tasks, 1 CPU each) on a partition named `highmemory`. The time limits are 240 hours for `sub_140.sh` and `sub_postgen_full_140.sh` and 24 hours for `sub_synth_scoring_240.sh`. The drivers request no GPU. Memory per rank was not recorded. Partition names and limits belong to the original cluster; adapt them
- **Run time**: the comments in `src/generation/GAN_240_final.py` state that the original generation run took more than 22 hours end to end, with some ranks finishing in about 1 hour and others taking more than 20 hours

### Operating Systems
- Windows: the pinned versions in `requirements.txt` were taken from CPython 3.12.0 on Windows, where the figures were regenerated and checked
- Linux: required for the HPC stages (SLURM and `sh` drivers)
- macOS and Linux for the figure scripts: not tested

### Software Dependencies
- Python 3.12
- Figure scripts and local analysis (pinned in `requirements.txt` and `environment.yml`): numpy 2.4.6, pandas 2.2.3, matplotlib 3.10.1, scipy 1.15.2, scikit-learn 1.8.0, pillow 11.1.0, rdkit 2026.3.4. Scripts 08, 10 and 12 need RDKit (10 also scikit-learn and SciPy, 12 also Pillow) and skip themselves with a message if it is missing
- HPC stages (not pinned, because the versions of the cluster environment `RAScore` were not recorded): numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, RDKit, Pillow, joblib, optuna, mpi4py, and (optionally, with fallbacks in the code) tensorflow, chemprop, fcd, selfies, rascore, aizynthfinder, syba, scscore, plus RDKit's contributed `sascorer.py` and `fpscores.pkl.gz`. `hpc/setup_run_dir.sh` copies the RDKit files when it finds them

## Installation

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/Bib569/opv-synthesizability-aware-generative-workflow.git
cd opv-synthesizability-aware-generative-workflow

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda environment

```bash
# Clone the repository
git clone https://github.com/Bib569/opv-synthesizability-aware-generative-workflow.git
cd opv-synthesizability-aware-generative-workflow

# Create the environment (same pins as requirements.txt, installed through pip inside the environment)
conda env create -f environment.yml
conda activate opv-gan-workflow
```

`requirements.txt` and `environment.yml` cover `scripts/figures/` and the local scripts in `src/analysis/`. The HPC stages need more packages and are not pinned; see [System Requirements](#system-requirements).

## Directory Structure

```
opv-synthesizability-aware-generative-workflow/
|-- README.md                                  # Project documentation
|-- LICENSE                                    # MIT License (code only)
|-- CITATION.cff                               # Machine-readable citation metadata
|-- CHANGELOG.md                               # Every change made to the original code when this repository was assembled
|-- requirements.txt                           # Pinned Python dependencies for figures and local analysis
|-- environment.yml                            # Conda equivalent of requirements.txt
|-- .gitignore                                 # Git ignore patterns
|
|-- src/                                       # Pipeline code
|   |-- generation/
|   |   `-- GAN_240_final.py                   # Stage 1: 12-route generation, pairing, surrogate, tiering, disjoint CV
|   |-- postprocessing/
|   |   `-- merge_rank_outputs_final.py        # Merges the 240 per-rank outputs (--stage generation | postgen)
|   |-- postgen/
|   |   `-- postgen_pipeline.py                # Stage 2: metrics, property models, E_loss gate, retrosynthesis audit
|   |-- synthesizability/
|   |   `-- synthesizability_scoring.py        # Stage 3: SAScore / RAScore / combined score
|   |-- plotting/
|   |   |-- manuscript_plots.py                # Exploratory plots; producer of some figure-input CSVs
|   |   `-- manuscript_plots_revised.py        # Revised plots; producer of the Fig. 5 input CSVs
|   `-- analysis/                              # Stage 4
|       |-- create_eloss_vs_pce_corrected.py   # E_loss vs PCE data for Fig. 9 (needs multi-GB inputs)
|       |-- frechet_baseline_analysis.py       # Split-half baseline data for Fig. S1
|       |-- make_representative_figure.py      # Tier-2 representative-pair panels (Fig. 8)
|       `-- tier2_chemical_analysis.py         # Tier-2 chemistry, scaffold tables, sweet spot (needs multi-GB input)
|
|-- scripts/figures/                           # Stage 5: one script per figure
|   |-- README.md                              # Per-script inputs, outputs and extra dependencies
|   |-- _common_style.py                       # Shared style, colors and helpers (not a figure script)
|   |-- run_all.py                             # Runs all 12 figure scripts
|   |-- 01_gan_training_donor.py               # Fig. 2
|   |-- 02_gan_training_acceptor.py            # Fig. 3
|   |-- 03_combined_space_analysis.py          # Fig. 4 (+ LaTeX summary table)
|   |-- 04_device_properties.py                # Fig. 6
|   |-- 05_disjoint_cv_r2.py                   # Fig. 7
|   |-- 06_eloss_vs_pce.py                     # Fig. 9
|   |-- 07_sascore_rascore_tango_4d.py         # Fig. 5
|   |-- 08_tier2_scaffold_family_bars.py       # Fig. S2
|   |-- 09_tier2_building_block_recurrence.py  # Fig. S3
|   |-- 10_frechet_baseline.py                 # Fig. S1
|   |-- 11_tier2_eloss_pce_sweet_spot.py       # Fig. S4
|   `-- 12_tier2_representative_pair.py        # Fig. 8
|
|-- hpc/                                       # SLURM drivers
|   |-- setup_run_dir.sh                       # Builds the flat run/ directory that stages 1 to 3 expect
|   |-- sub_140.sh                             # Stage 1 driver (240 ranks, SKIP_GENERATION=0)
|   |-- sub_postgen_full_140.sh                # Stage 2 driver
|   `-- sub_synth_scoring_240.sh               # Stage 3 driver
|
|-- data/
|   |-- input/
|   |   |-- DMPNN.csv                          # 300 curated device records (training set)
|   |   `-- Guzik_data.csv                     # Literature dataset, 558 rows (licence unconfirmed)
|   |-- figure_inputs/                         # CSV files read by scripts/figures/
|   |   |-- DMPNN.csv                          # Copy of the training set (used by script 10)
|   |   |-- Guzik_data.csv                     # Copy of the literature dataset (used by script 10)
|   |   |-- acceptor_space4d_data.csv          # Fig. 4 acceptor sample (20,000 molecules)
|   |   |-- donor_space4d_data.csv             # Fig. 4 donor sample (20,000 molecules)
|   |   |-- device_properties_distribution_method1_ridge_gp_data.csv  # Fig. 6, 500,000-pair subsample
|   |   |-- discriminator_accuracy_acceptor_optimized.csv  # Fig. 3 discriminator accuracy
|   |   |-- discriminator_accuracy_donor_optimized.csv     # Fig. 2 discriminator accuracy
|   |   |-- gan_losses_acceptor_optimized.csv  # Fig. 3 GAN losses
|   |   |-- gan_losses_donor_optimized.csv     # Fig. 2 GAN losses
|   |   |-- eloss_vs_pce_generated_corrected_data.csv  # Fig. 9, 20,000-pair sample
|   |   |-- eloss_vs_pce_training_data.csv     # Fig. 9, 300 reference records
|   |   |-- high_performance_pairs.csv         # The 2,832 Tier-2 pairs (Figs. S2, S3)
|   |   |-- method1_disjoint_cv_report.csv     # Fig. 7 structure-disjoint held-out metrics
|   |   |-- sascore_rascore_tango_4d_acceptor_data.csv  # Fig. 5 acceptor sample
|   |   |-- sascore_rascore_tango_4d_donor_data.csv     # Fig. 5 donor sample
|   |   |-- tier2_eloss_pce_sweet_spot_data.csv         # Fig. S4
|   |   `-- tier2_representative_pair_data.csv # Fig. 8
|   |-- generated_libraries/
|   |   |-- donor_molecule_predictions.csv     # 52,995 donors with predicted properties
|   |   |-- acceptor_molecule_predictions.csv  # 53,719 acceptors with predicted properties
|   |   |-- donor_synthesizability_scored.csv  # Donors with SAScore / RAScore / combined score
|   |   `-- acceptor_synthesizability_scored.csv  # Acceptors with SAScore / RAScore / combined score
|   `-- large_inputs/
|       `-- README.md                          # Placeholder; lists the multi-GB files that are not in git
|
|-- results/                                   # Output tables and reports quoted in the manuscript
|   |-- summary/
|   |   |-- approach_statistics.csv            # Molecules per generation route
|   |   |-- dmpnn_prediction_stats.csv         # Summary statistics of the four surrogate outputs
|   |   `-- output_metrics_generation.txt      # Pooled stage 1 metrics
|   |-- postgen/
|   |   |-- acceptor_postgen_analysis_report.txt  # Acceptor post-generation report
|   |   |-- donor_postgen_analysis_report.txt     # Donor post-generation report
|   |   |-- generative_metrics.csv             # Library metrics (source of most of SI Table S1)
|   |   |-- model_selection_horse_race.csv     # Classical models vs D-MPNN
|   |   |-- output_metrics_postgen.txt         # Pooled stage 2 metrics
|   |   |-- property_model_cv_report.csv       # Cross-validated property-model report
|   |   |-- retrosynthesis_audit.csv           # Audit of 864 molecules
|   |   `-- stage_a_dataset_diagnostics.csv    # Training-set counts
|   |-- synthesizability/
|   |   |-- acceptor_synthesizability_analysis_report.txt  # Acceptor report
|   |   |-- donor_synthesizability_analysis_report.txt     # Donor report
|   |   `-- output_metrics_synthesizability.txt  # Means, pass rates, backends used
|   |-- tier2_analysis/
|   |   |-- frechet_baseline_acceptor.csv      # 200 split-half draws, acceptors
|   |   |-- frechet_baseline_donor.csv         # 200 split-half draws, donors
|   |   |-- frechet_baseline_summary.txt       # Baseline summary
|   |   |-- tier2_acceptor_scaffold_counts.csv # Bemis-Murcko scaffolds of Tier-2 acceptors
|   |   |-- tier2_donor_scaffold_counts.csv    # Bemis-Murcko scaffolds of Tier-2 donors
|   |   |-- tier2_chemical_summary.txt         # Tier-2 chemical summary
|   |   |-- tier2_recurrence_and_saturation_summary.txt  # Building-block recurrence and saturation
|   |   |-- tier2_substructure_enrichment.csv  # Motif frequency, Tier-2 vs full libraries
|   |   |-- tier2_sweet_spot_candidates.csv    # Selected sweet-spot pairs
|   |   |-- tier2_top_recurring_acceptors.csv  # 20 most frequent Tier-2 acceptors
|   |   `-- tier2_top_recurring_donors.csv     # 20 most frequent Tier-2 donors
|   `-- reference_figures_png/                 # PNG outputs of scripts/figures/ for comparison (12 files)
|       |-- combined_space_analysis.png
|       |-- device_properties_distribution_method1_ridge_gp.png
|       |-- eloss_vs_pce_corrected.png
|       |-- frechet_baseline_plot.png
|       |-- gan_training_acceptor_optimized.png
|       |-- gan_training_donor_optimized.png
|       |-- method1_disjoint_cv_r2.png
|       |-- sascore_rascore_tango_4d_combined.png
|       |-- tier2_building_block_recurrence.png
|       |-- tier2_eloss_pce_sweet_spot.png
|       |-- tier2_representative_pair.png
|       `-- tier2_scaffold_family_bars.png
|
`-- docs/
    |-- PIPELINE.md                            # Stages, entry points, inputs that are not in the repository
    `-- DATA_AVAILABILITY.md                   # What is and is not distributed, licensing points
```

`results/regenerated_figures/` (default output folder of `scripts/figures/`) is created on demand and is git-ignored.

## Usage

### Complete Pipeline Execution

Stages 5 and 4 (partly) run from a fresh clone. Stages 1 to 3 need a SLURM cluster.

```bash
# Step 1 (local): redraw all 12 figures from data/figure_inputs/
python scripts/figures/run_all.py

# Step 2 (local): Fig. S1 baseline data and Fig. 8 panels
python src/analysis/frechet_baseline_analysis.py
python src/analysis/make_representative_figure.py

# Step 3 (HPC, SLURM): stages 1 to 3
sh hpc/setup_run_dir.sh          # builds run/ with the flat layout the stage 1-3 scripts expect
cd run
export CONDA_ACTIVATE=/path/to/conda/bin/activate    # environment named RAScore
sbatch sub_140.sh                # stage 1 (+ stage 2 and plots)
sbatch sub_postgen_full_140.sh   # stage 2 alone, on an existing stage 1 pool
sbatch sub_synth_scoring_240.sh  # stage 3

# Step 4 (needs the multi-GB files in data/large_inputs/): Tier-2 chemistry and Fig. 9 data
python src/analysis/tier2_chemical_analysis.py
python src/analysis/create_eloss_vs_pce_corrected.py
```

`runs/22051` in the drivers is the SLURM job ID of the original stage 1 run; replace it with your own. Steps 3 and 4 have not been rerun as part of assembling this repository.

### Redrawing the Manuscript Figures

```bash
# All 12 figures
python scripts/figures/run_all.py

# One figure
python scripts/figures/03_combined_space_analysis.py

# Other input / output folders
python scripts/figures/run_all.py --data-dir data/figure_inputs --outdir figures_out
```

The working directory does not matter. Each script accepts `--data-dir` and `--outdir`, reads `data/figure_inputs/` by default and writes to `results/regenerated_figures/`. The PNG files in `results/reference_figures_png/` are the outputs that the same scripts and CSV files produced when this repository was assembled. Per-script inputs and extra dependencies are in [`scripts/figures/README.md`](scripts/figures/README.md).

### Stage-by-Stage Commands

```bash
# Stage 4, local: Fig. S1 baseline (reads data/input/DMPNN.csv and Guzik_data.csv, writes to results/tier2_analysis/)
python src/analysis/frechet_baseline_analysis.py

# Stage 4, local: Fig. 8 panels (reads data/figure_inputs/high_performance_pairs.csv and
# results/tier2_analysis/tier2_sweet_spot_candidates.csv, writes to results/tier2_analysis/)
python src/analysis/make_representative_figure.py

# Stage 4, needs data/large_inputs/pair_predictions.csv
python src/analysis/tier2_chemical_analysis.py

# Stage 4, needs two multi-GB files in data/large_inputs/
python src/analysis/create_eloss_vs_pce_corrected.py

# Merge the 240 per-rank outputs (HPC)
python src/postprocessing/merge_rank_outputs_final.py --indir <rank outputs> --outdir <merged> --stage generation
python src/postprocessing/merge_rank_outputs_final.py --indir <rank outputs> --outdir <merged> --stage postgen
```

### Output Files

| Command | Output |
|---|---|
| `python scripts/figures/run_all.py` | In `results/regenerated_figures/`: `gan_training_donor_optimized.png`, `gan_training_acceptor_optimized.png`, `combined_space_analysis.png`, `combined_space_analysis_table.tex`, `device_properties_distribution_method1_ridge_gp.png`, `method1_disjoint_cv_r2.png`, `eloss_vs_pce_corrected.png`, `sascore_rascore_tango_4d_combined.png`, `tier2_scaffold_family_bars.png`, `tier2_building_block_recurrence.png`, `frechet_baseline_plot.png`, `tier2_eloss_pce_sweet_spot.png`, `tier2_representative_pair.png` |
| `python src/analysis/frechet_baseline_analysis.py` | `results/tier2_analysis/frechet_baseline_donor.csv`, `frechet_baseline_acceptor.csv`, `frechet_baseline_summary.txt` |
| `python src/analysis/make_representative_figure.py` | `results/tier2_analysis/tier2_representative_pair.png` and `tier2_representative_pair_final.png` (these two files are not byte-identical to the Fig. 8 PNG written by `scripts/figures/12_tier2_representative_pair.py`; they were not compared further) |

`scripts/figures/03_combined_space_analysis.py` also writes `combined_space_analysis_table.tex`, a LaTeX table of the Fig. 4 summary statistics.

## Figure-to-Script-to-Data Mapping

All scripts are in `scripts/figures/`. Input CSV files are in `data/figure_inputs/`. Each script writes a PNG; the repository ships the PNG renderings produced by these scripts in `results/reference_figures_png/`. Figure numbers follow the associated paper.

| Item | PNG output (`results/reference_figures_png/`) | Script | Input data | Upstream producer of the data |
|---|---|---|---|---|
| Fig. 1 workflow scheme | none | none | none | not reproducible (no source) |
| Fig. 2 donor GAN training | `gan_training_donor_optimized.png` | `01_gan_training_donor.py` | `gan_losses_donor_optimized.csv`, `discriminator_accuracy_donor_optimized.csv` | `src/generation/GAN_240_final.py` and `src/plotting/manuscript_plots.py` |
| Fig. 3 acceptor GAN training | `gan_training_acceptor_optimized.png` | `02_gan_training_acceptor.py` | `gan_losses_acceptor_optimized.csv`, `discriminator_accuracy_acceptor_optimized.csv` | same |
| Fig. 4 chemical-space analysis | `combined_space_analysis.png` | `03_combined_space_analysis.py` | `donor_space4d_data.csv`, `acceptor_space4d_data.csv` | producer script not in the repository |
| Fig. 5 SAScore x RAScore x TANGO-inspired | `sascore_rascore_tango_4d_combined.png` | `07_sascore_rascore_tango_4d.py` | `sascore_rascore_tango_4d_donor_data.csv`, `sascore_rascore_tango_4d_acceptor_data.csv` | `src/plotting/manuscript_plots_revised.py` |
| Fig. 6 predicted PCE, Jsc, FF, Voc distributions | `device_properties_distribution_method1_ridge_gp.png` | `04_device_properties.py` | `device_properties_distribution_method1_ridge_gp_data.csv` (500,000-row subsample) | stage 1; subsampling script not in the repository |
| Fig. 7 structure-disjoint held-out R2 | `method1_disjoint_cv_r2.png` | `05_disjoint_cv_r2.py` | `method1_disjoint_cv_report.csv` | stage 1 (`GAN_240_final.py`) |
| Fig. 8 representative Tier-2 pairs | `tier2_representative_pair.png` | `12_tier2_representative_pair.py` | `tier2_representative_pair_data.csv` | `src/analysis/make_representative_figure.py` |
| Fig. 9 E_loss vs PCE | `eloss_vs_pce_corrected.png` | `06_eloss_vs_pce.py` | `eloss_vs_pce_training_data.csv`, `eloss_vs_pce_generated_corrected_data.csv` | `src/analysis/create_eloss_vs_pce_corrected.py` (needs the multi-GB inputs) |
| Fig. S1 split-half baseline | `frechet_baseline_plot.png` | `10_frechet_baseline.py` | `DMPNN.csv`, `Guzik_data.csv` (recomputes 200 split-half draws) | `src/analysis/frechet_baseline_analysis.py` |
| Fig. S2 Tier-2 scaffold families | `tier2_scaffold_family_bars.png` | `08_tier2_scaffold_family_bars.py` | `high_performance_pairs.csv` | stage 1 Tier-2 filter; `src/analysis/tier2_chemical_analysis.py` |
| Fig. S3 building-block recurrence | `tier2_building_block_recurrence.png` | `09_tier2_building_block_recurrence.py` | `high_performance_pairs.csv` | same |
| Fig. S4 E_loss vs PCE sweet spot | `tier2_eloss_pce_sweet_spot.png` | `11_tier2_eloss_pce_sweet_spot.py` | `tier2_eloss_pce_sweet_spot_data.csv` | `src/analysis/tier2_chemical_analysis.py` (needs the multi-GB inputs) |
| Table S1 consolidated metrics | typed into the Supporting Information of the paper | none | values in `results/postgen/generative_metrics.csv`, `results/synthesizability/output_metrics_synthesizability.txt`, `results/postgen/output_metrics_postgen.txt`, `results/summary/output_metrics_generation.txt`, `results/tier2_analysis/frechet_baseline_summary.txt` | stages 1 to 4 |

Where the manuscript quotes a number, `results/` holds the file that contains it, except for the numbers listed under [Limitations and Reproducibility Notes](#limitations-and-reproducibility-notes).

## Model Summary

### Screening Surrogate

The only reported device-property model is the pair-level screening surrogate:

- **Representation**: radius-2, 2048-bit Morgan fingerprints; for each pair the donor and acceptor fingerprint vectors are concatenated, together with the available molecular descriptors
- **Targets**: separate regressors for J<sub>SC</sub>, V<sub>OC</sub>, FF and PCE<sub>max</sub>, with candidate model families compared by held-out performance for each response
- **Final model**: ridge regression (`Ridge(alpha=1.0)` in `src/generation/GAN_240_final.py`) + isotonic recalibration + Gaussian-process residual correction. Predictions are clipped to physical bounds (PCE 0 to 21 %, Jsc 0 to 30 mA cm<sup>-2</sup>, FF 0.25 to 0.85, Voc 0.3 to 2.0 V)
- **Purpose**: a consistent ranking of donor-acceptor combinations. The predictions are screening estimates and not a substitute for device measurements

### Tiered Shortlisting

Two nested criteria are applied to the pair predictions together with the pair-compatibility score T<sub>pair</sub> (`ScreeningSurrogatePipeline.filter_high_performance_pairs` in `src/generation/GAN_240_final.py`):

| Tier | PCE | J<sub>SC</sub> (mA cm<sup>-2</sup>) | FF | V<sub>OC</sub> (V) | T<sub>pair</sub> |
|---|---|---|---|---|---|
| Tier 1 | >= 18 % | >= 25 | >= 0.75 | >= 1.0 | >= 0.5 |
| Tier 2 | >= 14 % | >= 18 | >= 0.60 | >= 0.85 | >= 0.35 |

### Validation

Donor-disjoint, acceptor-disjoint and Bemis-Murcko scaffold-disjoint splits are used for the diagnostic evaluation of the molecular and device-property models (`data/figure_inputs/method1_disjoint_cv_report.csv`, Fig. 7). Donor-group-disjoint cross-validation is used for the paired-screening workflow. These procedures test transferability across withheld structures. They are not experimental device validation.

### Generation and Synthesizability Components

- **Generation**: donor and acceptor libraries come from 12 proposal and augmentation routes. The coordinate-based adversarial component is implemented with TensorFlow/Keras and trained separately for donors and acceptors for five epochs, with Optuna used for hyperparameter searches (details in the Supporting Information of the paper)
- **Synthetic practicality**: SAScore <= 6.0 and RAScore >= 0.5 define the operational screen (`--threshold-sa` and `--threshold-ra` in `synthesizability_scoring.py`). The combined synthesizability is 0.5 x normalized SAScore + 0.5 x RAScore. The TANGO-inspired score is S = alpha FMS + (1 - alpha) BB-Tanimoto with alpha = 0.5. It is a domain-specific compatibility heuristic and not a reproduction of the published TANGO method

## Results

All numbers below are taken from the associated paper.

### Generated Libraries

| Quantity | Donors | Acceptors |
|---|---|---|
| Unique molecules after filtering | 52,995 | 53,719 |
| Exact matches to the reference set | none | none |
| Mean nearest-neighbor Tanimoto similarity to the reference set | 0.171 | 0.199 |
| Fraction with nearest-neighbor similarity < 0.5 | 97.0 % | 99.9 % |
| Mean TANGO-inspired score | 0.463 | 0.502 |
| Pass the SAScore <= 6.0 and RAScore >= 0.5 screen | 85.0 % (45,066 / 52,995) | 99.3 % (53,351 / 53,719) |

The pass rates come from a run whose RAScore backend was the built-in heuristic fallback; see [Limitations and Reproducibility Notes](#limitations-and-reproducibility-notes), item 10.

### Pair-Level Predictions

Across 11,988,459 generated donor-acceptor pairs:

| Property | Mean | Maximum |
|---|---|---|
| PCE | 5.87 % | 18.22 % |
| J<sub>SC</sub> | 11.28 mA cm<sup>-2</sup> | 27.70 mA cm<sup>-2</sup> |
| FF | 0.534 | 0.785 |
| V<sub>OC</sub> | 0.854 V | 1.295 V |

### Tier-2 Shortlist

- No pair satisfied all Tier-1 criteria; **2,832** pairs met the Tier-2 criteria
- Mean predicted Tier-2 properties: PCE = 15.85 % (range 14.0 to 18.2 %), J<sub>SC</sub> = 23.5 mA cm<sup>-2</sup>, FF = 0.737, V<sub>OC</sub> = 0.864 V, T<sub>pair</sub> = 0.454
- The highest-ranked sweet-spot pair has predicted PCE = 18.22 % and E<sub>loss</sub> = 0.54 eV; a chemically distinct pair reaches the same predicted PCE at E<sub>loss</sub> = 0.53 eV with a lower predicted FF (0.61 versus 0.79). The manuscript presents them as prioritized starting points, not as validated device compositions
- Sensitivity analysis: removing the FF and V<sub>OC</sub> thresholds raises the Tier-2 set from 2,832 to 133,943 pairs and gives 1,584 nominal Tier-1 pairs (not traceable to a file here, see item 2 under [Limitations and Reproducibility Notes](#limitations-and-reproducibility-notes))

### Structure-Disjoint Validation (held-out R<sup>2</sup>)

| Target | Held-out R<sup>2</sup> |
|---|---|
| Acceptor HOMO, LUMO, E<sub>g</sub> | 0.44 to 0.69 |
| Donor HOMO, LUMO, E<sub>g</sub> | -0.33 to -0.25 |
| PCE | 0.11 to 0.41 |
| J<sub>SC</sub> | weak; slightly negative under scaffold-disjoint testing |
| FF | negative in all three schemes (donor-disjoint -0.85, acceptor-disjoint -0.57, scaffold-disjoint -1.04) |
| V<sub>OC</sub> | donor-disjoint -0.73, scaffold-disjoint -0.78, acceptor-disjoint 0.38 |

Transferability is therefore target dependent, and the FF and V<sub>OC</sub> predictions are not reliable quantitative estimates. The reported device properties are used for ranking and triage only.

## Datasets

Paths are relative to the repository root. Column descriptions follow the column names and the code that writes them; a description is given only where it could be confirmed from the code or the manuscript. "Surrogate" means the Ridge + isotonic + Gaussian-process residual model. Units: PCE in percent, Jsc in mA cm^-2, FF as a fraction, Voc and energies in V and eV.

**Data availability.** The manuscript states that the data "are available from the corresponding authors upon reasonable request". This repository provides the processed data behind the figures and the generated libraries. The multi-GB intermediate files are not included; see [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) and [`data/large_inputs/README.md`](data/large_inputs/README.md).

### `data/input/` (and identical copies in `data/figure_inputs/`)

| File | Rows x columns | Content |
|---|---|---|
| `DMPNN.csv` | 300 x 61 | Training set of 300 curated device records. Columns: `Ref`; acceptor descriptors (`Acceptor_Compounds`, `HOMO (eV)`, `HOMO-1 (eV)`, `LUMO (eV)`, `LUMO+1 (eV)`, `H-L Gap (eV)`, dipole moment, wavelength, `E_S`, `Osc. strength`, `E_b`, `Q20`, `A_EA` and related differences and inverses); the same descriptors for the donor with prefix `D_`; `ΔHOMO`, `ΔLUMO`, `ΔG_CR`, `ΔG_CS`, `ΔG_CT`; device values `Jsc(mA/cm2)`, `Voc(V)`, `FF`, `PCE_max(%)`; `V_OC_sim`, `E_loss`; `acceptor(SMILES)`, `donor(SMILES)`. Some header characters (for example the dipole and wavelength symbols) are stored in a legacy encoding and are kept byte-identical because the scripts look the columns up by name. |
| `Guzik_data.csv` | 558 x 45 | Literature device and material dataset: reference and DOI, donor and acceptor names and frontier-orbital energies, processing and device-stack parameters, mobilities, `Voc (V)`, `Jsc (mA cm^-2)`, `FF (%)`, `PCE (%)`, `calculated PCE (%)`, and donor and acceptor SMILES and SELFIES. Source and licence not confirmed (see [License](#license)). |

### `data/figure_inputs/` (the rest)

| File | Rows x columns | Columns |
|---|---|---|
| `gan_losses_donor_optimized.csv`, `gan_losses_acceptor_optimized.csv` | 5 x 3 | `epoch`, `discriminator_loss`, `generator_loss` |
| `discriminator_accuracy_donor_optimized.csv`, `discriminator_accuracy_acceptor_optimized.csv` | 5 x 2 | `epoch`, `discriminator_accuracy_percent` |
| `donor_space4d_data.csv`, `acceptor_space4d_data.csv` | 20,000 x 4 each | Per-molecule values for a 20,000-molecule sample: `similarities` (similarity to the training set, x axis of Fig. 4), `prop_scores` (property-match score, y axis), `intdiv` (internal diversity IntDiv, 1 minus the maximum nearest-neighbor Tanimoto similarity within the library, z axis), `combined_synthesizability` (color) |
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
| `donor_synthesizability_scored.csv`, `acceptor_synthesizability_scored.csv` | 52,995 x 10 and 53,719 x 10 | `smiles`, `SAScore`, `RAScore`, `combined_synthesizability` (0.5 x normalized SAScore + 0.5 x RAScore, 0 to 1), `valid`, `mol_type`, `synthesizable_SA` (SAScore <= 6.0), `synthesizable_RA` (RAScore >= 0.5), `synthesizable_both`, `synth_rank` |

### `results/`

| File | Content |
|---|---|
| `results/summary/approach_statistics.csv` (13 x 4) | `approach`, `donor_count`, `acceptor_count`, `total`: molecules per generation route before deduplication. It has 13 labels because `Training_Set_Augmentation` is listed in addition to the 12 routes. |
| `results/summary/dmpnn_prediction_stats.csv` (9 x 5) | Mean, std, min, max, median, physical bounds, and counts at the lower and upper bound for the four surrogate outputs (`PCE`, `Jsc`, `FF`, `Voc`). The first column has no header. |
| `results/summary/output_metrics_generation.txt` | Pooled stage 1 metrics (pair count, library counts, surrogate means and maxima). |
| `results/postgen/generative_metrics.csv` (2 x 76) | One row per set (donor, acceptor): validity, uniqueness, novelty, internal diversity, scaffold diversity, nearest-neighbor similarity, FP-PCA Frechet distance, property-embedding Frechet distance, FCD, property-match, Wasserstein and Mahalanobis distances, SAScore summary. Source of most of SI Table S1. |
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
| `results/reference_figures_png/*.png` | The 12 PNG outputs of `scripts/figures/`. |

### Files Not in the Repository

| File | Size | Produced by | Used by |
|---|---|---|---|
| `generated_molecules_with_predictions.csv` | about 7.4 GB (11,988,459 pairs, `surrogate_*`, `pair_tango`) | stage 1 (`hpc/sub_140.sh`) | `create_eloss_vs_pce_corrected.py`, `postgen_pipeline.py`, `synthesizability_scoring.py`; also source of the Tier counts and of the 133,943 / 1,584 sensitivity numbers |
| `pair_predictions.csv` | about 5.2 GB (`eloss_pred_physical`, `dev_*`) | stage 2 (`hpc/sub_postgen_full_140.sh` + `merge_rank_outputs_final.py --stage postgen`) | `tier2_chemical_analysis.py`, `create_eloss_vs_pce_corrected.py` |
| full Method 1 surrogate table for Fig. 6 | about 850 MB | stage 1 | `data/figure_inputs/device_properties_distribution_method1_ridge_gp_data.csv` is a 500,000-row subsample of it |
| trained models, `models/fcd`, `models/aizynthfinder` | not retained | cluster run | stage 2 |

Place the first two under `data/large_inputs/` if you regenerate them or obtain them from the corresponding authors.

## Customization

Only options that exist in the code are listed here.

### Synthesizability Thresholds and Weights

`src/synthesizability/synthesizability_scoring.py` accepts `--threshold-sa` (default 6.0), `--threshold-ra` (default 0.5), `--w-sa` (default 0.5) and `--w-ra` (default 0.5). Input can be a single CSV (`--input` with `--mol-type donor|acceptor`), separate `--donor-input` and `--acceptor-input` files, or a pairs file (`--pairs-input`). The SMILES column is auto-detected or set with `--smiles-col`, `--donor-smiles-col` and `--acceptor-smiles-col`. Use `--distributed` for MPI ranks and `--outdir` for the output folder.

### Tier Thresholds

The Tier-1 and Tier-2 thresholds are class constants of `ScreeningSurrogatePipeline` in `src/generation/GAN_240_final.py` (`PCE_THRESHOLD`, `JSC_THRESHOLD`, `FF_THRESHOLD_TIER1`, `VOC_THRESHOLD`, `SYNTH_THRESHOLD_TIER1`, `SYNTH_THRESHOLD_TIER2`) and are changed by editing the file. A stage 1 scoring change does not need regeneration: `--skip-generation` and `--use-existing-predictions <generated_molecules_with_predictions.csv>` re-score and re-filter an existing predictions CSV.

### Fingerprints

The TANGO-inspired reward class (`TANGOSynthesizabilityReward` in `GAN_240_final.py`) takes `alpha` (0.5), `radius` (2) and `n_bits` (2048) as constructor arguments. Library-level similarity, novelty and diversity use radius-2, 2048-bit Morgan fingerprints as described in the manuscript.

### Post-Generation Options

`src/postgen/postgen_pipeline.py` accepts, among others, `--pairs-csv`, `--dmpnn-csv`, `--guzik-csv`, `--models-dir`, `--max-pairs` (cap on pair rows, useful for smoke tests), `--optuna-trials` (default 12), `--cv-folds` (default 5), `--audit-top-n` (default 750), `--eloss-policy exclude|refit|clip` (default `exclude`), `--seed` (default 42), `--train-only`, `--score-only` and `--skip-dmpnn-candidate`. Run `python src/postgen/postgen_pipeline.py --help` for the full list.

### Rerunning With Other Inputs

- **Figures**: put your own CSV files with the same names and columns in a folder and pass `--data-dir <folder>` (and `--outdir <folder>`) to `scripts/figures/run_all.py` or to a single script. The column names are listed under [Datasets](#datasets)
- **Fig. S1 baseline**: `src/analysis/frechet_baseline_analysis.py` reads `data/input/DMPNN.csv` and `data/input/Guzik_data.csv`; the paths are the constants `DMPNN_CSV`, `GUZIK_CSV` and `OUT_DIR` in the script
- **Stages 1 to 3**: the scripts read `DMPNN.csv` and `Guzik_data.csv` from the working directory built by `hpc/setup_run_dir.sh`, so replacing the files in `data/input/` before running the setup script feeds other training data into the pipeline. The surrogate and the pairing logic were developed for the 300-record device table, and the results are not expected to carry over to a different table without a new validation
- **Cluster settings**: partition, job ID (`runs/22051`) and conda environment name (`RAScore`) in the `hpc/` drivers are specific to the original cluster

## Limitations and Reproducibility Notes

Statements about the scientific content of the manuscript are limited to what the files in this repository show.

### What Can and Cannot Be Reproduced From This Repository

| Part | Reproducible from a fresh clone? |
|---|---|
| Redrawing 12 of the 13 figures (Figs. 2 to 9 and S1 to S4) from the CSV files in `data/figure_inputs/` | Yes, on a laptop, in a few minutes (see [Usage](#usage)) |
| The Fig. S1 baseline (`src/analysis/frechet_baseline_analysis.py`) and the Fig. 8 panels (`src/analysis/make_representative_figure.py`) | Yes |
| Fig. 1 (workflow scheme) | No. It is a drawn schematic and its source is not in the repository |
| Stage 1 (generation, 240 ranks), stage 2 (post-generation metrics) and stage 3 (synthesizability scoring) | Not locally. They were run on a SLURM cluster; the scripts and drivers are included, but the multi-GB outputs and trained models are not |

### Scope of the Model

The Ridge + isotonic + Gaussian-process surrogate is a screening tool. The manuscript reports negative held-out R<sup>2</sup> values for FF under all three structure-disjoint schemes and for V<sub>OC</sub> under donor-disjoint and scaffold-disjoint evaluation (`data/figure_inputs/method1_disjoint_cv_report.csv`). Predicted values are for candidate prioritization, not for quantitative device prediction.

### Numbers That Cannot Be Traced to a File in This Repository

1. **Fig. 1** (workflow scheme): no source or generating file.
2. **Sensitivity analysis** (2,832 to 133,943 pairs; 1,584 nominal Tier-1 pairs): no script or output file stores these values. They can only be recomputed from the 7.4 GB stage 1 pair table, which is not distributed.
3. **Fig. 4 inputs** (`donor_space4d_data.csv`, `acceptor_space4d_data.csv`; IntDiv_i of 0.485 and 0.491): the CSV files are included, the script that computed them is not.
4. **Fig. 6 input**: the 500,000-row subsample is included, the script that drew it is not. The full table has 11,988,459 rows.
5. **Reference-set statement "46 entries with E_loss < 0.25 eV"**: `eloss_vs_pce_training_data.csv` flags 81 entries with `E_loss` below 0.25 eV (46 of them negative). The text and the data disagree.
6. **Acceptor reference counts**: `results/postgen/generative_metrics.csv` and SI Table S1 give 481 reference acceptors, while `src/analysis/frechet_baseline_analysis.py` and `results/tier2_analysis/frechet_baseline_summary.txt` give 483.
7. **"299 valid coordinate instances" for the donor GAN**: this count appears only in a job log that is not included. (`DMPNN.csv` has 300 rows and 299 distinct acceptor SMILES strings, consistent with the manuscript's "299 acceptor entries"; `results/postgen/stage_a_dataset_diagnostics.csv` reports 299 unique raw and 289 unique canonical acceptors.)
8. **Number of generation routes**: the manuscript says 12; `results/summary/approach_statistics.csv` lists 13 labels (the extra one is `Training_Set_Augmentation`).
9. **Sample sizes behind "complete library" wording**: nearest-neighbor similarity statistics use 20,000 molecules, MOSES IntDiv uses 5,000 molecules, and FCD uses 10,000 generated molecules (`results/postgen/generative_metrics.csv`).
10. **RAScore backend**: `results/synthesizability/output_metrics_synthesizability.txt` reports `RAScore backend: heuristic`. `synthesizability_scoring.py` falls back to a simplified heuristic when the `rascore` package is missing, so the RAScore values and the 85.0% / 99.3% pass rates in that run come from the fallback. The manuscript describes the published RAscore classifier. The 864-molecule retrosynthesis audit reports backend `rascore`; whether the real model was loaded for it is not documented.
11. **PFD** (property-embedding Frechet distance): the embedding model was not retained, so the value cannot be recomputed. The manuscript states this.
12. **Source-database counts** (1318 devices, 667 acceptors, 279 donors) and other literature statements are external.
13. **Everything that needs cluster-only assets**: the trained models, the `RAScore` conda environment, and the 240-rank run itself.

### Oversized Data That Is Not Distributed

The following files are needed to recompute some results but are too large for git. They are listed in [`data/large_inputs/README.md`](data/large_inputs/README.md) and [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md); obtain them by re-running the HPC stages or from the corresponding authors.

- `generated_molecules_with_predictions.csv`, about 7.4 GB (all 11,988,459 pairs with surrogate predictions). Needed for the Tier counts, the sensitivity analysis and the E_loss vs PCE sample.
- `pair_predictions.csv`, about 5.2 GB (stage 2 pair table with `eloss_pred_physical`). Needed by `tier2_chemical_analysis.py` and `create_eloss_vs_pce_corrected.py`.
- The full Fig. 6 table, about 850 MB. The repository ships a 500,000-row subsample.
- Trained models (`models/postgen`, `models/fcd`, `models/aizynthfinder`) were not retained.
- The HPC conda environment `RAScore` was not recorded, so the HPC stages are not pinned. Stages 1 to 3 were not rerun in any check.

### Other

- `src/plotting/manuscript_plots_revised.py` has the relative default `--data-dir latest_results`, which does not exist in this repository; pass `--data-dir` explicitly.
- Every change made to the original code when this repository was assembled is recorded in [`CHANGELOG.md`](CHANGELOG.md).

## License

The code in this repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for the open points.

## Citation

If you use this code in your research, please cite (machine-readable metadata is in [`CITATION.cff`](CITATION.cff); the manuscript has no DOI yet):

```bibtex
@article{das2026synthesizability,
  author = {Das, Bibhas and Mondal, Anirban},
  title = {Synthesizability-Aware Hybrid Generative Workflow for Donor--Acceptor Discovery in Organic Photovoltaics},
  journal = {},
  year = {2026},
  volume = {},
  pages = {},
  doi = {},
  publisher = {}
}
```

## Contact

For questions, issues, or collaborations:

- **Anirban Mondal** (Principal Investigator): amondal@iitgn.ac.in
- **Bibhas Das** (Lead Developer): dasbibhas@iitgn.ac.in

**GitHub Issues**: [Open an issue](https://github.com/Bib569/opv-synthesizability-aware-generative-workflow/issues)

## Acknowledgments

We gratefully acknowledge:

- **Indian Institute of Technology Gandhinagar** for research facilities and financial support
- **PARAM Ananta** for computational resources
- The open-source communities behind RDKit, scikit-learn, TensorFlow/Keras, Optuna, matplotlib, pandas, NumPy and SciPy

---

<p align="center">
  <i>Advancing synthesizability-aware molecular design for next-generation organic photovoltaics</i>
</p>
