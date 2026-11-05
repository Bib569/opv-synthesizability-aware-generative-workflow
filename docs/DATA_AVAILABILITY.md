# Data availability

## Statement in the manuscript

The Data Availability section of the associated paper ends with:

> The data that support the findings of this study are available from the corresponding authors upon reasonable request.

That sentence is unchanged. This repository does not replace it. It makes the processed data behind the
figures and the reported tables available in a convenient form, and it documents which larger files exist
but are not distributed here.

## What is in the repository

| Folder | Content | Size |
|---|---|---|
| `data/input/` | `DMPNN.csv` (300 curated device records) and `Guzik_data.csv` (558 rows). Training and reference inputs of the pipeline. | 1.3 MB |
| `data/figure_inputs/` | The 17 CSV files that `scripts/figures/` reads, including a 500,000-row subsample of the 11,988,459 pair-level surrogate predictions (Fig. 6) and the 2,832 Tier-2 pairs. | 46 MB |
| `data/generated_libraries/` | The 52,995 donor and 53,719 acceptor molecules, with predicted properties and SAScore/RAScore/combined synthesizability. | 56 MB |
| `results/` (without the PNG folder) | Report and table files written by the pipeline and quoted in the manuscript. | 0.4 MB |
| `results/reference_figures_png/` | PNG renderings of the 12 data figures produced by the scripts in this repository. | 8.7 MB |

Column definitions for every CSV are in the "Data dictionary" section of `README.md`.

## What is not in the repository

| File | Approximate size | Why it is not included | How to obtain it |
|---|---|---|---|
| `generated_molecules_with_predictions.csv` (all 11,988,459 donor-acceptor pairs with Ridge + isotonic + GP predictions) | 7.4 GB | Exceeds git and GitHub file-size limits | Re-run stage 1 (`hpc/sub_140.sh`) or request from the corresponding authors |
| `pair_predictions.csv` (pair table with `eloss_pred_physical`) | 5.2 GB | Same | Re-run stage 2 (`hpc/sub_postgen_full_140.sh`) or request from the corresponding authors |
| Full Fig. 6 table (all pairs, four surrogate columns) | about 850 MB | Same | Derive from the first file, or request |
| Trained models (`models/postgen`, `models/fcd`, `models/aizynthfinder`) | not retained | Not kept after the cluster run | Not available; stage 2 would need to retrain or reinstall them |
| Per-job logs of the 240-rank runs | tens of MB | Job logs, not data | Request |
| Source database behind the 300 device records (1318 devices, 667 acceptors, 279 donors) | n/a | Third-party literature resource (Miyake and Saeki, ref. `miyake2021machine`) | See the cited work |

## Third-party and licensing points that need the authors' decision

* `data/input/Guzik_data.csv` is a literature-derived dataset that is named in the Supporting Information of the paper (Table S1, footnote a) but
  was not found in the paper's reference list. Its source and licence have not been confirmed. Confirm before the
  repository is made public, or remove the file and its copy in `data/figure_inputs/`.
* `data/input/DMPNN.csv` holds 300 device records with computed electronic descriptors; the manuscript cites the literature compilation of Miyake and Saeki (miyake2021machine) and the authors' earlier work (das2024predictive) in this context. Its
  redistribution terms have not been confirmed either.
* The repository code is released under the MIT licence (`LICENSE`). The licence text covers the code. For the data,
  the paper's text and figures, the terms above and the publisher's terms apply; the MIT licence was chosen as a
  permissive default because the manuscript states no licence, and it is an open point for the authors.
* The Data Availability sentence in the manuscript ("upon reasonable request") and a public repository that ships
  processed data are not worded the same way. The authors should decide whether to update the manuscript sentence to
  point to this repository.
