# data/large_inputs

This folder is a placeholder. The files below are too large for git (several GB) and are not part of the
repository. Scripts that need them look for them here.

| Expected file | Approximate size | Produced by | Needed by |
|---|---|---|---|
| `pair_predictions.csv` | 5.2 GB | stage 2: `hpc/sub_postgen_full_140.sh`, then `src/postprocessing/merge_rank_outputs_final.py --stage postgen` | `src/analysis/tier2_chemical_analysis.py`, `src/analysis/create_eloss_vs_pce_corrected.py` |
| `generated_molecules_with_predictions.csv` | 7.4 GB | stage 1: `hpc/sub_140.sh` | `src/analysis/create_eloss_vs_pce_corrected.py` (and stages 2 and 3 on the cluster) |

Obtain them by re-running the HPC stages (see `docs/PIPELINE.md`) or by asking the corresponding authors.
Everything in `scripts/figures/` runs without these files.
