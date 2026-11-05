#!/bin/sh
# -----------------------------------------------------------------------------
# setup_run_dir.sh - entry point for the HPC stages (1-3).
#
# The pipeline scripts (GAN_240_final.py, postgen_pipeline.py,
# merge_rank_outputs_final.py, synthesizability_scoring.py, manuscript_plots.py)
# were written to run from ONE flat working directory that also holds
# DMPNN.csv and Guzik_data.csv. This script builds that directory from the
# repository layout so nothing in the scripts has to be edited:
#
#     sh hpc/setup_run_dir.sh              # creates <repo>/run
#     sh hpc/setup_run_dir.sh /path/to/dir # or any directory of your choice
#     cd run
#     sbatch sub_140.sh                    # stage 1 (+ stage 2 + plots)
#     sbatch sub_postgen_full_140.sh       # stage 2 alone (needs the stage-1 pool)
#     sbatch sub_synth_scoring_240.sh      # stage 3 (synthesizability scoring)
#
# See docs/PIPELINE.md for the full description. The run directory is ignored
# by git (.gitignore).
# -----------------------------------------------------------------------------
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:-$REPO/run}"
mkdir -p "$RUN"

cp "$REPO/src/generation/GAN_240_final.py"                    "$RUN/"
cp "$REPO/src/postprocessing/merge_rank_outputs_final.py"     "$RUN/"
cp "$REPO/src/postgen/postgen_pipeline.py"                    "$RUN/"
cp "$REPO/src/synthesizability/synthesizability_scoring.py"   "$RUN/"
cp "$REPO/src/plotting/manuscript_plots.py"                   "$RUN/"
cp "$REPO/data/input/DMPNN.csv" "$REPO/data/input/Guzik_data.csv" "$RUN/"
cp "$REPO/hpc/sub_140.sh" "$REPO/hpc/sub_postgen_full_140.sh" "$REPO/hpc/sub_synth_scoring_240.sh" "$RUN/"

# SAScore helper files (sascorer.py, fpscores.pkl.gz) ship with RDKit's Contrib
# folder; the scoring scripts expect them in the working directory.
SA_DIR=$(python -c "import os; from rdkit.Chem import RDConfig; print(os.path.join(RDConfig.RDContribDir, 'SA_Score'))" 2>/dev/null || true)
if [ -n "$SA_DIR" ] && [ -f "$SA_DIR/sascorer.py" ] && [ -f "$SA_DIR/fpscores.pkl.gz" ]; then
  cp "$SA_DIR/sascorer.py" "$SA_DIR/fpscores.pkl.gz" "$RUN/"
else
  echo "WARNING: could not locate RDKit Contrib/SA_Score; copy sascorer.py and fpscores.pkl.gz into $RUN manually."
fi

echo "Run directory ready: $RUN"
echo "Next: cd \"$RUN\" && sbatch sub_140.sh"
