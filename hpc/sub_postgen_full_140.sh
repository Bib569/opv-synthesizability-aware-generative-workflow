#!/bin/sh
#SBATCH -N 5
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=240
#SBATCH --cpus-per-task=1
#SBATCH --time=240:00:00
#SBATCH --job-name=postgen_full_240
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=highmemory

cd "$SLURM_SUBMIT_DIR"

# Path to the "activate" script of your conda installation (needs an env named RAScore
# with RDKit, TensorFlow, scikit-learn, rascore ...). Override by exporting CONDA_ACTIVATE.
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}"
export CONDA_ACTIVATE
source "$CONDA_ACTIVATE"
conda activate RAScore

export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Keep the UTF-8 tick/cross marks in the plot script from crashing stdout.
export PYTHONIOENCODING=utf-8

# =============================================================================
# FULL POST-GENERATION ANALYSIS FROM SCRATCH
#
# Runs Stage A+B+C+D+E of postgen_pipeline.py (trains all property models fresh,
# then scores the pool), merges BOTH stages, and produces every figure and every
# interpretive text report.
#
# Molecule GENERATION is not re-run -- it reuses the existing pooled pair CSV.
#
# PREREQUISITES ON THE CLUSTER (verify before submitting):
#   1. DMPNN.csv MUST be the 61-column version containing V_OC_sim and E_loss.
#      The copy used in job 53180 had only 59 columns, which silently disabled
#      E_loss diagnostics, the dev_E_loss model and the eloss_vs_pce figure.
#   2. Guzik_data.csv MUST be present. It was entirely absent in job 53180, so
#      all 9 guz_* targets (donor PDI/Mn/Mw, Ehl_D/A, Eg_D/A, hole/electron
#      mobility) were skipped.
#   The preflight check below refuses to start if either is wrong, instead of
#   burning hours and producing a silently incomplete result.
# =============================================================================

EXISTING_PREDICTIONS="runs/22051/merged/results/generated_molecules_with_predictions.csv"

# ---------------------------------------------------------------------------
# PREFLIGHT: fail fast and loudly on the exact data problems seen last time.
# ---------------------------------------------------------------------------
echo "=========================== PREFLIGHT ==========================="
PREFLIGHT_OK=1

if [ ! -f "DMPNN.csv" ]; then
  echo "FATAL: DMPNN.csv not found in $SLURM_SUBMIT_DIR"
  PREFLIGHT_OK=0
else
  NCOL=$(head -1 DMPNN.csv | awk -F',' '{print NF}')
  echo "DMPNN.csv column count: $NCOL"
  if head -1 DMPNN.csv | grep -q "E_loss"; then
    echo "  E_loss column      : PRESENT"
  else
    echo "  E_loss column      : *** MISSING *** (stale 59-column file?)"
    echo "  -> E_loss analysis, the dev_E_loss model and eloss_vs_pce.png"
    echo "     cannot be produced. Upload the 61-column DMPNN.csv."
    PREFLIGHT_OK=0
  fi
  if head -1 DMPNN.csv | grep -q "V_OC_sim"; then
    echo "  V_OC_sim column    : PRESENT"
  else
    echo "  V_OC_sim column    : *** MISSING ***"
    PREFLIGHT_OK=0
  fi
fi

if [ -f "Guzik_data.csv" ]; then
  echo "Guzik_data.csv       : PRESENT ($(head -1 Guzik_data.csv | awk -F',' '{print NF}') columns)"
else
  echo "Guzik_data.csv       : *** MISSING ***"
  echo "  -> all guz_* targets (donor PDI/Mn/Mw, Ehl_D/A, Eg_D/A, mobilities)"
  echo "     will be skipped. Upload Guzik_data.csv."
  PREFLIGHT_OK=0
fi

if [ ! -f "$EXISTING_PREDICTIONS" ]; then
  echo "FATAL: pooled pair CSV not found at $EXISTING_PREDICTIONS"
  PREFLIGHT_OK=0
else
  echo "Pair pool            : PRESENT"
fi

if [ "$PREFLIGHT_OK" != "1" ]; then
  echo "================================================================="
  echo "PREFLIGHT FAILED -- aborting before wasting compute time."
  echo "Fix the items marked *** MISSING *** above and resubmit."
  echo "================================================================="
  exit 1
fi
echo "PREFLIGHT PASSED."
echo "================================================================="

POSTGEN_RUN_DIR="runs_postgen/${SLURM_JOB_ID}"
mkdir -p "$POSTGEN_RUN_DIR"

# Pre-create rank directories serially to avoid metadata storm on shared FS
for i in $(seq -f "%03g" 0 239); do
  mkdir -p "$POSTGEN_RUN_DIR/rank_${i}"
done

# ---------------------------------------------------------------------------
# STAGE A+B+C+D+E: full postgen pipeline. Rank 0 trains every property model
# (Stage B, several hours); all other ranks wait for its sentinel with NO
# TIMEOUT, then all 240 ranks score their shard.
#
# CRITICAL: remove any stale TRAINING_DONE sentinel from a prior run BEFORE
# launching. Without this, non-master ranks find the old sentinel immediately
# and load old models instead of waiting for the new training (this exact bug
# invalidated job 70355's results). The Python code also removes it, but
# doing it here too provides a race-condition-proof guarantee regardless of
# rank startup ordering.
# ---------------------------------------------------------------------------
rm -f models/postgen/TRAINING_DONE
echo "Cleared stale TRAINING_DONE sentinel (if any)."

srun --ntasks=240 bash -lc 'sleep $((SLURM_PROCID % 20)); python "postgen_pipeline.py" --distributed --outdir "'"$POSTGEN_RUN_DIR"'" --pairs-csv "'"$EXISTING_PREDICTIONS"'"'

# ---------------------------------------------------------------------------
# MERGE 1 (generation stage): produces plot_data/approach_full_stats_data.csv,
# i.e. the per-approach + pooled(all) + pooled(excluding Training_Set_
# Augmentation) FP-PCA-FD / IntDiv / TANGO table. Without this the
# per_approach_full_stats_*.png figures cannot be drawn -- that is exactly why
# they were missing from job 53180.
# --recompute-pooled-stats-only keeps this cheap: it re-derives the pooled
# statistics without re-merging every rank CSV or regenerating old plots.
# ---------------------------------------------------------------------------
python merge_rank_outputs_final.py --stage generation \
    --recompute-pooled-stats-only \
    --indir "runs/22051" --outdir "runs/22051/merged" \
    --training-csv DMPNN.csv || \
  echo "WARN: generation-stage pooled-stats recompute failed; per-approach figures may be skipped."

# ---------------------------------------------------------------------------
# MERGE 2 (postgen stage): property predictions, E_loss, SAScore, MOSES
# metrics, PFD/FCD distances, retrosynthesis audit, plus
# output_metrics_postgen.txt and the two interpretive
# <side>_postgen_analysis_report.txt files.
# ---------------------------------------------------------------------------
python merge_rank_outputs_final.py --stage postgen \
    --indir "$POSTGEN_RUN_DIR" --outdir "$POSTGEN_RUN_DIR/merged"

# ---------------------------------------------------------------------------
# CONSOLIDATE into <script_dir>/merged (the fixed relative path
# manuscript_plots.py reads). Order matters: generation-stage data first, then
# postgen-stage data overlays it.
#
# results/ is copied in explicitly because method1_disjoint_cv_report.csv is
# written there by GAN_240_final.py and belongs to NEITHER merge output -- its
# absence is why method1_disjoint_cv_r2.png was skipped in job 53180.
# ---------------------------------------------------------------------------
mkdir -p merged
cp -r runs/22051/merged/. merged/ 2>/dev/null
cp -r "$POSTGEN_RUN_DIR/merged/." merged/ 2>/dev/null

mkdir -p merged/results
if [ -f "results/method1_disjoint_cv_report.csv" ]; then
  cp results/method1_disjoint_cv_report.csv merged/results/
  echo "Copied method1_disjoint_cv_report.csv into merged/results/"
else
  echo "WARN: results/method1_disjoint_cv_report.csv not found."
  echo "      Run GAN_240_final.py --skip-generation --use-existing-predictions $EXISTING_PREDICTIONS"
  echo "      once to produce it; method1_disjoint_cv_r2.png will be skipped without it."
fi

# Remove stale Method-2 (Scharber physics) plot data so nothing downstream can
# accidentally resurrect the retracted figure from an old run's leftovers.
rm -f merged/plot_data/device_properties_distribution_method2_physics_data.csv
rm -f manuscript_plots_output/device_properties_distribution_method2_physics.png

# ---------------------------------------------------------------------------
# FIGURES: 18 manuscript-ready plots (Method-2 physics figure deliberately
# dropped). The script now prints an honest produced/skipped/errored summary.
# ---------------------------------------------------------------------------
python manuscript_plots.py

echo "================================================================="
echo "DONE. Key outputs:"
echo "  merged/output_metrics_postgen.txt"
echo "  merged/donor_postgen_analysis_report.txt"
echo "  merged/acceptor_postgen_analysis_report.txt"
echo "  merged/results_postgen/*.csv"
echo "  merged/plot_data/*.csv"
echo "  manuscript_plots_output/*.png"
echo "================================================================="
