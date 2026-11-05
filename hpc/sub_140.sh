#!/bin/sh
#SBATCH -N 5
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=240
#SBATCH --cpus-per-task=1
#SBATCH --time=240:00:00
#SBATCH --job-name=GAN_240_split
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

# ----------------------------------------------------------------------------
# Toggle: set to 1 to reuse an existing generated_molecules_with_predictions.csv
# and skip the 240-rank GAN training/generation stage entirely. Only rank 0
# does anything in that case (re-score/re-filter); postgen_pipeline.py still
# runs across all 240 procs afterwards.
# ----------------------------------------------------------------------------
SKIP_GENERATION=0
EXISTING_PREDICTIONS="runs/22051/merged/results/generated_molecules_with_predictions.csv"

RUN_DIR="runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Pre-create rank directories serially to avoid metadata storm on shared FS
for i in $(seq -f "%03g" 0 239); do
  mkdir -p "$RUN_DIR/rank_${i}"
done

# ----------------------------------------------------------------------------
# STAGE 1: Molecule generation (GAN_240_final.py) -- unchanged unless skipped
# ----------------------------------------------------------------------------
if [ "$SKIP_GENERATION" = "1" ]; then
  echo "SKIP_GENERATION=1 : re-scoring existing predictions on rank 0 only."
  python GAN_240_final.py --skip-generation --use-existing-predictions "$EXISTING_PREDICTIONS"
else
  srun --ntasks=240 bash -lc 'sleep $((SLURM_PROCID % 20)); python "GAN_240_final.py" --distributed --outdir "'"$RUN_DIR"'"'
fi

# Merge generation-stage rank outputs (label/pooling fixes, per-approach +
# pooled-excluding-augmentation stats) once all ranks complete.
if [ "$SKIP_GENERATION" != "1" ]; then
  python merge_rank_outputs_final.py --stage generation --indir "$RUN_DIR" --outdir "$RUN_DIR/merged"
fi

# ----------------------------------------------------------------------------
# STAGE 2: Post-generation metrics / prediction / synthesizability pipeline
#   Rank 0 trains models (Stage B); all 240 ranks score shards (Stage C/E).
# ----------------------------------------------------------------------------
POSTGEN_RUN_DIR="runs_postgen/${SLURM_JOB_ID}"
mkdir -p "$POSTGEN_RUN_DIR"
for i in $(seq -f "%03g" 0 239); do
  mkdir -p "$POSTGEN_RUN_DIR/rank_${i}"
done

srun --ntasks=240 bash -lc 'sleep $((SLURM_PROCID % 20)); python "postgen_pipeline.py" --distributed --outdir "'"$POSTGEN_RUN_DIR"'" --pairs-csv "'"$EXISTING_PREDICTIONS"'"'

# Merge postgen-stage rank outputs (property predictions, E_loss, SAScore,
# MOSES metrics, PFD/FCD distances, retrosynthesis audit).
python merge_rank_outputs_final.py --stage postgen --indir "$POSTGEN_RUN_DIR" --outdir "$POSTGEN_RUN_DIR/merged"

# ----------------------------------------------------------------------------
# STAGE 3: Manuscript-ready plots
#   manuscript_plots.py reads from <script_dir>/merged (a fixed relative
#   path). Consolidate both merge outputs there (generation stage first,
#   postgen stage second so its plot_data/results overlay without clobbering
#   unrelated generation-stage files) before invoking it.
# ----------------------------------------------------------------------------
mkdir -p merged
if [ "$SKIP_GENERATION" != "1" ]; then
  cp -r "$RUN_DIR/merged/." merged/
fi
cp -r "$POSTGEN_RUN_DIR/merged/." merged/

python manuscript_plots.py
