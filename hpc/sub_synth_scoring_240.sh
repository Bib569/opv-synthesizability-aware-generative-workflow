#!/bin/sh
#SBATCH -N 5
#SBATCH --ntasks-per-node=48
#SBATCH --ntasks=240
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=synth_score_240
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

# ── Thread control (one thread per task for maximum parallelism) ───────────────
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# =============================================================================
# SYNTHESIZABILITY SCORING PIPELINE (240 distributed ranks)
#
# Scores ALL unique donor and acceptor molecules from the pooled pair CSV
# using SAScore (via sascorer.py + fpscores.pkl.gz) and RAScore (rascore pkg).
#
# The work is split evenly across 240 ranks:
#   - Each rank scores ~N/240 molecules (both SAScore + RAScore per molecule)
#   - Rank 0 merges all shard outputs and produces:
#     * Scored CSVs (donor + acceptor)
#     * 14 manuscript-quality plots
#     * Plot data CSVs for manual re-plotting
#     * Analysis reports (donor + acceptor)
#     * Output metrics summary
#
# PREREQUISITES (must be in $SLURM_SUBMIT_DIR):
#   1. sascorer.py        — RDKit SAScore calculator
#   2. fpscores.pkl.gz    — Fragment frequency data for SAScore
#   3. synthesizability_scoring.py — this pipeline script
#   4. Input CSV (generated_molecules_with_predictions.csv or separate files)
#   5. rascore package installed in the RAScore conda env
# =============================================================================

# ── Input file ─────────────────────────────────────────────────────────────────
# Pooled pair table written by stage 1 (relative to the submit directory; 22051 was the
# SLURM job ID of the original run - change it to your own stage-1 job ID).
PAIRS_CSV="${PAIRS_CSV:-runs/22051/merged/results/generated_molecules_with_predictions.csv}"
OUTDIR="synthesizability_output_${SLURM_JOB_ID}"

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "=========================== PREFLIGHT ==========================="
PREFLIGHT_OK=1

if [ ! -f "sascorer.py" ]; then
    echo "FATAL: sascorer.py not found in $SLURM_SUBMIT_DIR"
    echo "  Copy sascorer.py from RDKit Contrib/SA_Score/ into this directory."
    PREFLIGHT_OK=0
else
    echo "sascorer.py           : PRESENT"
fi

if [ ! -f "fpscores.pkl.gz" ]; then
    echo "FATAL: fpscores.pkl.gz not found in $SLURM_SUBMIT_DIR"
    echo "  Copy fpscores.pkl.gz from RDKit Contrib/SA_Score/ into this directory."
    PREFLIGHT_OK=0
else
    echo "fpscores.pkl.gz       : PRESENT"
fi

if [ ! -f "synthesizability_scoring.py" ]; then
    echo "FATAL: synthesizability_scoring.py not found in $SLURM_SUBMIT_DIR"
    PREFLIGHT_OK=0
else
    echo "synthesizability_scoring.py : PRESENT"
fi

if [ ! -f "$PAIRS_CSV" ]; then
    echo "FATAL: Input CSV not found at $PAIRS_CSV"
    PREFLIGHT_OK=0
else
    NROWS=$(wc -l < "$PAIRS_CSV")
    echo "Input CSV             : PRESENT ($NROWS lines)"
fi

# Quick check that rascore is importable
python -c "from rascore import RAscore_NN; print('RAScore import: OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    python -c "from rascore import RAscore_XGB; print('RAScore XGB import: OK')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "WARNING: rascore package not importable. Will use heuristic fallback."
        echo "  Install with: pip install rascore"
    fi
fi

# Quick check sascorer works
python -c "
import sys; sys.path.insert(0, '.')
from sascorer import calculateScore
from rdkit import Chem
mol = Chem.MolFromSmiles('c1ccccc1')
s = calculateScore(mol)
print(f'SAScore import: OK (benzene={s:.3f})')
" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: sascorer.py failed self-test. Check fpscores.pkl.gz is present."
fi

if [ "$PREFLIGHT_OK" != "1" ]; then
    echo "================================================================="
    echo "PREFLIGHT FAILED -- fix the items above and resubmit."
    echo "================================================================="
    exit 1
fi
echo "PREFLIGHT PASSED."
echo "================================================================="

# ── Pre-create rank directories (avoid metadata storm from 240 ranks) ──────────
mkdir -p "$OUTDIR"
for i in $(seq -f "%03g" 0 239); do
    mkdir -p "$OUTDIR/rank_${i}"
done
mkdir -p "$OUTDIR/results"
mkdir -p "$OUTDIR/plots"
mkdir -p "$OUTDIR/plot_data"

echo "Created output directory: $OUTDIR"
echo "Starting distributed scoring with $SLURM_NTASKS ranks..."
echo ""

# ── Launch distributed scoring ─────────────────────────────────────────────────
# Stagger rank startup slightly to avoid I/O thundering herd on shared FS
srun --ntasks=240 bash -lc '
    source "$CONDA_ACTIVATE"
    conda activate RAScore
    sleep $((SLURM_PROCID % 10))
    python "synthesizability_scoring.py" \
        --distributed \
        --pairs-input "'"$PAIRS_CSV"'" \
        --outdir "'"$OUTDIR"'"
'

echo ""
echo "================================================================="
echo "SYNTHESIZABILITY SCORING PIPELINE COMPLETE"
echo "================================================================="
echo ""
echo "Key outputs in: $OUTDIR/"
echo "  synthesizability_scoring_progress.log   (full pipeline log)"
echo "  output_metrics_synthesizability.txt     (summary numbers)"
echo "  donor_synthesizability_analysis_report.txt"
echo "  acceptor_synthesizability_analysis_report.txt"
echo "  results/"
echo "    donor_synthesizability_scored.csv"
echo "    acceptor_synthesizability_scored.csv"
echo "  plots/"
echo "    sascore_distribution_donor.png"
echo "    rascore_distribution_donor.png"
echo "    combined_synthesizability_donor.png"
echo "    sascore_vs_rascore_donor.png"
echo "    boxplots_synth_scores_donor.png"
echo "    cdf_synth_scores_donor.png"
echo "    (same 6 for acceptor)"
echo "    sascore_comparison_donor_vs_acceptor.png"
echo "    rascore_comparison_donor_vs_acceptor.png"
echo "  plot_data/*.csv  (for manual re-plotting)"
echo "================================================================="
