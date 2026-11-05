#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Pipeline stage 2: post-generation metrics, property models, E_loss gate, retrosynthesis audit.
# Run from the flat run directory built by `sh hpc/setup_run_dir.sh` (see docs/PIPELINE.md);
# the script expects DMPNN.csv / Guzik_data.csv and its sibling scripts in that directory.
# Launched by hpc/sub_postgen_full_140.sh (or by hpc/sub_140.sh). Needs the stage-1 pooled pair table
# (--pairs-csv, generated_molecules_with_predictions.csv, about 7.4 GB, NOT in the repository: produced by
# stage 1 or on request from the authors). Optional model assets (models/fcd, models/aizynthfinder) are not
# shipped; when absent the pipeline falls back as described in the code. Its merged outputs used in the
# paper are committed under results/postgen/ and data/generated_libraries/.
# -----------------------------------------------------------------------------
"""
================================================================================
POST-GENERATION PIPELINE
Metrics, Property Prediction, Energy-Loss Analysis and Synthesizability Auditing
================================================================================

This script runs AFTER GAN_240_final.py has finished generating molecules. It
consumes the already-generated molecule pool and produces every *quantitative*
scientific result reported in the manuscript. GAN_240_final.py's own
ScreeningSurrogatePipeline (Method 1) is a fast interpolative TRIAGE SCREEN
only; this file owns the honest, cross-validated numbers.

--------------------------------------------------------------------------------
WHAT THIS FILE DOES (stage by stage)
--------------------------------------------------------------------------------
STAGE A  Dataset ingestion + canonical overlap verification
         - Loads DMPNN.csv (300 rows; 38 unique donors, 299 unique acceptors)
           and Guzik_data.csv (558 rows; 143 unique donor SMILES, 261 unique
           acceptor SMILES).
         - Re-checks donor/acceptor overlap on RDKit-CANONICALISED SMILES
           (an earlier exact-string check reported zero overlap; canonical
           comparison is the defensible one). Result is LOGGED, never assumed.

STAGE B  Model training (rank 0 only; artefacts saved for all ranks)
         - Per-target CV "horse race" between a classical featurised ensemble
           (Ridge / RandomForest / GaussianProcess on Morgan fingerprints) and
           a lightweight regularised D-MPNN (real directed message passing over
           bond-directed graphs, implemented in TensorFlow/Keras).
         - The winner is decided by SCAFFOLD-DISJOINT (or group-disjoint) CV
           R^2, per target, and reported transparently. GNNs are NOT assumed
           superior at these sample sizes; at N=38 unique donors a GNN cannot
           generalise and the classical ensemble is expected to win.
         - Optuna tunes hyperparameters for both candidate families.
         - Targets:
             * Acceptor (DMPNN.csv): E_b, Q20, A_EA, lambda(nm), E_S,
               Osc. strength, Diff_L-L+1, H-L Gap, HOMO, LUMO
             * Donor (DMPNN.csv):    D_lambda(nm), D_IP, D_H-L Gap, D_E_S,
               D_HOMO, D_LUMO, D_E_b
             * Device (DMPNN.csv):   Jsc, Voc, FF, PCE_max, E_loss
             * Guzik:                Donor PDI/Mn/Mw, Ehl_D, Eg_D, Ehl_A,
                                     Eg_A, hole & electron mobility (blend)

STAGE C  Rank-sharded scoring of the generated pool
         - Individual donor + acceptor property prediction.
         - Pair-level Jsc/Voc/FF/PCE + E_loss prediction.
         - E_loss physical-plausibility gating (see E_LOSS POLICY below).
         - SAScore (RDKit Contrib) for every unique molecule.
         - MOSES-standard generative metrics: Validity, Uniqueness, Novelty,
           Internal Diversity (IntDiv) -- IntDiv replaces the misnamed "MIS".
         - Three clearly-distinguished distribution distances:
             1. FP-PCA-FD  : the ORIGINAL fingerprint-PCA Frechet distance,
                             honestly relabelled (NOT Frechet ChemNet Distance)
             2. PFD        : Property-Embedding Frechet Distance -- Frechet
                             distance in the penultimate-layer activation space
                             of the Stage-B property model. Domain-appropriate.
             3. FCD        : real Frechet ChemNet Distance via the `fcd`
                             package, if installed (ChemNet is ChEMBL-trained,
                             so this carries a documented domain caveat).
         - Wasserstein per-descriptor distances + Mahalanobis distance to the
           training manifold, replacing the coarse 6-condition binary
           property-match score (which is retained for backward comparability).
         - Retrosynthesis audit of the Tier-1 shortlist (AiZynthFinder if its
           assets are present, else RAscore, else SAScore-only) -- gracefully
           degrades and never blocks the pipeline.

STAGE D  Optional RL fine-tuning (--run-rl-finetune, OFF by default)
         - Small-scale reward-shaped fine-tune of the Base/RL GAN generator.
         - Reward = w*SAScore_reward + (1-w)*TANGO-inspired heuristic, with
           Optuna tuning w and the learning rate.
         - Outputs to a SEPARATE file so the already-reported 13-approach
           totals are never retroactively altered.

STAGE E  Per-rank shard write-out to results_postgen/rank_XXX/, merged later by
         merge_rank_outputs_final.py --stage postgen

--------------------------------------------------------------------------------
E_LOSS POLICY (scientifically defensible handling of negative values)
--------------------------------------------------------------------------------
DMPNN.csv stores E_loss = E_S - V_OC_sim, where V_OC_sim comes from the crude
frontier-orbital estimate (|HOMO_D| - |LUMO_A|) - 0.3. That formula has no
mechanism preventing V_OC_sim from exceeding the optical gap, so 46 of 300 rows
(15.3%) carry a NEGATIVE E_loss, down to -0.812 eV. This is doubly unphysical:
  (i)  Voc can never exceed the optical gap (photon-energy ceiling), and
  (ii) detailed-balance/radiative-limit theory requires a non-zero MINIMUM
       radiative loss even for an ideal device (~0.25-0.3 eV).
Policy implemented (primary = OPTION 1, most defensible for publication):
  OPTION 1 (PRIMARY, default): flag and EXCLUDE the affected rows from all
       E_loss training and analysis. No numbers are invented; known-bad rows
       are simply not analysed. Controlled by E_LOSS_POLICY = "exclude".
  OPTION 2 (SUPPLEMENTARY robustness check): refit the empirical 0.3 eV offset
       against MEASURED Voc (DMPNN.csv 'Voc(V)'), recompute V_OC_sim and
       E_loss, and report how many rows remain unphysical. Written to a
       separate robustness CSV; never silently substituted for Option 1.
  OPTION 3 (reported for completeness only, NOT used for headline numbers):
       clip E_loss at the detailed-balance floor (0.25 eV).

--------------------------------------------------------------------------------
INPUTS EXPECTED IN THE WORKING DIRECTORY
--------------------------------------------------------------------------------
  DMPNN.csv                                   (training: device + descriptors)
  Guzik_data.csv                              (training: polymer/mobility data)
  <pool>/generated_molecules_with_predictions.csv   (from GAN_240_final.py)
  sascorer.py  +  fpscores.pkl.gz             (optional local copies; RDKit
                                               Contrib copy is used first)
  models/fcd/                                 (optional cached ChemNet weights)
  models/aizynthfinder/                       (optional; config.yml + assets)

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  # distributed (240 ranks), normal post-generation run
  srun --ntasks=240 python postgen_pipeline.py --distributed \
       --outdir runs/$SLURM_JOB_ID/postgen \
       --pairs-csv runs/$SLURM_JOB_ID/merged/results/generated_molecules_with_predictions.csv

  # single-process smoke test on a small slice
  python postgen_pipeline.py --pairs-csv plot_data/generated_molecules_with_predictions.csv \
       --max-pairs 5000

  # train models only (no scoring)
  python postgen_pipeline.py --train-only

  # optional RL fine-tune
  python postgen_pipeline.py --run-rl-finetune
================================================================================
"""

import argparse
import datetime
import glob
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from functools import lru_cache

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# RUNTIME ARGUMENTS  (parsed before heavy imports so --help stays fast)
# ----------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Post-generation metrics / prediction / synthesizability pipeline",
        add_help=True,
    )
    p.add_argument("--distributed", action="store_true",
                   help="Enable SLURM rank sharding (uses SLURM_PROCID / SLURM_NTASKS).")
    p.add_argument("--outdir", type=str, default="results_postgen",
                   help="Output root. Per-rank shards go to <outdir>/rank_XXX/.")
    p.add_argument("--pairs-csv", type=str, default=None,
                   help="Path to generated_molecules_with_predictions.csv. "
                        "If omitted, a set of standard locations is searched.")
    p.add_argument("--dmpnn-csv", type=str, default="DMPNN.csv",
                   help="Training CSV with device properties + DFT descriptors.")
    p.add_argument("--guzik-csv", type=str, default="Guzik_data.csv",
                   help="Training CSV with polymer/mobility/blend data.")
    p.add_argument("--models-dir", type=str, default="models",
                   help="Root for model assets (models/postgen, models/fcd, models/aizynthfinder).")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="Cap on pair rows read from the pool (0 = no cap). Useful for smoke tests.")
    p.add_argument("--optuna-trials", type=int, default=12,
                   help="Optuna trials per model family per target.")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of disjoint CV folds for the model horse race.")
    p.add_argument("--train-only", action="store_true",
                   help="Run Stage A+B (training) and exit; skip scoring.")
    p.add_argument("--score-only", action="store_true",
                   help="Skip Stage B training; load saved models and score only.")
    p.add_argument("--skip-dmpnn-candidate", action="store_true",
                   help="Exclude the D-MPNN from the horse race (classical ensemble only).")
    p.add_argument("--run-rl-finetune", action="store_true",
                   help="Run optional Stage D RL fine-tuning (OFF by default).")
    p.add_argument("--rl-epochs", type=int, default=15,
                   help="Epochs for the optional RL fine-tune.")
    p.add_argument("--rl-batch", type=int, default=64,
                   help="Sampled batch size per RL epoch (kept small on purpose).")
    p.add_argument("--audit-top-n", type=int, default=750,
                   help="Top-N pairs per approach sent to the retrosynthesis audit.")
    p.add_argument("--eloss-policy", type=str, default="exclude",
                   choices=["exclude", "refit", "clip"],
                   help="Primary E_loss handling policy (default 'exclude' = Option 1).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (rank is added to it for shard diversity).")
    p.add_argument("--log-file", type=str, default="postgen_pipeline_progress.log",
                   help="Single global log file (all ranks append, rank-tagged).")
    args, _ = p.parse_known_args()
    return args


ARGS = _parse_args()

RANK = int(os.environ.get("SLURM_PROCID", "0")) if ARGS.distributed else 0
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1")) if ARGS.distributed else 1
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", "0"))
CPUS_PER_TASK = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1))))
IS_DISTRIBUTED = bool(ARGS.distributed and WORLD_SIZE > 1)
IS_MASTER = (RANK == 0)

BASE_WORKDIR = os.getcwd()
MODELS_ROOT = os.path.abspath(os.path.join(BASE_WORKDIR, ARGS.models_dir))
POSTGEN_MODELS_DIR = os.path.join(MODELS_ROOT, "postgen")
FCD_MODELS_DIR = os.path.join(MODELS_ROOT, "fcd")
AIZYNTH_MODELS_DIR = os.path.join(MODELS_ROOT, "aizynthfinder")

OUT_ROOT = os.path.abspath(ARGS.outdir)
RANK_OUTDIR = os.path.join(OUT_ROOT, "rank_%03d" % RANK)


def safe_makedirs(path, retries=8, base_sleep=0.2):
    """Create directory with retry/backoff for flaky shared filesystems.

    Copied verbatim in behaviour from GAN_240_final.py so that both pipelines
    tolerate the same NFS/Lustre metadata storms on a 240-rank launch.
    """
    for attempt in range(retries):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt))
    return False


for _d in (MODELS_ROOT, POSTGEN_MODELS_DIR, FCD_MODELS_DIR, AIZYNTH_MODELS_DIR,
           OUT_ROOT, RANK_OUTDIR):
    safe_makedirs(_d)

# ----------------------------------------------------------------------------
# LOGGING -- ONE global rank-tagged log file, mirroring the pattern already
# used for gan_pipeline_progress.log so both pipelines look identical in the
# job output and can be grepped the same way (e.g. "rank=017").
# ----------------------------------------------------------------------------


def setup_logging():
    log_path = os.path.join(BASE_WORKDIR, ARGS.log_file)
    lg = logging.getLogger("postgen")
    lg.setLevel(logging.DEBUG)
    lg.handlers = []
    lg.propagate = False

    fmt = logging.Formatter(
        fmt="%%(asctime)s - rank=%03d - %%(levelname)s - %%(message)s" % RANK
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


logger = setup_logging()

logger.info("=" * 80)
logger.info("POST-GENERATION PIPELINE STARTING")
logger.info("=" * 80)
logger.info("  Timestamp        : %s" % datetime.datetime.now().isoformat())
logger.info("  Rank / World     : %d / %d (distributed=%s)" % (RANK, WORLD_SIZE, IS_DISTRIBUTED))
logger.info("  Local rank       : %d, CPUs/task: %d" % (LOCAL_RANK, CPUS_PER_TASK))
logger.info("  Base workdir     : %s" % BASE_WORKDIR)
logger.info("  Output root      : %s" % OUT_ROOT)
logger.info("  Rank outdir      : %s" % RANK_OUTDIR)
logger.info("  Models root      : %s" % MODELS_ROOT)
logger.info("  E_loss policy    : %s" % ARGS.eloss_policy)
logger.info("  Optuna trials    : %d, CV folds: %d" % (ARGS.optuna_trials, ARGS.cv_folds))
logger.info("  RL fine-tune     : %s" % ARGS.run_rl_finetune)
logger.info("=" * 80)

# Rank-differentiated seeding (same rationale as GAN_240_final.py: without it,
# every rank does identical work and parallelism yields zero diversity gain).
_SEED = ARGS.seed + 7919 * RANK
random.seed(_SEED)
np.random.seed(_SEED % (2 ** 32 - 1))
logger.info("Seeded RNGs with base=%d rank-adjusted seed=%d" % (ARGS.seed, _SEED))

# ----------------------------------------------------------------------------
# OPTIONAL / GUARDED THIRD-PARTY IMPORTS
# Every heavy or optional dependency is imported behind a guard that records a
# module-level *_AVAILABLE flag, exactly like GAN_240_final.py does. The
# pipeline degrades gracefully instead of crashing when something is missing.
# ----------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit import RDConfig
    from rdkit import DataStructs
    from rdkit.Chem import AllChem, BRICS, Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit.Chem.AllChem import GetMorganGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
    logger.info("RDKit loaded successfully")
except ImportError as e:
    RDKIT_AVAILABLE = False
    logger.error("RDKit NOT available (%s). This pipeline cannot run without RDKit." % e)
    raise

try:
    import selfies as sf
    SELFIES_AVAILABLE = True
    logger.info("SELFIES loaded successfully for fallback molecule representation")
except ImportError:
    sf = None
    SELFIES_AVAILABLE = False
    logger.warning("SELFIES not available. Token/hash fallback representations will still be used.")

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn loaded successfully")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.error("scikit-learn NOT available (%s). Required." % e)
    raise

try:
    from scipy.linalg import sqrtm
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import wasserstein_distance, pearsonr
    SCIPY_AVAILABLE = True
    logger.info("SciPy loaded successfully")
except ImportError as e:
    SCIPY_AVAILABLE = False
    logger.error("SciPy NOT available (%s). Required." % e)
    raise

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    logger.info("Optuna loaded successfully (hyperparameter optimisation enabled)")
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available -- models will use fixed default hyperparameters.")

# TensorFlow is needed for the real D-MPNN candidate and the optional RL
# fine-tune. Kept optional so the classical-ensemble path still runs without it.
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    TF_AVAILABLE = True
    logger.info("TensorFlow loaded successfully (D-MPNN + RL fine-tune enabled)")
except Exception as e:
    tf = None
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available (%s). D-MPNN candidate and RL fine-tune disabled; "
                   "classical ensemble will be used for every target." % e)

# ----------------------------------------------------------------------------
# SAScore SETUP (RDKit Contrib)
# Uses the exact import pattern requested: prefer RDKit's own Contrib copy;
# the urllib download lines are retained (commented) because the two required
# files are already present locally.
# ----------------------------------------------------------------------------

# Download sascorer.py if not exists
sascorer_path = "sascorer.py"
# if not os.path.exists(sascorer_path):
#     url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py"
#     urllib.request.urlretrieve(url, sascorer_path)

# Download fpscores.pkl.gz if not exists
fpscores_path = "fpscores.pkl.gz"
# if not os.path.exists(fpscores_path):
#     url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"
#     urllib.request.urlretrieve(url, fpscores_path)

# --- SAScore setup via RDKit Contrib -----------------------------------------
SASCORER_AVAILABLE = False
sascorer = None
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # noqa: E402
    SASCORER_AVAILABLE = True
    logger.info("SAScore (RDKit Contrib sascorer) loaded successfully from %s"
                % os.path.join(RDConfig.RDContribDir, "SA_Score"))
except Exception as e:
    logger.warning("SAScore via RDKit Contrib failed (%s); trying local sascorer.py" % e)
    try:
        sys.path.insert(0, BASE_WORKDIR)
        import sascorer  # noqa: E402
        SASCORER_AVAILABLE = True
        logger.info("SAScore loaded from local sascorer.py in %s" % BASE_WORKDIR)
    except Exception as e2:
        SASCORER_AVAILABLE = False
        logger.warning("SAScore unavailable (%s). SA-based reward/metrics will be skipped." % e2)

# Real Frechet ChemNet Distance (optional).
try:
    if os.path.isdir(FCD_MODELS_DIR):
        os.environ.setdefault("TORCH_HOME", FCD_MODELS_DIR)
    import fcd as fcd_module
    FCD_AVAILABLE = True
    logger.info("fcd package loaded successfully (real Frechet ChemNet Distance enabled)")
except Exception as e:
    fcd_module = None
    FCD_AVAILABLE = False
    logger.warning("fcd package not available (%s). Real FCD will be reported as NaN; "
                   "PFD and FP-PCA-FD are unaffected." % e)

# RAscore (distilled AiZynthFinder classifier) -- optional audit backend.
try:
    from RAscore import RAscore_XGB
    RASCORE_AVAILABLE = True
    logger.info("RAscore (XGB) loaded successfully")
except Exception:
    try:
        from RAscore import RAscore_NN
        RASCORE_AVAILABLE = True
        logger.info("RAscore (NN) loaded successfully")
    except Exception as e:
        RASCORE_AVAILABLE = False
        logger.warning("RAscore not available (%s). Retrosynthesis audit will fall back." % e)

# AiZynthFinder -- optional audit backend. Requires stock + expansion policy
# assets in models/aizynthfinder/. Absent by design (multi-GB download); the
# audit degrades to RAscore/SAScore instead of failing.
try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
    AIZYNTHFINDER_AVAILABLE = True
    logger.info("AiZynthFinder package importable")
except Exception as e:
    AiZynthFinder = None
    AIZYNTHFINDER_AVAILABLE = False
    logger.warning("AiZynthFinder not importable (%s)." % e)

AIZYNTH_CONFIG_PATH = os.path.join(AIZYNTH_MODELS_DIR, "config.yml")
AIZYNTH_ASSETS_PRESENT = os.path.exists(AIZYNTH_CONFIG_PATH)
if AIZYNTHFINDER_AVAILABLE and not AIZYNTH_ASSETS_PRESENT:
    logger.warning("AiZynthFinder installed but assets missing (%s not found). "
                   "Run 'download_public_data %s' to enable full retrosynthesis search."
                   % (AIZYNTH_CONFIG_PATH, AIZYNTH_MODELS_DIR))


# ============================================================================
# MOLECULE PROCESSING UTILITIES
#
# These are copied VERBATIM (same logic, same behaviour, same edge-case
# handling) from GAN_240_final.py so that this pipeline treats SMILES exactly
# as the generation pipeline did -- identical sanitisation, identical valence
# repair, identical canonicalisation, identical Morgan fingerprints and the
# identical SELFIES/token-hash fallback chain for molecules RDKit cannot parse.
#
# WHY COPIED RATHER THAN IMPORTED: GAN_240_final.py executes its full dataset
# ingestion and GAN training as module-level top-level code (it is not guarded
# behind `if __name__ == "__main__"`). Importing it would re-trigger the entire
# 240-rank generation run as an import side effect. Duplicating this small set
# of pure functions is the safe choice; it keeps the generation pipeline's risk
# surface completely untouched. This is a deliberate tradeoff, not an oversight.
# ============================================================================


def sanitize_molecule(mol):
    """Attempt to sanitize and fix common issues in molecules"""
    logger.debug("Starting molecule sanitization...")
    try:
        if mol is None:
            logger.debug("Received None molecule for sanitization")
            return None
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL)
        logger.debug("Full sanitization successful")
        return mol
    except Exception as e:
        logger.debug("Full sanitization failed: %s, trying step-by-step approach..." % e)
        try:
            sanitize_ops = [
                Chem.SANITIZE_CLEANUP,
                Chem.SANITIZE_PROPERTIES,
                Chem.SANITIZE_SYMMRINGS,
                Chem.SANITIZE_KEKULIZE,
                Chem.SANITIZE_FINDRADICALS,
                Chem.SANITIZE_SETAROMATICITY,
                Chem.SANITIZE_SETCONJUGATION,
                Chem.SANITIZE_SETHYBRIDIZATION,
                Chem.SANITIZE_CLEANUPCHIRALITY,
            ]
            for op in sanitize_ops:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=op)
                except Exception:
                    continue
            return mol
        except Exception as final_e:
            logger.debug("All sanitization attempts failed: %s" % final_e)
            return None


def fix_valence_issues(mol):
    """Fix common valence issues"""
    if mol is None:
        return None
    try:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N" and atom.GetExplicitValence() > 3:
                atom.SetFormalCharge(1)
            elif atom.GetSymbol() == "Se":
                atom.SetFormalCharge(0)
            elif atom.GetSymbol() == "Zn":
                atom.SetFormalCharge(2)
            if atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED:
                atom.SetHybridization(Chem.HybridizationType.SP3)
        return mol
    except Exception:
        return None


def is_valid_smiles(smiles):
    """Check if SMILES string is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


@lru_cache(maxsize=4096)
def safe_mol_from_smarts(pattern):
    """Parse SMARTS safely; return None if invalid."""
    try:
        return Chem.MolFromSmarts(pattern)
    except Exception:
        return None


def count_substruct(mol, pattern):
    """Count substructure matches safely; returns 0 if SMARTS invalid."""
    q = safe_mol_from_smarts(pattern)
    if mol is None or q is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(q))
    except Exception:
        return 0


@lru_cache(maxsize=16)
def _get_morgan_gen_cached(radius=2, nBits=2048):
    return GetMorganGenerator(radius, fpSize=nBits)


def morgan_fp(mol, radius=2, nBits=2048):
    """Morgan fingerprint as ExplicitBitVect (Tanimoto-compatible)."""
    try:
        return GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    except Exception as e:
        logger.debug("Morgan fingerprint failed (r=%s, nBits=%s): %s" % (radius, nBits, e))
        return None


def morgan_fp_numpy(mol, radius=2, nBits=2048):
    """Morgan fingerprint as a float32 numpy array (model input)."""
    try:
        return _get_morgan_gen_cached(radius, nBits).GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        fp = morgan_fp(mol, radius=radius, nBits=nBits)
        if fp is None:
            return None
        arr = np.zeros((nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)


def canonical_smiles(smiles):
    try:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonical_smiles_set(smiles_list):
    can = set()
    for smi in smiles_list:
        c = canonical_smiles(smi)
        if c:
            can.add(c)
    return can


def fingerprints_from_smiles(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = morgan_fp(mol, radius=2, nBits=2048)
        if fp is not None:
            fps.append(fp)
    return fps


def tokenize_smiles_fallback(smiles):
    text = str(smiles).strip()
    if not text:
        return []
    token_pattern = r"Cl|Br|Si|Se|@@?|%\d{2}|\[[^\]]+\]|\d|\(|\)|=|#|-|\\|/|\.|\+|:|~|@|\?|>|\*|\$|[A-Za-z]"
    tokens = re.findall(token_pattern, text)
    if tokens:
        return tokens
    return list(text)


def stable_hashed_fingerprint_from_tokens(tokens, n_bits=2048):
    fp = np.zeros(n_bits, dtype=np.float32)
    if not tokens:
        return fp
    for i, token in enumerate(tokens):
        for feature in (token, "%s@%d" % (token, i),
                        "".join(tokens[i:i + 2]), "".join(tokens[i:i + 3])):
            if not feature:
                continue
            digest = hashlib.blake2b(feature.encode("utf-8", errors="ignore"),
                                     digest_size=8).hexdigest()
            fp[int(digest, 16) % n_bits] = 1.0
    return fp


def fallback_representation_from_smiles(smiles, n_bits=2048):
    """SELFIES-preferred, token-hash fallback representation for unparseable SMILES."""
    text = str(smiles).strip()
    tokens = tokenize_smiles_fallback(text)
    selfies_text = None
    selfies_tokens = []
    if SELFIES_AVAILABLE and text:
        try:
            selfies_text = sf.encoder(text)
            selfies_tokens = list(sf.split_selfies(selfies_text))
        except Exception:
            selfies_text = None
            selfies_tokens = []
    feature_tokens = selfies_tokens if selfies_tokens else tokens
    hashed_fp = stable_hashed_fingerprint_from_tokens(feature_tokens, n_bits=n_bits)
    usable = bool(text and (tokens or selfies_tokens) and np.any(hashed_fp))
    return {
        "raw_smiles": text,
        "tokens": tokens,
        "selfies": selfies_text,
        "selfies_tokens": selfies_tokens,
        "hashed_fp": hashed_fp,
        "usable": usable,
        "representation": "selfies_hash" if selfies_tokens else "token_hash",
    }


_FEATURIZE_CACHE = {}
_SCAFFOLD_CACHE = {}


def featurize_smiles(smiles, n_bits=2048):
    """Primary featurisation path: RDKit Morgan; SELFIES/token-hash fallback.

    Mirrors ScreeningSurrogatePipeline._smiles_to_fp() in GAN_240_final.py so
    that a molecule which was featurisable during generation stays featurisable
    here, and vice versa.

    Memoised: the training CSVs reuse the same molecules across many targets
    (299 acceptors x 11 targets) and the generated pool is a cross-product, so
    without caching the same fingerprint would be recomputed thousands of times.
    """
    key = (str(smiles).strip(), n_bits)
    if key in _FEATURIZE_CACHE:
        return _FEATURIZE_CACHE[key]
    val = _featurize_smiles_uncached(smiles, n_bits=n_bits)
    if len(_FEATURIZE_CACHE) < 400000:
        _FEATURIZE_CACHE[key] = val
    return val


def _featurize_smiles_uncached(smiles, n_bits=2048):
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        fb = fallback_representation_from_smiles(text, n_bits=n_bits)
        if fb["usable"]:
            logger.debug("Featurisation using %s fallback for: %s" % (fb["representation"], text[:80]))
            return fb["hashed_fp"]
        return None
    arr = morgan_fp_numpy(mol, radius=2, nBits=n_bits)
    if arr is None:
        fb = fallback_representation_from_smiles(text, n_bits=n_bits)
        if fb["usable"]:
            return fb["hashed_fp"]
        return None
    return arr


def osc_trait_signature(smiles, molecule_type="Generic"):
    """OSC-aware structural trait signature (used for balanced ordering/stratification)."""
    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        tokens = tokenize_smiles_fallback(text)
        token_text = " ".join(tokens).lower()
        fallback_tags = [
            "fallback",
            "cyano" if "n" in token_text and "#" in token_text else "no_cyano",
            "halogen" if any(x in token_text for x in ["f", "cl", "br", "i"]) else "no_halogen",
            "sulfur" if "s" in token_text else "no_sulfur",
            "selenium" if "se" in token_text else "no_selenium",
            "len_%d" % min(len(tokens) // 20, 6),
        ]
        return "|".join(fallback_tags)
    try:
        heavy = mol.GetNumHeavyAtoms()
        rings = mol.GetRingInfo().NumRings()
        aromatic = Descriptors.NumAromaticRings(mol)
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        rot = Descriptors.NumRotatableBonds(mol)
        cyano = count_substruct(mol, "C#N")
        carbonyl = count_substruct(mol, "C=O") + count_substruct(mol, "C(=O)")
        halogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in {"F", "Cl", "Br", "I"})
        sulfur = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
        selenium = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "Se")
        thiophene = count_substruct(mol, "c1ccsc1") + count_substruct(mol, "c1sccc1")
        bdt = count_substruct(mol, "c1ccc2sccc2c1")
        benzothiadiazole = count_substruct(mol, "c1nsnc1") + count_substruct(mol, "n1sncc1")
        indanone = count_substruct(mol, "O=c1c(*)c2ccccc2c1")
        mt = str(molecule_type).lower()
        tags = [
            "heavy_%d" % min(heavy // 20, 8),
            "rings_%d" % min(rings, 9),
            "arom_%d" % min(aromatic, 8),
            "hetero_%d" % min(hetero // 3, 8),
            "rot_%d" % min(rot // 8, 8),
            "hal_%d" % min(halogen, 6),
            "se" if selenium else "no_se",
            "s_rich" if sulfur >= 3 else ("s_some" if sulfur else "s_none"),
        ]
        if "accept" in mt:
            tags.extend([
                "cyano_%d" % min(cyano, 4),
                "carbonyl_%d" % min(carbonyl, 4),
                "indanone_like" if indanone else "non_indanone",
                "ada_push_pull" if cyano >= 2 and rings >= 3 else "non_ada",
            ])
        elif "donor" in mt:
            tags.extend([
                "bdt" if bdt else "non_bdt",
                "thiophene_rich" if thiophene >= 2 else ("thiophene" if thiophene else "no_thiophene"),
                "bt_bse_like" if benzothiadiazole or selenium else "non_bt_bse",
                "fluorinated" if halogen else "non_fluorinated",
            ])
        else:
            tags.extend(["cyano_%d" % min(cyano, 4), "bdt" if bdt else "non_bdt"])
        return "|".join(tags)
    except Exception as e:
        logger.debug("Trait signature failed for %s: %s" % (text[:80], e))
        return "trait_error"


def balanced_trait_order(smiles_list, molecule_type="Generic", max_items=None):
    """Round-robin ordering across OSC trait groups (rare traits first)."""
    groups = {}
    for smi in smiles_list:
        groups.setdefault(osc_trait_signature(smi, molecule_type), []).append(smi)
    ordered_keys = sorted(groups.keys(), key=lambda key: (len(groups[key]), key))
    ordered = []
    max_group_len = max((len(values) for values in groups.values()), default=0)
    for i in range(max_group_len):
        for key in ordered_keys:
            values = groups[key]
            if i < len(values):
                ordered.append(values[i])
                if max_items is not None and len(ordered) >= max_items:
                    return ordered
    return ordered


def murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES; used to build scaffold-disjoint CV groups.

    Memoised for the same reason as featurize_smiles: scaffold groups are rebuilt
    once per target over the same molecule set.
    """
    key = str(smiles).strip()
    if key in _SCAFFOLD_CACHE:
        return _SCAFFOLD_CACHE[key]
    val = _murcko_scaffold_uncached(smiles)
    if len(_SCAFFOLD_CACHE) < 400000:
        _SCAFFOLD_CACHE[key] = val
    return val


def _murcko_scaffold_uncached(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return "UNPARSEABLE"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return "NO_SCAFFOLD"
        s = Chem.MolToSmiles(scaf, canonical=True)
        return s if s else "NO_SCAFFOLD"
    except Exception:
        return "SCAFFOLD_ERROR"


def shard_list(items, rank, world_size):
    """Deterministic list sharding (identical to GAN_240_final.py)."""
    if world_size <= 1:
        return items
    return items[rank::world_size]


def shard_dataframe(df, rank, world_size):
    """Deterministic row-striped DataFrame sharding."""
    if world_size <= 1:
        return df
    return df.iloc[rank::world_size].copy()


def compute_sascore(mol):
    """SAScore in its native 1 (easy) .. 10 (hard) range. None if unavailable."""
    if not SASCORER_AVAILABLE or mol is None:
        return None
    try:
        return float(sascorer.calculateScore(mol))
    except Exception as e:
        logger.debug("SAScore failed: %s" % e)
        return None


def sascore_to_reward(sa):
    """Map SAScore (1..10, lower=easier) to a [0,1] reward (higher=easier).

    Linear rescale: reward = (10 - SA) / 9. Deterministic and monotone, which
    is what an RL reward needs; no tunable shape parameters hidden in here.
    """
    if sa is None:
        return 0.0
    return float(np.clip((10.0 - float(sa)) / 9.0, 0.0, 1.0))


# ============================================================================
# TANGO-INSPIRED SYNTHESIZABILITY HEURISTIC  (SCREENING USE ONLY)
#
# HONEST NAMING: this is a TANGO-INSPIRED building-block-overlap heuristic. It
# combines a fuzzy-matching-substructure (FMS) term with a max-Tanimoto-to-
# building-block term over an OPV-specific block library. It is NOT the
# published retrosynthesis-constrained TANGO framework and its value must NOT
# be interpreted as "the probability that a synthesis route exists". It is used
# here strictly as a cheap FIRST-PASS SCREEN and as one component of the
# optional RL reward. Publishable synthesizability claims are gated on the
# retrosynthesis audit (Stage C), never on this number alone.
#
# Behaviour is kept identical to GAN_240_final.py's TANGOSynthesizabilityReward
# so heuristic scores remain comparable across the two pipelines.
# ============================================================================


class TangoInspiredHeuristic:
    """TANGO-inspired continuous synthesizability screen in [0, 1] (higher = easier)."""

    OPV_BUILDING_BLOCKS = [
        "c1ccsc1", "c1ccc2sccc2c1", "c1cc[se]c1",
        "C1=CC(=O)NC(=O)C1", "c1ccncc1", "c1ccc(cc1)",
        "c1ccc2[nH]ccc2c1", "c1nsnc1", "C#N",
        "C(=O)", "F", "Cl", "Br",
        "C=C", "C#C", "C(=O)O",
        "c1ccc2c(c1)C(=O)NC2=O",
        "c1sc2c(c1)sc1ccccc12",
        "C1C(=O)c2ccccc2C1=O",
        "c1ccoc1",
    ]

    def __init__(self, training_smiles=None, alpha=0.5, radius=2, n_bits=2048,
                 max_brics_sources=150):
        # PERFORMANCE NOTE (measured, not guessed): BRICS decomposition of these
        # OPV molecules is expensive -- decomposing 400 of them took ~4.6 min in
        # a single-process smoke test. The block library saturates long before
        # that (400 molecules yielded 2023 fragments but only 184 usable unique
        # blocks after sanitisation), so the source cap is 150 by default. Raise
        # it only if the logged block count looks starved.
        self.alpha = float(alpha)
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.building_block_mols = []
        self.building_block_fps = []
        self.building_block_smiles = []

        logger.info("=" * 80)
        logger.info("INITIALIZING TANGO-INSPIRED SYNTHESIZABILITY HEURISTIC (screen only)")
        logger.info("  alpha (FMS weight): %.2f | radius: %d | bits: %d"
                    % (self.alpha, self.radius, self.n_bits))

        all_bb = set(self.OPV_BUILDING_BLOCKS)
        if training_smiles:
            src = list(training_smiles)[:max_brics_sources]
            logger.info("  Extracting BRICS fragments from %d training molecules..." % len(src))
            frag_count = 0
            for smi in src:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    continue
                try:
                    frags = BRICS.BRICSDecompose(mol)
                    all_bb.update(frags)
                    frag_count += len(frags)
                except Exception:
                    continue
            logger.info("  Extracted %d BRICS fragments" % frag_count)

        gen = _get_morgan_gen_cached(self.radius, self.n_bits)
        for bb_smi in all_bb:
            try:
                bb_mol = Chem.MolFromSmiles(bb_smi)
                if bb_mol is None:
                    continue
                Chem.SanitizeMol(bb_mol)
                self.building_block_mols.append(bb_mol)
                self.building_block_fps.append(gen.GetFingerprintAsNumPy(bb_mol).astype(np.float32))
                self.building_block_smiles.append(bb_smi)
            except Exception:
                continue

        # Precompute building-block on-bit sets for the FMS term (big speedup
        # over recomputing a fingerprint per block per molecule).
        self._bb_onbits = []
        for bb_mol in self.building_block_mols:
            try:
                fp = GetMorganFingerprintAsBitVect(bb_mol, self.radius, nBits=self.n_bits)
                self._bb_onbits.append(set(fp.GetOnBits()))
            except Exception:
                self._bb_onbits.append(set())

        self._bb_matrix = (np.vstack(self.building_block_fps)
                           if self.building_block_fps else np.zeros((0, self.n_bits), dtype=np.float32))
        self._bb_sums = self._bb_matrix.sum(axis=1) if self._bb_matrix.size else np.zeros((0,))

        logger.info("  Building-block library: %d blocks loaded" % len(self.building_block_fps))
        logger.info("TANGO-INSPIRED HEURISTIC INITIALIZED")
        logger.info("=" * 80)

    def score(self, mol):
        """Return TANGO-inspired heuristic score in [0,1] (0.0 on any failure)."""
        if mol is None or self._bb_matrix.size == 0:
            return 0.0
        try:
            mol_fp_bv = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            mol_on = set(mol_fp_bv.GetOnBits())
            best_fms = 0.0
            for bb_on in self._bb_onbits:
                if not bb_on:
                    continue
                fms = len(bb_on & mol_on) / float(len(bb_on))
                if fms > best_fms:
                    best_fms = fms

            mol_vec = np.zeros((self.n_bits,), dtype=np.float32)
            for b in mol_on:
                mol_vec[b] = 1.0
            dots = self._bb_matrix.dot(mol_vec)
            denoms = self._bb_sums + mol_vec.sum() - dots
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(denoms > 0, dots / denoms, 0.0)
            tan_sim = float(np.max(sims)) if sims.size else 0.0

            return float(self.alpha * best_fms + (1.0 - self.alpha) * tan_sim)
        except Exception as e:
            logger.debug("TANGO-inspired scoring failed: %s" % e)
            return 0.0

    def score_smiles(self, smiles):
        return self.score(Chem.MolFromSmiles(str(smiles).strip()))


# ============================================================================
# REAL DIRECTED MESSAGE-PASSING NEURAL NETWORK (D-MPNN)
#
# This is an actual D-MPNN in the Gilmer/Yang ("chemprop") sense: messages live
# on DIRECTED BONDS, are updated for `depth` steps with a shared weight matrix
# and a residual connection to the initial bond feature, then aggregated to
# atoms and pooled to a molecule vector, which is fed to a small readout MLP.
#
# Implementation notes:
#  * Featurisation is precomputed into dense, padded tensors so the Keras model
#    stays simple and CPU-friendly (this cluster runs 1 thread per rank).
#  * The message passing is expressed with a padded bond-incidence tensor and
#    masked gathers, avoiding ragged tensors for portability across TF versions.
#  * The `reverse bond` subtraction that distinguishes a true D-MPNN from a
#    naive edge network IS implemented (a bond never passes its own reverse
#    message back to itself).
#  * A penultimate embedding is exposed via `embedding_model` for the
#    Property-Embedding Frechet Distance (PFD).
#
# HONEST EXPECTATION: with 299 unique acceptors (and only 38 unique donors) this
# D-MPNN is data-starved by GNN standards. It is entered into a per-target CV
# HORSE RACE against the classical featurised ensemble and only used where it
# actually wins on held-out, group-disjoint folds.
# ============================================================================

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "Se", "Si", "P", "B", "H", "OTHER"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

MAX_ATOMS_DEFAULT = 160
MAX_BONDS_DEFAULT = 360


def _onehot(value, choices):
    vec = [0.0] * len(choices)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom):
    """Per-atom feature vector (deterministic, RDKit-only)."""
    feats = []
    feats += _onehot(atom.GetSymbol(), ATOM_SYMBOLS)
    feats += _onehot(str(atom.GetHybridization()), HYBRIDIZATIONS)
    feats.append(float(atom.GetDegree()) / 4.0)
    feats.append(float(atom.GetTotalNumHs()) / 4.0)
    feats.append(float(atom.GetFormalCharge()))
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(float(atom.GetMass()) / 200.0)
    return feats


def bond_features(bond):
    """Per-bond feature vector (deterministic, RDKit-only)."""
    feats = []
    feats += _onehot(str(bond.GetBondType()), BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    return feats


ATOM_FDIM = len(ATOM_SYMBOLS) + len(HYBRIDIZATIONS) + 6
BOND_FDIM = len(BOND_TYPES) + 2


def mol_to_dmpnn_graph(smiles, max_atoms=MAX_ATOMS_DEFAULT, max_bonds=MAX_BONDS_DEFAULT):
    """Convert SMILES to padded directed-bond graph tensors for the D-MPNN.

    Returns dict with:
      f_atoms      (max_atoms, ATOM_FDIM)
      f_bonds      (max_bonds, ATOM_FDIM + BOND_FDIM)  initial directed messages
      bond_src     (max_bonds,)  origin atom of each directed bond
      bond_dst     (max_bonds,)  target atom of each directed bond
      bond_rev     (max_bonds,)  index of the reverse directed bond
      a2b          (max_atoms, MAX_DEG) incoming directed-bond indices per atom
      a2b_mask     (max_atoms, MAX_DEG)
      atom_mask    (max_atoms,)
      bond_mask    (max_bonds,)
    Returns None when the molecule cannot be represented (unparseable, or larger
    than the padding budget).
    """
    MAX_DEG = 6
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_atoms > max_atoms:
        return None
    n_dir_bonds = 2 * mol.GetNumBonds()
    if n_dir_bonds > max_bonds:
        return None

    f_atoms = np.zeros((max_atoms, ATOM_FDIM), dtype=np.float32)
    for atom in mol.GetAtoms():
        f_atoms[atom.GetIdx()] = atom_features(atom)

    f_bonds = np.zeros((max_bonds, ATOM_FDIM + BOND_FDIM), dtype=np.float32)
    bond_src = np.zeros((max_bonds,), dtype=np.int32)
    bond_dst = np.zeros((max_bonds,), dtype=np.int32)
    bond_rev = np.zeros((max_bonds,), dtype=np.int32)
    bond_mask = np.zeros((max_bonds,), dtype=np.float32)
    a2b = np.zeros((max_atoms, MAX_DEG), dtype=np.int32)
    a2b_mask = np.zeros((max_atoms, MAX_DEG), dtype=np.float32)
    a2b_fill = np.zeros((max_atoms,), dtype=np.int32)

    bi = 0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # directed bond a1 -> a2
        f_bonds[bi] = list(f_atoms[a1]) + bf
        bond_src[bi] = a1
        bond_dst[bi] = a2
        bond_mask[bi] = 1.0
        # directed bond a2 -> a1
        f_bonds[bi + 1] = list(f_atoms[a2]) + bf
        bond_src[bi + 1] = a2
        bond_dst[bi + 1] = a1
        bond_mask[bi + 1] = 1.0
        bond_rev[bi] = bi + 1
        bond_rev[bi + 1] = bi
        # incoming bond lists (bond bi terminates at a2; bi+1 terminates at a1)
        if a2b_fill[a2] < MAX_DEG:
            a2b[a2, a2b_fill[a2]] = bi
            a2b_mask[a2, a2b_fill[a2]] = 1.0
            a2b_fill[a2] += 1
        if a2b_fill[a1] < MAX_DEG:
            a2b[a1, a2b_fill[a1]] = bi + 1
            a2b_mask[a1, a2b_fill[a1]] = 1.0
            a2b_fill[a1] += 1
        bi += 2

    atom_mask = np.zeros((max_atoms,), dtype=np.float32)
    atom_mask[:n_atoms] = 1.0

    return {
        "f_atoms": f_atoms, "f_bonds": f_bonds,
        "bond_src": bond_src, "bond_dst": bond_dst, "bond_rev": bond_rev,
        "a2b": a2b, "a2b_mask": a2b_mask,
        "atom_mask": atom_mask, "bond_mask": bond_mask,
        "n_atoms": n_atoms, "n_bonds": n_dir_bonds,
    }


def build_dmpnn_model(hidden_dim=64, depth=3, dropout=0.15, ffn_hidden=64,
                      learning_rate=1e-3, max_atoms=MAX_ATOMS_DEFAULT,
                      max_bonds=MAX_BONDS_DEFAULT, n_outputs=1):
    """Construct the D-MPNN. Returns (train_model, embedding_model).

    embedding_model exposes the pooled molecule representation (the penultimate
    layer) used as the PFD embedding space.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow unavailable; cannot build D-MPNN.")

    MAX_DEG = 6
    in_f_atoms = Input(shape=(max_atoms, ATOM_FDIM), name="f_atoms")
    in_f_bonds = Input(shape=(max_bonds, ATOM_FDIM + BOND_FDIM), name="f_bonds")
    in_a2b = Input(shape=(max_atoms, MAX_DEG), dtype="int32", name="a2b")
    in_a2b_mask = Input(shape=(max_atoms, MAX_DEG), name="a2b_mask")
    in_bond_src = Input(shape=(max_bonds,), dtype="int32", name="bond_src")
    in_bond_rev = Input(shape=(max_bonds,), dtype="int32", name="bond_rev")
    in_atom_mask = Input(shape=(max_atoms,), name="atom_mask")
    in_bond_mask = Input(shape=(max_bonds,), name="bond_mask")

    W_i = layers.Dense(hidden_dim, use_bias=False, name="W_i")
    W_h = layers.Dense(hidden_dim, use_bias=False, name="W_h")
    W_o = layers.Dense(hidden_dim, name="W_o")
    act = layers.ReLU()
    drop = layers.Dropout(dropout)

    # Initial directed-bond hidden state h0 = ReLU(W_i * f_bond)
    h0 = act(W_i(in_f_bonds))                                  # (B, max_bonds, H)
    h = h0
    bmask = layers.Reshape((max_bonds, 1))(in_bond_mask)
    h = layers.Multiply()([h, bmask])

    for step in range(max(1, int(depth)) - 1):
        # Aggregate incoming messages per atom: m_atom[v] = sum_{b -> v} h[b]
        def _gather_atom_msgs(args):
            hh, a2b_idx, a2b_m = args
            g = tf.gather(hh, a2b_idx, batch_dims=1)           # (B, A, DEG, H)
            g = g * tf.expand_dims(a2b_m, -1)
            return tf.reduce_sum(g, axis=2)                    # (B, A, H)

        m_atom = layers.Lambda(_gather_atom_msgs, name="agg_atom_%d" % step)(
            [h, in_a2b, in_a2b_mask])

        # Directed-bond message: m_bond[b] = m_atom[src(b)] - h[rev(b)]
        # The subtraction of the reverse bond is what makes this a true D-MPNN
        # rather than a naive edge network (no bond echoes itself back).
        def _bond_msgs(args):
            ma, hh, src_idx, rev_idx = args
            from_src = tf.gather(ma, src_idx, batch_dims=1)     # (B, max_bonds, H)
            rev_h = tf.gather(hh, rev_idx, batch_dims=1)        # (B, max_bonds, H)
            return from_src - rev_h

        m_bond = layers.Lambda(_bond_msgs, name="bond_msg_%d" % step)(
            [m_atom, h, in_bond_src, in_bond_rev])

        # h = ReLU(h0 + W_h * m_bond)   (residual to the initial bond feature)
        h = act(layers.Add()([h0, W_h(m_bond)]))
        h = drop(h)
        h = layers.Multiply()([h, bmask])

    # Final atom representation from incoming bond messages + atom features
    def _final_atom(args):
        hh, a2b_idx, a2b_m = args
        g = tf.gather(hh, a2b_idx, batch_dims=1)
        g = g * tf.expand_dims(a2b_m, -1)
        return tf.reduce_sum(g, axis=2)

    m_atom_final = layers.Lambda(_final_atom, name="agg_atom_final")(
        [h, in_a2b, in_a2b_mask])
    atom_repr = act(W_o(layers.Concatenate(axis=-1)([in_f_atoms, m_atom_final])))
    amask = layers.Reshape((max_atoms, 1))(in_atom_mask)
    atom_repr = layers.Multiply()([atom_repr, amask])

    # Mean-pool over real atoms only -> molecule embedding (penultimate layer)
    def _masked_mean(args):
        ar, am = args
        s = tf.reduce_sum(ar, axis=1)
        n = tf.maximum(tf.reduce_sum(am, axis=1, keepdims=True), 1.0)
        return s / n

    mol_vec = layers.Lambda(_masked_mean, name="mol_embedding")([atom_repr, in_atom_mask])

    ffn = layers.Dense(ffn_hidden, activation="relu", name="ffn_1")(mol_vec)
    ffn = layers.Dropout(dropout)(ffn)
    out = layers.Dense(n_outputs, name="prediction")(ffn)

    inputs = [in_f_atoms, in_f_bonds, in_a2b, in_a2b_mask,
              in_bond_src, in_bond_rev, in_atom_mask, in_bond_mask]

    train_model = Model(inputs=inputs, outputs=out, name="DMPNN")
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss="mse", metrics=["mae"])
    embedding_model = Model(inputs=inputs, outputs=mol_vec, name="DMPNN_embedding")
    return train_model, embedding_model


def graphs_to_inputs(graphs):
    """Stack a list of per-molecule graph dicts into model input arrays."""
    keys = ["f_atoms", "f_bonds", "a2b", "a2b_mask",
            "bond_src", "bond_rev", "atom_mask", "bond_mask"]
    return [np.stack([g[k] for g in graphs], axis=0) for k in keys]


# ============================================================================
# CLASSICAL FEATURISED ENSEMBLE  (Ridge / RandomForest / GaussianProcess)
#
# The honest baseline -- and, at these sample sizes (38 unique donors, 299
# unique acceptors, 182-460 usable Guzik rows per target), the expected WINNER
# for most targets. Featurisation is the same Morgan-fingerprint path used by
# the generation pipeline, with PCA compression before the GP so the kernel is
# not fitted in 2048 dimensions on a few hundred samples.
# ============================================================================


class ClassicalEnsembleRegressor:
    """Ridge + RandomForest + GaussianProcess with inverse-error blending.

    Blend weights are derived from INNER-fold validation error, never from the
    held-out fold being scored, so the reported outer-fold error stays honest.
    """

    def __init__(self, ridge_alpha=1.0, rf_n_estimators=300, rf_max_depth=None,
                 gp_length_scale=1.0, gp_alpha=1e-2, pca_components=32,
                 use_isotonic=True, random_state=42):
        self.ridge_alpha = float(ridge_alpha)
        self.rf_n_estimators = int(rf_n_estimators)
        self.rf_max_depth = rf_max_depth
        self.gp_length_scale = float(gp_length_scale)
        self.gp_alpha = float(gp_alpha)
        self.pca_components = int(pca_components)
        self.use_isotonic = bool(use_isotonic)
        self.random_state = int(random_state)

        self.scaler_x = None
        self.scaler_y = None
        self.pca = None
        self.ridge = None
        self.rf = None
        self.gp = None
        self.iso = None
        self.y_min = None
        self.y_max = None
        self.weights = {"ridge": 1.0, "rf": 0.0, "gp": 0.0}
        self.embedding_dim = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))

        self.scaler_x = StandardScaler().fit(X)
        Xs = self.scaler_x.transform(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        ys = self.scaler_y.transform(y.reshape(-1, 1)).ravel()

        # Ridge
        self.ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.ridge.fit(Xs, ys)

        # Isotonic calibration of Ridge (bounded to observed training range)
        if self.use_isotonic and n >= 5:
            try:
                raw = self.scaler_y.inverse_transform(
                    self.ridge.predict(Xs).reshape(-1, 1)).ravel()
                self.iso = IsotonicRegression(out_of_bounds="clip",
                                              y_min=self.y_min, y_max=self.y_max)
                self.iso.fit(raw, y)
            except Exception as e:
                logger.debug("Isotonic calibration skipped: %s" % e)
                self.iso = None

        # Random Forest
        try:
            self.rf = RandomForestRegressor(
                n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth,
                random_state=self.random_state, n_jobs=1)
            self.rf.fit(Xs, y)
        except Exception as e:
            logger.debug("RandomForest fit failed: %s" % e)
            self.rf = None

        # Gaussian Process on PCA-compressed features
        try:
            k = max(2, min(self.pca_components, n - 1, Xs.shape[1]))
            self.pca = PCA(n_components=k, random_state=self.random_state).fit(Xs)
            Xp = self.pca.transform(Xs)
            self.embedding_dim = k
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(length_scale=self.gp_length_scale, length_scale_bounds=(1e-2, 1e3))
                      + WhiteKernel(noise_level=self.gp_alpha, noise_level_bounds=(1e-8, 1e1)))
            # n_restarts_optimizer=0: kernel-hyperparameter restarts were the
            # single largest cost in the CV horse race (a measured ~6 min per
            # target at 300 samples). Optuna already searches length_scale and
            # alpha externally, so internal restarts duplicate that search at
            # much higher cost.
            self.gp = GaussianProcessRegressor(
                kernel=kernel, alpha=self.gp_alpha, normalize_y=True,
                n_restarts_optimizer=0, random_state=self.random_state)
            self.gp.fit(Xp, y)
        except Exception as e:
            logger.debug("GaussianProcess fit failed: %s" % e)
            self.gp = None
            if self.pca is None:
                try:
                    k = max(2, min(self.pca_components, max(2, n - 1), Xs.shape[1]))
                    self.pca = PCA(n_components=k, random_state=self.random_state).fit(Xs)
                    self.embedding_dim = k
                except Exception:
                    self.pca = None

        self._fit_blend_weights(Xs, y)
        return self

    def _fit_blend_weights(self, Xs, y):
        """Inner-fold inverse-MSE blend weights over the available submodels."""
        n = len(y)
        members = ["ridge"] + (["rf"] if self.rf is not None else []) + \
                  (["gp"] if self.gp is not None else [])
        if n < 6 or len(members) == 1:
            self.weights = {m: (1.0 if m == "ridge" else 0.0) for m in ["ridge", "rf", "gp"]}
            return
        try:
            kf = KFold(n_splits=min(5, n), shuffle=True, random_state=self.random_state)
            errs = {m: [] for m in members}
            for tr, va in kf.split(Xs):
                ys_tr = self.scaler_y.transform(y[tr].reshape(-1, 1)).ravel()
                r = Ridge(alpha=self.ridge_alpha, random_state=self.random_state).fit(Xs[tr], ys_tr)
                pr = self.scaler_y.inverse_transform(r.predict(Xs[va]).reshape(-1, 1)).ravel()
                errs["ridge"].append(mean_squared_error(y[va], pr))
                if "rf" in members:
                    rf = RandomForestRegressor(n_estimators=min(120, self.rf_n_estimators),
                                               max_depth=self.rf_max_depth,
                                               random_state=self.random_state, n_jobs=1)
                    rf.fit(Xs[tr], y[tr])
                    errs["rf"].append(mean_squared_error(y[va], rf.predict(Xs[va])))
                if "gp" in members and self.pca is not None:
                    Xp_tr = self.pca.transform(Xs[tr])
                    Xp_va = self.pca.transform(Xs[va])
                    kernel = (ConstantKernel(1.0) * RBF(length_scale=self.gp_length_scale)
                              + WhiteKernel(noise_level=self.gp_alpha))
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=self.gp_alpha,
                                                  normalize_y=True, n_restarts_optimizer=0,
                                                  random_state=self.random_state)
                    gp.fit(Xp_tr, y[tr])
                    errs["gp"].append(mean_squared_error(y[va], gp.predict(Xp_va)))
            inv = {}
            for m in members:
                e = float(np.mean(errs[m])) if errs[m] else np.inf
                inv[m] = 1.0 / max(e, 1e-9)
            tot = sum(inv.values())
            self.weights = {m: (inv.get(m, 0.0) / tot if tot > 0 else 0.0)
                            for m in ["ridge", "rf", "gp"]}
        except Exception as e:
            logger.debug("Blend-weight estimation failed (%s); Ridge-only." % e)
            self.weights = {"ridge": 1.0, "rf": 0.0, "gp": 0.0}

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_x.transform(X)
        preds = np.zeros(len(Xs), dtype=np.float64)
        wsum = 0.0

        w = self.weights.get("ridge", 0.0)
        if w > 0:
            raw = self.scaler_y.inverse_transform(
                self.ridge.predict(Xs).reshape(-1, 1)).ravel()
            if self.iso is not None:
                try:
                    raw = self.iso.predict(raw)
                except Exception:
                    pass
            preds += w * raw
            wsum += w

        w = self.weights.get("rf", 0.0)
        if w > 0 and self.rf is not None:
            preds += w * self.rf.predict(Xs)
            wsum += w

        w = self.weights.get("gp", 0.0)
        if w > 0 and self.gp is not None and self.pca is not None:
            preds += w * self.gp.predict(self.pca.transform(Xs))
            wsum += w

        if wsum <= 0:
            raw = self.scaler_y.inverse_transform(
                self.ridge.predict(Xs).reshape(-1, 1)).ravel()
            return raw
        return preds / wsum

    def embed(self, X):
        """Penultimate-style embedding for PFD: PCA-compressed standardised features."""
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_x.transform(X)
        if self.pca is not None:
            return self.pca.transform(Xs)
        return Xs


# ============================================================================
# GROUP-DISJOINT CROSS-VALIDATION
#
# The core of the honesty fix. Random K-fold leaks structural information when
# the same donor / acceptor / scaffold appears in both train and test. These
# splitters guarantee a group never straddles the split, which is what makes
# the reported error a defensible estimate of performance on genuinely NOVEL
# generated molecules.
# ============================================================================


def make_disjoint_folds(groups, n_splits=5, seed=42):
    """Yield (train_idx, test_idx) with no group shared across the split.

    Falls back to leave-one-group-out when there are fewer groups than folds
    (which is the situation for the 38-unique-donor targets).
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    n_groups = len(uniq)
    if n_groups < 2:
        return []
    effective = int(min(n_splits, n_groups))
    rng = np.random.RandomState(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    buckets = [shuffled[i::effective] for i in range(effective)]
    folds = []
    for b in buckets:
        test_mask = np.isin(groups, b)
        tr = np.where(~test_mask)[0]
        te = np.where(test_mask)[0]
        if len(tr) >= 3 and len(te) >= 1:
            folds.append((tr, te))
    return folds


def cv_evaluate_classical(X, y, groups, params, n_splits=5, seed=42):
    """Group-disjoint CV for the classical ensemble. Returns (r2, rmse, mae, n_eval)."""
    folds = make_disjoint_folds(groups, n_splits=n_splits, seed=seed)
    if not folds:
        return (np.nan, np.nan, np.nan, 0)
    yt_all, yp_all = [], []
    for tr, te in folds:
        try:
            m = ClassicalEnsembleRegressor(random_state=seed, **params).fit(X[tr], y[tr])
            p = m.predict(X[te])
            yt_all.append(y[te])
            yp_all.append(p)
        except Exception as e:
            logger.debug("classical CV fold failed: %s" % e)
            continue
    if not yt_all:
        return (np.nan, np.nan, np.nan, 0)
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    if len(yt) < 3 or np.std(yt) < 1e-12:
        return (np.nan, float(np.sqrt(mean_squared_error(yt, yp))),
                float(mean_absolute_error(yt, yp)), int(len(yt)))
    return (float(r2_score(yt, yp)),
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
            int(len(yt)))


def cv_evaluate_dmpnn(graphs, y, groups, params, n_splits=5, seed=42, max_epochs=140):
    """Group-disjoint CV for the D-MPNN. Returns (r2, rmse, mae, n_eval)."""
    if not TF_AVAILABLE:
        return (np.nan, np.nan, np.nan, 0)
    folds = make_disjoint_folds(groups, n_splits=n_splits, seed=seed)
    if not folds:
        return (np.nan, np.nan, np.nan, 0)
    yt_all, yp_all = [], []
    for fi, (tr, te) in enumerate(folds):
        try:
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed + fi)
            model, _ = build_dmpnn_model(
                hidden_dim=params.get("hidden_dim", 64),
                depth=params.get("depth", 3),
                dropout=params.get("dropout", 0.15),
                ffn_hidden=params.get("ffn_hidden", 64),
                learning_rate=params.get("learning_rate", 1e-3))
            Xtr = graphs_to_inputs([graphs[i] for i in tr])
            Xte = graphs_to_inputs([graphs[i] for i in te])
            ymu, ysd = float(np.mean(y[tr])), float(np.std(y[tr]) + 1e-9)
            ytr = (y[tr] - ymu) / ysd
            es = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=params.get("patience", 20),
                restore_best_weights=True, verbose=0)
            model.fit(Xtr, ytr, epochs=max_epochs,
                      batch_size=params.get("batch_size", 16),
                      verbose=0, callbacks=[es], shuffle=True)
            p = model.predict(Xte, verbose=0).ravel() * ysd + ymu
            yt_all.append(y[te])
            yp_all.append(p)
        except Exception as e:
            logger.debug("D-MPNN CV fold %d failed: %s" % (fi, e))
            continue
    if not yt_all:
        return (np.nan, np.nan, np.nan, 0)
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    if len(yt) < 3 or np.std(yt) < 1e-12:
        return (np.nan, float(np.sqrt(mean_squared_error(yt, yp))),
                float(mean_absolute_error(yt, yp)), int(len(yt)))
    return (float(r2_score(yt, yp)),
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
            int(len(yt)))


# ============================================================================
# OPTUNA-DRIVEN PER-TARGET MODEL HORSE RACE
#
# For every target we tune BOTH candidate families with Optuna under
# group-disjoint CV, then select the family with the better held-out R^2. The
# decision, the tuned hyperparameters and BOTH families' scores are recorded so
# the manuscript can state plainly which model won each target and by how much.
# ============================================================================


def optimize_classical(X, y, groups, n_trials=12, n_splits=5, seed=42, tag=""):
    default = {"ridge_alpha": 1.0, "rf_n_estimators": 300, "rf_max_depth": None,
               "gp_length_scale": 1.0, "gp_alpha": 1e-2, "pca_components": 32}
    if not OPTUNA_AVAILABLE or n_trials <= 1:
        r2, rmse, mae, n = cv_evaluate_classical(X, y, groups, default, n_splits, seed)
        return default, {"r2": r2, "rmse": rmse, "mae": mae, "n_eval": n}

    def objective(trial):
        params = {
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True),
            "rf_n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=100),
            "rf_max_depth": trial.suggest_categorical("rf_max_depth", [None, 6, 12, 24]),
            "gp_length_scale": trial.suggest_float("gp_length_scale", 1e-1, 1e2, log=True),
            "gp_alpha": trial.suggest_float("gp_alpha", 1e-6, 1e0, log=True),
            "pca_components": trial.suggest_int("pca_components", 8, 64, step=8),
        }
        r2, _, _, n = cv_evaluate_classical(X, y, groups, params, n_splits, seed)
        if not np.isfinite(r2) or n == 0:
            return -1e6
        return r2

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(default)
    best.update(study.best_params)
    r2, rmse, mae, n = cv_evaluate_classical(X, y, groups, best, n_splits, seed)
    logger.info("      [Optuna/classical%s] best CV R2=%.4f over %d trials" % (tag, r2, n_trials))
    return best, {"r2": r2, "rmse": rmse, "mae": mae, "n_eval": n}


def optimize_dmpnn(graphs, y, groups, n_trials=8, n_splits=5, seed=42, tag=""):
    default = {"hidden_dim": 64, "depth": 3, "dropout": 0.15, "ffn_hidden": 64,
               "learning_rate": 1e-3, "batch_size": 16, "patience": 20}
    if not TF_AVAILABLE:
        return default, {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "n_eval": 0}
    if not OPTUNA_AVAILABLE or n_trials <= 1:
        r2, rmse, mae, n = cv_evaluate_dmpnn(graphs, y, groups, default, n_splits, seed)
        return default, {"r2": r2, "rmse": rmse, "mae": mae, "n_eval": n}

    def objective(trial):
        params = {
            # Deliberately SMALL search space: heavy regularisation is the
            # correct prior when training a GNN on a few hundred molecules.
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64, 96]),
            "depth": trial.suggest_int("depth", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.05, 0.45),
            "ffn_hidden": trial.suggest_categorical("ffn_hidden", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "patience": 20,
        }
        r2, _, _, n = cv_evaluate_dmpnn(graphs, y, groups, params, n_splits, seed, max_epochs=110)
        if not np.isfinite(r2) or n == 0:
            return -1e6
        return r2

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(default)
    best.update(study.best_params)
    r2, rmse, mae, n = cv_evaluate_dmpnn(graphs, y, groups, best, n_splits, seed)
    logger.info("      [Optuna/D-MPNN%s] best CV R2=%.4f over %d trials" % (tag, r2, n_trials))
    return best, {"r2": r2, "rmse": rmse, "mae": mae, "n_eval": n}


# ============================================================================
# E_LOSS PHYSICS  (negative-value artefact handling)
#
# See the module docstring for the full scientific rationale. Summary:
#   E_loss = E_opt - Voc must satisfy  E_loss >= ~0.25 eV  because
#     (i)  Voc <= E_opt          (photon-energy ceiling), and
#     (ii) a non-zero radiative loss is unavoidable even in an ideal device
#          (detailed-balance / radiative limit).
# DMPNN.csv's E_loss column violates this for 46/300 rows because its V_OC_sim
# comes from the crude (|HOMO_D| - |LUMO_A|) - 0.3 estimate.
# ============================================================================

DETAILED_BALANCE_ELOSS_FLOOR_EV = 0.25
FRONTIER_ORBITAL_OFFSET_DEFAULT = 0.30


def analyse_eloss_column(df, eloss_col="E_loss", es_col="E_S", voc_sim_col="V_OC_sim",
                         voc_meas_col="Voc(V)", homo_d_col="D_HOMO (eV)",
                         lumo_a_col="LUMO (eV)"):
    """Diagnose and repair the E_loss column under all three documented options.

    Returns (df_with_flags, diagnostics_dict). Adds columns:
      eloss_raw                original value
      eloss_is_unphysical      True when raw < DETAILED_BALANCE_ELOSS_FLOOR_EV
      eloss_negative           True when raw < 0
      eloss_option1_exclude    raw, or NaN when unphysical      <-- PRIMARY
      voc_sim_refit            Voc recomputed with a data-refit offset
      eloss_option2_refit      E_S - voc_sim_refit
      eloss_option3_clip       max(raw, floor)
    """
    out = df.copy()
    diag = {}

    if eloss_col not in out.columns:
        logger.warning("E_loss column '%s' absent; E_loss analysis skipped." % eloss_col)
        diag["available"] = False
        return out, diag
    diag["available"] = True

    raw = pd.to_numeric(out[eloss_col], errors="coerce")
    out["eloss_raw"] = raw
    out["eloss_negative"] = raw < 0
    out["eloss_is_unphysical"] = raw < DETAILED_BALANCE_ELOSS_FLOOR_EV

    n_total = int(raw.notna().sum())
    n_neg = int((raw < 0).sum())
    n_unphys = int((raw < DETAILED_BALANCE_ELOSS_FLOOR_EV).sum())

    diag.update({
        "n_rows_with_eloss": n_total,
        "n_negative": n_neg,
        "pct_negative": 100.0 * n_neg / max(1, n_total),
        "n_below_db_floor": n_unphys,
        "pct_below_db_floor": 100.0 * n_unphys / max(1, n_total),
        "raw_min": float(raw.min()) if n_total else np.nan,
        "raw_max": float(raw.max()) if n_total else np.nan,
        "raw_mean": float(raw.mean()) if n_total else np.nan,
        "db_floor_ev": DETAILED_BALANCE_ELOSS_FLOOR_EV,
    })

    logger.info("-" * 80)
    logger.info("E_LOSS DIAGNOSTIC (DMPNN.csv)")
    logger.info("  rows with E_loss           : %d" % n_total)
    logger.info("  NEGATIVE E_loss            : %d (%.2f%%)  <-- unphysical" % (n_neg, diag["pct_negative"]))
    logger.info("  below detailed-balance floor %.2f eV : %d (%.2f%%)"
                % (DETAILED_BALANCE_ELOSS_FLOOR_EV, n_unphys, diag["pct_below_db_floor"]))
    logger.info("  raw range                  : [%.4f, %.4f], mean %.4f"
                % (diag["raw_min"], diag["raw_max"], diag["raw_mean"]))

    # OPTION 1 (PRIMARY): exclude known-bad rows. Invent nothing.
    out["eloss_option1_exclude"] = raw.where(~out["eloss_is_unphysical"], np.nan)
    diag["option1_n_retained"] = int(out["eloss_option1_exclude"].notna().sum())
    logger.info("  OPTION 1 (exclude, PRIMARY): retained %d rows" % diag["option1_n_retained"])

    # OPTION 2 (SUPPLEMENTARY): refit the empirical frontier-orbital offset
    # against MEASURED Voc rather than assuming a fixed 0.3 V.
    try:
        if all(c in out.columns for c in (homo_d_col, lumo_a_col, voc_meas_col, es_col)):
            homo_d = pd.to_numeric(out[homo_d_col], errors="coerce").abs()
            lumo_a = pd.to_numeric(out[lumo_a_col], errors="coerce").abs()
            voc_meas = pd.to_numeric(out[voc_meas_col], errors="coerce")
            es = pd.to_numeric(out[es_col], errors="coerce")
            gap = homo_d - lumo_a
            ok = gap.notna() & voc_meas.notna()
            if int(ok.sum()) >= 10:
                # measured Voc ~= gap - offset  =>  offset ~= mean(gap - Voc_meas)
                offset_refit = float(np.median((gap[ok] - voc_meas[ok]).values))
                voc_refit = (gap - offset_refit).clip(lower=0.0)
                out["voc_sim_refit"] = voc_refit
                out["eloss_option2_refit"] = es - voc_refit
                n_neg2 = int((out["eloss_option2_refit"] < 0).sum())
                n_unphys2 = int((out["eloss_option2_refit"] < DETAILED_BALANCE_ELOSS_FLOOR_EV).sum())
                diag.update({
                    "option2_offset_default": FRONTIER_ORBITAL_OFFSET_DEFAULT,
                    "option2_offset_refit": offset_refit,
                    "option2_n_negative": n_neg2,
                    "option2_n_below_db_floor": n_unphys2,
                    "option2_n_fit_rows": int(ok.sum()),
                })
                logger.info("  OPTION 2 (refit offset)    : default=%.3f V -> refit=%.4f V "
                            "(from %d rows with measured Voc)"
                            % (FRONTIER_ORBITAL_OFFSET_DEFAULT, offset_refit, int(ok.sum())))
                logger.info("                               negatives %d -> %d, below-floor %d -> %d"
                            % (n_neg, n_neg2, n_unphys, n_unphys2))
            else:
                logger.warning("  OPTION 2 skipped: too few rows with measured Voc (%d)" % int(ok.sum()))
        else:
            logger.warning("  OPTION 2 skipped: required columns missing.")
    except Exception as e:
        logger.warning("  OPTION 2 refit failed: %s" % e)

    # OPTION 3 (completeness only): clip at the detailed-balance floor.
    out["eloss_option3_clip"] = raw.clip(lower=DETAILED_BALANCE_ELOSS_FLOOR_EV)
    diag["option3_n_clipped"] = n_unphys
    logger.info("  OPTION 3 (clip @ %.2f eV)  : %d rows clipped (reported, NOT used for headline)"
                % (DETAILED_BALANCE_ELOSS_FLOOR_EV, n_unphys))
    logger.info("  ACTIVE POLICY              : %s" % ARGS.eloss_policy)
    logger.info("-" * 80)
    return out, diag


def active_eloss_series(df):
    """Return the E_loss series selected by the active --eloss-policy."""
    if ARGS.eloss_policy == "refit" and "eloss_option2_refit" in df.columns:
        return df["eloss_option2_refit"]
    if ARGS.eloss_policy == "clip" and "eloss_option3_clip" in df.columns:
        return df["eloss_option3_clip"]
    if "eloss_option1_exclude" in df.columns:
        return df["eloss_option1_exclude"]
    return pd.Series([np.nan] * len(df), index=df.index)


# ============================================================================
# STAGE A -- DATASET INGESTION AND CANONICAL OVERLAP VERIFICATION
# ============================================================================

DMPNN_ACCEPTOR_TARGETS = {
    "acc_HOMO":          ("HOMO (eV)", True),
    "acc_LUMO":          ("LUMO (eV)", True),
    "acc_HL_gap":        ("H-L Gap (eV)", False),
    "acc_Diff_L_Lp1":    ("Diff_L-L+1", False),
    "acc_lambda_nm":     ("λ(nm)", False),
    "acc_E_S":           ("E_S", False),
    "acc_osc_strength":  ("Osc. strength", False),
    "acc_E_b":           ("E_b", False),
    "acc_Q20":           ("Q20", False),
    "acc_EA":            ("A_EA", False),
    "acc_dipole_debye":  ("μ (Debye)", False),
}

DMPNN_DONOR_TARGETS = {
    "don_HOMO":          ("D_HOMO (eV)", True),
    "don_LUMO":          ("D_LUMO (eV)", True),
    "don_HL_gap":        ("D_H-L Gap (eV)", False),
    "don_Diff_L_Lp1":    ("D_Diff_L-L+1", False),
    "don_lambda_nm":     ("D_λ(nm)", False),
    "don_E_S":           ("D_E_S", False),
    "don_osc_strength":  ("D_Osc. strength", False),
    "don_E_b":           ("D_E_b", False),
    "don_IP":            ("D_IP", False),
    "don_dipole_debye":  ("D_μ (Debye)", False),
}

DMPNN_DEVICE_TARGETS = {
    "dev_Jsc": ("Jsc(mA/cm2)", False),
    "dev_Voc": ("Voc(V)", False),
    "dev_FF":  ("FF", False),
    "dev_PCE": ("PCE_max(%)", False),
}

GUZIK_DONOR_TARGETS = {
    "guz_donor_PDI":     "Donor PDI",
    "guz_donor_Mn_kDa":  "Donor Mn (kDa)",
    "guz_donor_Mw_kDa":  "Donor Mw (kDa)",
    "guz_Ehl_D":         "Ehl_D (eV)",
    "guz_Eg_D":          "Eg_D (eV)",
    "guz_HOMO_D":        "HOMO_D (eV)",
    "guz_LUMO_D":        "LUMO_D (eV)",
}

GUZIK_ACCEPTOR_TARGETS = {
    "guz_Ehl_A":   "Ehl_A (eV)",
    "guz_Eg_A":    "Eg_A (eV)",
    "guz_HOMO_A":  "HOMO_A (eV)",
    "guz_LUMO_A":  "LUMO_A (eV)",
}

GUZIK_BLEND_TARGETS = {
    "guz_hole_mobility_blend":     "hole mobility blend (cm^2 V^-1 s^-1)",
    "guz_electron_mobility_blend": "electron mobility blend (cm^2 V^-1 s^-1)",
    "guz_log_hole_mobility":       "log hole mobility blend (cm^2 V^-1 s^-1)",
    "guz_log_electron_mobility":   "log electron mobility blend (cm^2 V^-1 s^-1)",
}


def stage_a_load_data():
    """Load + canonicalise both training CSVs; verify overlap; diagnose E_loss."""
    logger.info("=" * 80)
    logger.info("STAGE A -- DATASET INGESTION AND CANONICAL OVERLAP VERIFICATION")
    logger.info("=" * 80)

    dmpnn_path = os.path.join(BASE_WORKDIR, ARGS.dmpnn_csv)
    guzik_path = os.path.join(BASE_WORKDIR, ARGS.guzik_csv)

    if not os.path.exists(dmpnn_path):
        raise FileNotFoundError("DMPNN CSV not found: %s" % dmpnn_path)
    dmpnn = pd.read_csv(dmpnn_path)
    logger.info("Loaded DMPNN.csv: shape=%s" % (dmpnn.shape,))

    guzik = None
    if os.path.exists(guzik_path):
        guzik = pd.read_csv(guzik_path)
        logger.info("Loaded Guzik_data.csv: shape=%s" % (guzik.shape,))
    else:
        logger.warning("Guzik CSV not found at %s -- Guzik targets will be skipped." % guzik_path)

    # --- canonicalisation --------------------------------------------------
    logger.info("Canonicalising DMPNN SMILES via the same RDKit path used in generation...")
    dmpnn["donor_canonical"] = [canonical_smiles(s) for s in dmpnn.get("donor(SMILES)", pd.Series(dtype=str))]
    dmpnn["acceptor_canonical"] = [canonical_smiles(s) for s in dmpnn.get("acceptor(SMILES)", pd.Series(dtype=str))]
    n_d_raw = dmpnn["donor(SMILES)"].nunique() if "donor(SMILES)" in dmpnn.columns else 0
    n_a_raw = dmpnn["acceptor(SMILES)"].nunique() if "acceptor(SMILES)" in dmpnn.columns else 0
    n_d_can = dmpnn["donor_canonical"].dropna().nunique()
    n_a_can = dmpnn["acceptor_canonical"].dropna().nunique()
    logger.info("  DMPNN donors   : %d unique raw -> %d unique canonical" % (n_d_raw, n_d_can))
    logger.info("  DMPNN acceptors: %d unique raw -> %d unique canonical" % (n_a_raw, n_a_can))
    n_d_unparse = int(dmpnn["donor_canonical"].isna().sum())
    n_a_unparse = int(dmpnn["acceptor_canonical"].isna().sum())
    if n_d_unparse or n_a_unparse:
        logger.warning("  RDKit could not canonicalise %d donor / %d acceptor rows "
                       "(SELFIES/token-hash fallback will be used for featurisation)."
                       % (n_d_unparse, n_a_unparse))

    overlap_diag = {
        "dmpnn_donor_unique_raw": n_d_raw, "dmpnn_donor_unique_canonical": n_d_can,
        "dmpnn_acceptor_unique_raw": n_a_raw, "dmpnn_acceptor_unique_canonical": n_a_can,
        "dmpnn_donor_unparseable": n_d_unparse, "dmpnn_acceptor_unparseable": n_a_unparse,
    }

    if guzik is not None:
        logger.info("Canonicalising Guzik SMILES...")
        guzik["donor_canonical"] = [canonical_smiles(s) for s in guzik.get("Donor SMILES", pd.Series(dtype=str))]
        guzik["acceptor_canonical"] = [canonical_smiles(s) for s in guzik.get("Acceptor SMILES", pd.Series(dtype=str))]
        gd_raw = guzik["Donor SMILES"].nunique() if "Donor SMILES" in guzik.columns else 0
        ga_raw = guzik["Acceptor SMILES"].nunique() if "Acceptor SMILES" in guzik.columns else 0
        gd_can = guzik["donor_canonical"].dropna().nunique()
        ga_can = guzik["acceptor_canonical"].dropna().nunique()
        logger.info("  Guzik donors   : %d unique raw -> %d unique canonical" % (gd_raw, gd_can))
        logger.info("  Guzik acceptors: %d unique raw -> %d unique canonical" % (ga_raw, ga_can))

        # THE VERIFICATION: canonical (not exact-string) cross-dataset overlap.
        d_overlap = set(dmpnn["donor_canonical"].dropna()) & set(guzik["donor_canonical"].dropna())
        a_overlap = set(dmpnn["acceptor_canonical"].dropna()) & set(guzik["acceptor_canonical"].dropna())
        logger.info("  CANONICAL donor overlap  DMPNN <-> Guzik : %d molecules" % len(d_overlap))
        logger.info("  CANONICAL acceptor overlap DMPNN <-> Guzik: %d molecules" % len(a_overlap))
        if len(d_overlap) == 0 and len(a_overlap) == 0:
            logger.info("  => Datasets are DISJOINT even after canonicalisation. They are modelled "
                        "SEPARATELY (no joint training, no shared normalisation).")
        else:
            logger.info("  => Some molecules are shared. Overlapping molecules are kept in the SAME "
                        "CV group so no molecule straddles a train/test split.")
        overlap_diag.update({
            "guzik_donor_unique_raw": gd_raw, "guzik_donor_unique_canonical": gd_can,
            "guzik_acceptor_unique_raw": ga_raw, "guzik_acceptor_unique_canonical": ga_can,
            "canonical_donor_overlap": len(d_overlap),
            "canonical_acceptor_overlap": len(a_overlap),
        })

        # per-target availability (labels are never imputed)
        for key, col in list(GUZIK_DONOR_TARGETS.items()) + \
                        list(GUZIK_ACCEPTOR_TARGETS.items()) + \
                        list(GUZIK_BLEND_TARGETS.items()):
            if col in guzik.columns:
                n_ok = int(pd.to_numeric(guzik[col], errors="coerce").notna().sum())
                logger.info("    Guzik target %-30s : %d/%d labelled rows usable"
                            % (key, n_ok, len(guzik)))
                overlap_diag["guzik_%s_n_labelled" % key] = n_ok

    # --- E_loss diagnostic + repair ---------------------------------------
    dmpnn, eloss_diag = analyse_eloss_column(dmpnn)
    overlap_diag.update({"eloss_%s" % k: v for k, v in eloss_diag.items()})

    if IS_MASTER:
        try:
            pd.DataFrame([overlap_diag]).T.rename(columns={0: "value"}).to_csv(
                os.path.join(OUT_ROOT, "stage_a_dataset_diagnostics.csv"))
            logger.info("Wrote %s" % os.path.join(OUT_ROOT, "stage_a_dataset_diagnostics.csv"))
        except Exception as e:
            logger.warning("Could not write Stage-A diagnostics: %s" % e)

    return dmpnn, guzik, overlap_diag


# ============================================================================
# STAGE B -- MODEL TRAINING  (rank 0 trains; every rank loads the artefacts)
#
# For each target:
#   1. Build (X_fingerprint, graph, y, group) arrays from labelled rows only.
#      Labels are NEVER imputed -- a row missing that target is dropped for
#      that target and only that target.
#   2. Optuna-tune the classical ensemble under group-disjoint CV.
#   3. Optuna-tune the D-MPNN under the same folds (skipped if TF missing or
#      too few molecules for a GNN to be meaningful).
#   4. Pick the winner by held-out R^2. Record BOTH scores.
#   5. Refit the winner on all labelled rows and persist it.
# ============================================================================

MIN_ROWS_FOR_TARGET = 12
MIN_UNIQUE_MOLS_FOR_DMPNN = 60   # below this a GNN cannot generalise; skip it


class TargetModel:
    """A fitted model for one target plus its provenance and honest CV scores."""

    def __init__(self, key, kind, model, cv_scores, params, n_train,
                 n_unique_groups, group_kind, source_dataset, target_column,
                 needs_abs=False, embedding_available=False):
        self.key = key
        self.kind = kind                      # 'classical' | 'dmpnn'
        self.model = model
        self.cv_scores = cv_scores
        self.params = params
        self.n_train = n_train
        self.n_unique_groups = n_unique_groups
        self.group_kind = group_kind
        self.source_dataset = source_dataset
        self.target_column = target_column
        self.needs_abs = needs_abs
        self.embedding_available = embedding_available
        self._dmpnn_embed = None
        self._y_mu = 0.0
        self._y_sd = 1.0

    def predict_smiles(self, smiles_list, batch_size=256):
        """Predict the target for a list of SMILES (NaN where featurisation fails)."""
        n = len(smiles_list)
        out = np.full(n, np.nan, dtype=np.float64)
        if self.kind == "classical":
            feats, idx = [], []
            for i, s in enumerate(smiles_list):
                f = featurize_smiles(s)
                if f is not None:
                    feats.append(f)
                    idx.append(i)
            if feats:
                try:
                    p = self.model.predict(np.asarray(feats, dtype=np.float32))
                    out[np.asarray(idx)] = p
                except Exception as e:
                    logger.debug("classical predict failed for %s: %s" % (self.key, e))
            return out
        # D-MPNN
        graphs, idx = [], []
        for i, s in enumerate(smiles_list):
            g = mol_to_dmpnn_graph(s)
            if g is not None:
                graphs.append(g)
                idx.append(i)
        if graphs:
            try:
                for st in range(0, len(graphs), batch_size):
                    chunk = graphs[st:st + batch_size]
                    X = graphs_to_inputs(chunk)
                    p = self.model.predict(X, verbose=0).ravel() * self._y_sd + self._y_mu
                    out[np.asarray(idx[st:st + batch_size])] = p
            except Exception as e:
                logger.debug("dmpnn predict failed for %s: %s" % (self.key, e))
        return out

    def embed_smiles(self, smiles_list, batch_size=256):
        """Penultimate-layer / PCA embedding for the Property-Embedding Frechet Distance."""
        if self.kind == "classical":
            feats, idx = [], []
            for i, s in enumerate(smiles_list):
                f = featurize_smiles(s)
                if f is not None:
                    feats.append(f)
                    idx.append(i)
            if not feats:
                return None, []
            try:
                return self.model.embed(np.asarray(feats, dtype=np.float32)), idx
            except Exception:
                return None, []
        if self._dmpnn_embed is None:
            return None, []
        graphs, idx = [], []
        for i, s in enumerate(smiles_list):
            g = mol_to_dmpnn_graph(s)
            if g is not None:
                graphs.append(g)
                idx.append(i)
        if not graphs:
            return None, []
        try:
            chunks = []
            for st in range(0, len(graphs), batch_size):
                X = graphs_to_inputs(graphs[st:st + batch_size])
                chunks.append(self._dmpnn_embed.predict(X, verbose=0))
            return np.vstack(chunks), idx
        except Exception as e:
            logger.debug("dmpnn embed failed for %s: %s" % (self.key, e))
            return None, []


def _dmpnn_candidate_possible():
    """True when the D-MPNN can be entered into the race at all."""
    return bool(TF_AVAILABLE and not ARGS.skip_dmpnn_candidate)


def _prepare_target_arrays(df, smiles_col, target_col, needs_abs=False,
                           group_mode="scaffold", build_graphs=None):
    """Featurise labelled rows for one target and build disjoint CV groups.

    group_mode:
      'scaffold' -> Bemis-Murcko scaffold groups (structural generalisation)
      'molecule' -> molecule-identity groups (donor-/acceptor-disjoint)
    """
    if build_graphs is None:
        # Skip graph construction entirely when no D-MPNN will be trained; it is
        # pure wasted work in the classical-only path.
        build_graphs = _dmpnn_candidate_possible()
    if smiles_col not in df.columns or target_col not in df.columns:
        return None
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    if needs_abs:
        y_raw = y_raw.abs()
    mask = y_raw.notna() & df[smiles_col].notna()
    sub = df.loc[mask]
    if len(sub) < MIN_ROWS_FOR_TARGET:
        return None

    feats, graphs, ys, smis = [], [], [], []
    for smi, yv in zip(sub[smiles_col].astype(str).values, y_raw.loc[mask].values):
        f = featurize_smiles(smi)
        if f is None:
            continue
        g = mol_to_dmpnn_graph(smi) if build_graphs else None
        feats.append(f)
        graphs.append(g)
        ys.append(float(yv))
        smis.append(smi)

    if len(ys) < MIN_ROWS_FOR_TARGET:
        return None

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float64)
    if group_mode == "molecule":
        groups = np.asarray([canonical_smiles(s) or s for s in smis])
    else:
        groups = np.asarray([murcko_scaffold(s) for s in smis])

    graph_ok = [i for i, g in enumerate(graphs) if g is not None]
    return {
        "X": X, "y": y, "groups": groups, "smiles": smis,
        "graphs": graphs, "graph_ok": graph_ok,
        "n": len(y), "n_groups": int(len(np.unique(groups))),
        "n_unique_mols": int(len(set(smis))),
    }


def _train_one_target(key, data, target_col, source_dataset, needs_abs, group_kind):
    """Run the horse race for one target and return (TargetModel, race_record)."""
    n, ngrp = data["n"], data["n_groups"]
    logger.info("    target %-28s | n=%-5d unique_mols=%-5d groups=%-5d (%s-disjoint CV)"
                % (key, n, data["n_unique_mols"], ngrp, group_kind))

    if ngrp < 2:
        logger.warning("      only %d CV group(s); cannot cross-validate -> SKIPPED" % ngrp)
        return None, {
            "target": key, "source": source_dataset, "column": target_col,
            "n_train": n, "n_unique_molecules": data["n_unique_mols"],
            "n_cv_groups": ngrp, "group_kind": group_kind,
            "winner": "none_insufficient_groups",
            "classical_r2": np.nan, "classical_rmse": np.nan, "classical_mae": np.nan,
            "dmpnn_r2": np.nan, "dmpnn_rmse": np.nan, "dmpnn_mae": np.nan,
            "dmpnn_attempted": False, "notes": "fewer than 2 disjoint groups",
        }

    folds = ARGS.cv_folds
    # ---- candidate 1: classical featurised ensemble ----
    c_params, c_scores = optimize_classical(
        data["X"], data["y"], data["groups"],
        n_trials=ARGS.optuna_trials, n_splits=folds, seed=ARGS.seed, tag="/" + key)
    logger.info("      classical ensemble : R2=%7.4f RMSE=%9.4f MAE=%9.4f (n_eval=%d)"
                % (c_scores["r2"], c_scores["rmse"], c_scores["mae"], c_scores["n_eval"]))

    # ---- candidate 2: real D-MPNN ----
    d_params, d_scores = ({}, {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "n_eval": 0})
    dmpnn_attempted = False
    dmpnn_skip_reason = ""
    graph_frac = len(data["graph_ok"]) / max(1, data["n"])
    if ARGS.skip_dmpnn_candidate:
        dmpnn_skip_reason = "disabled by --skip-dmpnn-candidate"
    elif not TF_AVAILABLE:
        dmpnn_skip_reason = "TensorFlow unavailable"
    elif data["n_unique_mols"] < MIN_UNIQUE_MOLS_FOR_DMPNN:
        dmpnn_skip_reason = ("only %d unique molecules (< %d); a GNN cannot generalise at this "
                             "sample size, so it is not entered into the race"
                             % (data["n_unique_mols"], MIN_UNIQUE_MOLS_FOR_DMPNN))
    elif graph_frac < 0.8:
        dmpnn_skip_reason = "only %.0f%% of molecules graph-representable" % (100 * graph_frac)

    if dmpnn_skip_reason:
        logger.info("      D-MPNN             : SKIPPED (%s)" % dmpnn_skip_reason)
    else:
        dmpnn_attempted = True
        gi = data["graph_ok"]
        graphs_ok = [data["graphs"][i] for i in gi]
        y_ok = data["y"][gi]
        groups_ok = data["groups"][gi]
        d_trials = max(2, ARGS.optuna_trials // 2)   # GNN trials are far costlier
        d_params, d_scores = optimize_dmpnn(
            graphs_ok, y_ok, groups_ok, n_trials=d_trials,
            n_splits=folds, seed=ARGS.seed, tag="/" + key)
        logger.info("      D-MPNN             : R2=%7.4f RMSE=%9.4f MAE=%9.4f (n_eval=%d)"
                    % (d_scores["r2"], d_scores["rmse"], d_scores["mae"], d_scores["n_eval"]))

    # ---- decide the winner on held-out R^2 ----
    c_r2 = c_scores["r2"] if np.isfinite(c_scores["r2"]) else -np.inf
    d_r2 = d_scores["r2"] if np.isfinite(d_scores["r2"]) else -np.inf
    winner = "dmpnn" if (d_r2 > c_r2 and dmpnn_attempted) else "classical"
    logger.info("      WINNER             : %s (classical R2=%.4f vs D-MPNN R2=%.4f)"
                % (winner.upper(), c_r2 if np.isfinite(c_r2) else float("nan"),
                   d_r2 if np.isfinite(d_r2) else float("nan")))

    race = {
        "target": key, "source": source_dataset, "column": target_col,
        "n_train": n, "n_unique_molecules": data["n_unique_mols"],
        "n_cv_groups": ngrp, "group_kind": group_kind,
        "winner": winner,
        "classical_r2": c_scores["r2"], "classical_rmse": c_scores["rmse"],
        "classical_mae": c_scores["mae"], "classical_n_eval": c_scores["n_eval"],
        "dmpnn_r2": d_scores["r2"], "dmpnn_rmse": d_scores["rmse"],
        "dmpnn_mae": d_scores["mae"], "dmpnn_n_eval": d_scores["n_eval"],
        "dmpnn_attempted": dmpnn_attempted,
        "dmpnn_skip_reason": dmpnn_skip_reason,
        "classical_params": json.dumps(c_params, default=str),
        "dmpnn_params": json.dumps(d_params, default=str),
        "notes": "",
    }

    # ---- refit the winner on ALL labelled rows ----
    try:
        if winner == "classical":
            final = ClassicalEnsembleRegressor(random_state=ARGS.seed, **c_params)
            final.fit(data["X"], data["y"])
            tm = TargetModel(key, "classical", final, c_scores, c_params, n, ngrp,
                             group_kind, source_dataset, target_col, needs_abs,
                             embedding_available=True)
        else:
            tf.keras.backend.clear_session()
            tf.random.set_seed(ARGS.seed)
            model, emb = build_dmpnn_model(
                hidden_dim=d_params.get("hidden_dim", 64),
                depth=d_params.get("depth", 3),
                dropout=d_params.get("dropout", 0.15),
                ffn_hidden=d_params.get("ffn_hidden", 64),
                learning_rate=d_params.get("learning_rate", 1e-3))
            gi = data["graph_ok"]
            X = graphs_to_inputs([data["graphs"][i] for i in gi])
            yv = data["y"][gi]
            mu, sd = float(np.mean(yv)), float(np.std(yv) + 1e-9)
            es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=25,
                                                  restore_best_weights=True, verbose=0)
            model.fit(X, (yv - mu) / sd, epochs=200,
                      batch_size=d_params.get("batch_size", 16),
                      verbose=0, callbacks=[es], shuffle=True)
            tm = TargetModel(key, "dmpnn", model, d_scores, d_params, n, ngrp,
                             group_kind, source_dataset, target_col, needs_abs,
                             embedding_available=True)
            tm._dmpnn_embed = emb
            tm._y_mu, tm._y_sd = mu, sd
        return tm, race
    except Exception as e:
        logger.error("      final refit FAILED for %s: %s" % (key, e))
        race["notes"] = "final refit failed: %s" % e
        return None, race


def stage_b_train_models(dmpnn_df, guzik_df):
    """Train every target model; return (models_dict, race_dataframe)."""
    logger.info("=" * 80)
    logger.info("STAGE B -- MODEL TRAINING (per-target Optuna + group-disjoint CV horse race)")
    logger.info("=" * 80)
    logger.info("  D-MPNN candidate entered only when a target has >= %d unique molecules."
                % MIN_UNIQUE_MOLS_FOR_DMPNN)
    logger.info("  Labels are NEVER imputed: each target trains only on its own labelled rows.")

    models = {}
    races = []
    t_start = time.time()

    # ---------------- ACCEPTOR descriptor targets (DMPNN.csv, ~299 unique) ----
    logger.info("-" * 80)
    logger.info("  [B1] ACCEPTOR molecular property targets (DMPNN.csv)")
    for key, (col, needs_abs) in DMPNN_ACCEPTOR_TARGETS.items():
        data = _prepare_target_arrays(dmpnn_df, "acceptor(SMILES)", col,
                                      needs_abs=needs_abs, group_mode="scaffold")
        if data is None:
            logger.warning("    target %-28s | insufficient labelled data -> SKIPPED" % key)
            continue
        tm, race = _train_one_target(key, data, col, "DMPNN.csv", needs_abs, "scaffold")
        races.append(race)
        if tm is not None:
            models[key] = tm

    # ---------------- DONOR descriptor targets (DMPNN.csv, only 38 unique) ----
    logger.info("-" * 80)
    logger.info("  [B2] DONOR molecular property targets (DMPNN.csv)")
    logger.info("       NOTE: only ~38 unique donor structures exist. Donor-molecule-disjoint")
    logger.info("       CV is used and results are PROVISIONAL/high-variance by construction.")
    for key, (col, needs_abs) in DMPNN_DONOR_TARGETS.items():
        data = _prepare_target_arrays(dmpnn_df, "donor(SMILES)", col,
                                      needs_abs=needs_abs, group_mode="molecule")
        if data is None:
            logger.warning("    target %-28s | insufficient labelled data -> SKIPPED" % key)
            continue
        tm, race = _train_one_target(key, data, col, "DMPNN.csv", needs_abs, "donor-molecule")
        race["notes"] = ("provisional: only %d unique donor structures" % data["n_unique_mols"])
        races.append(race)
        if tm is not None:
            models[key] = tm

    # ---------------- DEVICE targets (pair-level, DMPNN.csv) ------------------
    logger.info("-" * 80)
    logger.info("  [B3] DEVICE targets (pair-level: donor+acceptor concatenated features)")
    for key, (col, needs_abs) in DMPNN_DEVICE_TARGETS.items():
        tm, race = _train_pair_target(dmpnn_df, key, col, needs_abs)
        if race is not None:
            races.append(race)
        if tm is not None:
            models[key] = tm

    # ---------------- E_LOSS (pair-level, physically gated) -------------------
    logger.info("-" * 80)
    logger.info("  [B4] E_LOSS target (pair-level, physically gated -- policy '%s')" % ARGS.eloss_policy)
    dmpnn_eloss = dmpnn_df.copy()
    dmpnn_eloss["__eloss_active__"] = active_eloss_series(dmpnn_eloss)
    n_use = int(dmpnn_eloss["__eloss_active__"].notna().sum())
    logger.info("       usable E_loss rows after policy '%s': %d / %d"
                % (ARGS.eloss_policy, n_use, len(dmpnn_eloss)))
    tm, race = _train_pair_target(dmpnn_eloss, "dev_E_loss", "__eloss_active__", False)
    if race is not None:
        race["notes"] = ("E_loss policy=%s; %d/%d rows retained (unphysical rows dropped: "
                         "Voc<=Eopt ceiling + detailed-balance floor %.2f eV)"
                         % (ARGS.eloss_policy, n_use, len(dmpnn_eloss),
                            DETAILED_BALANCE_ELOSS_FLOOR_EV))
        races.append(race)
    if tm is not None:
        models["dev_E_loss"] = tm

    # ---------------- GUZIK targets -----------------------------------------
    if guzik_df is not None:
        logger.info("-" * 80)
        logger.info("  [B5] GUZIK donor-side targets (per-target labelled subsets)")
        for key, col in GUZIK_DONOR_TARGETS.items():
            data = _prepare_target_arrays(guzik_df, "Donor SMILES", col,
                                          needs_abs=False, group_mode="molecule")
            if data is None:
                logger.warning("    target %-28s | insufficient labelled data -> SKIPPED" % key)
                continue
            tm, race = _train_one_target(key, data, col, "Guzik_data.csv", False, "donor-molecule")
            races.append(race)
            if tm is not None:
                models[key] = tm

        logger.info("-" * 80)
        logger.info("  [B6] GUZIK acceptor-side targets")
        for key, col in GUZIK_ACCEPTOR_TARGETS.items():
            data = _prepare_target_arrays(guzik_df, "Acceptor SMILES", col,
                                          needs_abs=False, group_mode="scaffold")
            if data is None:
                logger.warning("    target %-28s | insufficient labelled data -> SKIPPED" % key)
                continue
            tm, race = _train_one_target(key, data, col, "Guzik_data.csv", False, "scaffold")
            races.append(race)
            if tm is not None:
                models[key] = tm

        logger.info("-" * 80)
        logger.info("  [B7] GUZIK blend-level targets (mobilities: depend on BOTH components)")
        for key, col in GUZIK_BLEND_TARGETS.items():
            tm, race = _train_pair_target(
                guzik_df, key, col, False,
                donor_col="Donor SMILES", acceptor_col="Acceptor SMILES",
                source="Guzik_data.csv")
            if race is not None:
                races.append(race)
            if tm is not None:
                models[key] = tm

    elapsed = time.time() - t_start
    logger.info("-" * 80)
    logger.info("STAGE B COMPLETE: %d target models trained in %.1f s" % (len(models), elapsed))

    race_df = pd.DataFrame(races)
    n_class = int((race_df["winner"] == "classical").sum()) if not race_df.empty else 0
    n_dmpnn = int((race_df["winner"] == "dmpnn").sum()) if not race_df.empty else 0
    logger.info("  HORSE-RACE OUTCOME: classical ensemble won %d targets, D-MPNN won %d targets"
                % (n_class, n_dmpnn))
    logger.info("  (Reported transparently per target -- no model family was assumed superior.)")
    return models, race_df


def _train_pair_target(df, key, target_col, needs_abs,
                       donor_col="donor(SMILES)", acceptor_col="acceptor(SMILES)",
                       source="DMPNN.csv"):
    """Train a PAIR-level target on concatenated donor+acceptor fingerprints.

    CV groups are donor-identity based, so a donor never appears in both train
    and test -- the honest test of generalisation to new donor chemistry.
    Pair-level targets are classical-only: a pair is not a single molecular
    graph, so the D-MPNN candidate does not apply without a two-tower model,
    which these sample sizes cannot support.
    """
    if donor_col not in df.columns or acceptor_col not in df.columns or target_col not in df.columns:
        logger.warning("    target %-28s | required columns missing -> SKIPPED" % key)
        return None, None
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    if needs_abs:
        y_raw = y_raw.abs()
    mask = y_raw.notna() & df[donor_col].notna() & df[acceptor_col].notna()
    sub = df.loc[mask]
    if len(sub) < MIN_ROWS_FOR_TARGET:
        logger.warning("    target %-28s | only %d labelled rows -> SKIPPED" % (key, len(sub)))
        return None, None

    feats, ys, dsm = [], [], []
    for d, a, yv in zip(sub[donor_col].astype(str).values,
                        sub[acceptor_col].astype(str).values,
                        y_raw.loc[mask].values):
        fd = featurize_smiles(d)
        fa = featurize_smiles(a)
        if fd is None or fa is None:
            continue
        feats.append(np.concatenate([fd, fa]))
        ys.append(float(yv))
        dsm.append(d)
    if len(ys) < MIN_ROWS_FOR_TARGET:
        logger.warning("    target %-28s | only %d featurisable rows -> SKIPPED" % (key, len(ys)))
        return None, None

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float64)
    groups = np.asarray([canonical_smiles(s) or s for s in dsm])
    ngrp = int(len(np.unique(groups)))
    logger.info("    target %-28s | n=%-5d donor_groups=%-5d (donor-disjoint CV, classical-only)"
                % (key, len(y), ngrp))

    if ngrp < 2:
        logger.warning("      only %d donor group(s); cannot cross-validate -> SKIPPED" % ngrp)
        return None, {
            "target": key, "source": source, "column": target_col,
            "n_train": len(y), "n_unique_molecules": ngrp, "n_cv_groups": ngrp,
            "group_kind": "donor-molecule(pair)", "winner": "none_insufficient_groups",
            "classical_r2": np.nan, "classical_rmse": np.nan, "classical_mae": np.nan,
            "dmpnn_r2": np.nan, "dmpnn_rmse": np.nan, "dmpnn_mae": np.nan,
            "dmpnn_attempted": False,
            "dmpnn_skip_reason": "pair-level target (no single molecular graph)",
            "notes": "fewer than 2 disjoint donor groups",
        }

    params, scores = optimize_classical(
        X, y, groups, n_trials=ARGS.optuna_trials, n_splits=ARGS.cv_folds,
        seed=ARGS.seed, tag="/" + key)
    logger.info("      classical ensemble : R2=%7.4f RMSE=%9.4f MAE=%9.4f (n_eval=%d)"
                % (scores["r2"], scores["rmse"], scores["mae"], scores["n_eval"]))

    race = {
        "target": key, "source": source, "column": target_col,
        "n_train": len(y), "n_unique_molecules": ngrp, "n_cv_groups": ngrp,
        "group_kind": "donor-molecule(pair)", "winner": "classical",
        "classical_r2": scores["r2"], "classical_rmse": scores["rmse"],
        "classical_mae": scores["mae"], "classical_n_eval": scores["n_eval"],
        "dmpnn_r2": np.nan, "dmpnn_rmse": np.nan, "dmpnn_mae": np.nan, "dmpnn_n_eval": 0,
        "dmpnn_attempted": False,
        "dmpnn_skip_reason": "pair-level target (not a single molecular graph)",
        "classical_params": json.dumps(params, default=str),
        "dmpnn_params": "{}", "notes": "",
    }
    try:
        final = ClassicalEnsembleRegressor(random_state=ARGS.seed, **params).fit(X, y)
        tm = TargetModel(key, "classical", final, scores, params, len(y), ngrp,
                         "donor-molecule(pair)", source, target_col, needs_abs,
                         embedding_available=False)
        tm.is_pair_level = True
        return tm, race
    except Exception as e:
        logger.error("      final refit FAILED for %s: %s" % (key, e))
        race["notes"] = "final refit failed: %s" % e
        return None, race


def save_models(models, race_df):
    """Persist trained models + the horse-race report (rank 0 only)."""
    safe_makedirs(POSTGEN_MODELS_DIR)
    saved, failed = 0, 0
    for key, tm in models.items():
        try:
            if tm.kind == "classical":
                with open(os.path.join(POSTGEN_MODELS_DIR, "%s.pkl" % key), "wb") as fh:
                    pickle.dump({
                        "kind": "classical", "model": tm.model, "cv": tm.cv_scores,
                        "params": tm.params, "n_train": tm.n_train,
                        "n_groups": tm.n_unique_groups, "group_kind": tm.group_kind,
                        "source": tm.source_dataset, "column": tm.target_column,
                        "needs_abs": tm.needs_abs,
                        "is_pair_level": getattr(tm, "is_pair_level", False),
                    }, fh, protocol=4)
            else:
                tm.model.save(os.path.join(POSTGEN_MODELS_DIR, "%s_dmpnn.keras" % key))
                with open(os.path.join(POSTGEN_MODELS_DIR, "%s.meta.pkl" % key), "wb") as fh:
                    pickle.dump({
                        "kind": "dmpnn", "cv": tm.cv_scores, "params": tm.params,
                        "n_train": tm.n_train, "n_groups": tm.n_unique_groups,
                        "group_kind": tm.group_kind, "source": tm.source_dataset,
                        "column": tm.target_column, "needs_abs": tm.needs_abs,
                        "y_mu": tm._y_mu, "y_sd": tm._y_sd,
                        "is_pair_level": False,
                    }, fh, protocol=4)
            saved += 1
        except Exception as e:
            failed += 1
            logger.warning("Could not persist model %s: %s" % (key, e))
    logger.info("Persisted %d models to %s (%d failed)" % (saved, POSTGEN_MODELS_DIR, failed))

    try:
        p = os.path.join(OUT_ROOT, "model_selection_horse_race.csv")
        race_df.to_csv(p, index=False)
        logger.info("Wrote %s" % p)
        p2 = os.path.join(OUT_ROOT, "property_model_cv_report.csv")
        cols = ["target", "source", "column", "n_train", "n_unique_molecules",
                "n_cv_groups", "group_kind", "winner",
                "classical_r2", "classical_rmse", "classical_mae",
                "dmpnn_r2", "dmpnn_rmse", "dmpnn_mae",
                "dmpnn_attempted", "dmpnn_skip_reason", "notes"]
        race_df[[c for c in cols if c in race_df.columns]].to_csv(p2, index=False)
        logger.info("Wrote %s" % p2)
    except Exception as e:
        logger.warning("Could not write horse-race report: %s" % e)


# ============================================================================
# GENERATIVE-MODEL METRICS
#
# Every metric below is a DETERMINISTIC function of a fixed molecule set. That
# is a deliberate design constraint: a reported metric must be reproducible by
# a reviewer given the same molecules. This is precisely why reinforcement
# learning is NOT used anywhere in this section -- an RL agent's output depends
# on seeds, reward shaping and training length, so "RL-based diversity" would
# not return the same number twice on the same input. RL is used only to DRIVE
# GENERATION (Stage D), never to define a metric.
#
# NAMING CORRECTIONS applied here:
#   "Mean Inception Score" (MIS)  ->  Internal Diversity (IntDiv)
#        The old quantity was 1 - mean(pairwise Tanimoto). That is not an
#        inception score (no classifier, no softmax, no KL divergence); it is
#        the MOSES-standard internal-diversity statistic. Renamed accordingly.
#   "Frechet Distance" (MFD)      ->  FP-PCA-FD  (fingerprint-PCA Frechet dist.)
#        The old quantity projected Morgan fingerprints onto 2 PCA components
#        and applied the Frechet formula there. It is NOT Frechet ChemNet
#        Distance. Relabelled, retained for backward comparability, and
#        reported alongside two better-founded distances (PFD and real FCD).
# ============================================================================


def metric_validity(raw_smiles_list):
    """MOSES Validity: fraction of proposed strings that parse to a valid molecule."""
    n = len(raw_smiles_list)
    if n == 0:
        return 0.0, 0, 0
    valid = 0
    for s in raw_smiles_list:
        if s and is_valid_smiles(str(s)):
            valid += 1
    return float(valid) / n, valid, n


def metric_uniqueness(valid_smiles_list):
    """MOSES Uniqueness: fraction of VALID molecules that are canonically distinct."""
    n = len(valid_smiles_list)
    if n == 0:
        return 0.0, 0, 0
    can = set()
    for s in valid_smiles_list:
        c = canonical_smiles(s)
        if c:
            can.add(c)
    return float(len(can)) / n, len(can), n


def metric_novelty(generated_smiles_list, training_canonical_set):
    """MOSES Novelty: fraction of unique generated molecules absent from training."""
    gen_can = set()
    for s in generated_smiles_list:
        c = canonical_smiles(s)
        if c:
            gen_can.add(c)
    if not gen_can:
        return 0.0, 0, 0
    novel = gen_can - set(training_canonical_set)
    return float(len(novel)) / len(gen_can), len(novel), len(gen_can)


def metric_internal_diversity(mols, sample_cap=5000, seed=42, p=1):
    """MOSES Internal Diversity (IntDiv_p).

    IntDiv = 1 - mean(pairwise Tanimoto similarity) over the molecule set.
    This is the correctly-named replacement for the old "MIS". Deterministic:
    the subsample uses a FIXED seed and the seed is reported.
    Returns (intdiv_mean, pairwise_sim_std, n_used, n_pairs).
    """
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0, 0.0, len(mols), 0
    if len(mols) > sample_cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(mols), sample_cap, replace=False)
        mols = [mols[i] for i in idx]
    fps = [f for f in (morgan_fp(m) for m in mols) if f is not None]
    if len(fps) < 2:
        return 0.0, 0.0, len(fps), 0
    sims = []
    for i in range(len(fps)):
        row = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        if row:
            sims.extend(row)
    if not sims:
        return 0.0, 0.0, len(fps), 0
    sims = np.asarray(sims, dtype=np.float64)
    if p == 1:
        mean_sim = float(np.mean(sims))
    else:
        mean_sim = float(np.power(np.mean(np.power(sims, p)), 1.0 / p))
    return float(1.0 - mean_sim), float(np.std(sims)), len(fps), int(len(sims))


def metric_nearest_neighbour_similarity(gen_mols, train_fps, sample_cap=20000, seed=42):
    """Max-Tanimoto-to-training per generated molecule (the 'similarity' axis).

    Retained unchanged in definition so the new numbers remain directly
    comparable with the previously reported donor/acceptor similarity figures.
    """
    gen_mols = [m for m in gen_mols if m is not None]
    if not gen_mols or not train_fps:
        return np.asarray([])
    if len(gen_mols) > sample_cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(gen_mols), sample_cap, replace=False)
        gen_mols = [gen_mols[i] for i in idx]
    sims = []
    for m in gen_mols:
        fp = morgan_fp(m)
        if fp is None:
            continue
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return np.asarray(sims, dtype=np.float64)


def _frechet_from_features(feat_real, feat_gen):
    """Frechet distance between two Gaussians fitted to the supplied features."""
    feat_real = np.asarray(feat_real, dtype=np.float64)
    feat_gen = np.asarray(feat_gen, dtype=np.float64)
    if feat_real.ndim != 2 or feat_gen.ndim != 2:
        return float("nan")
    if len(feat_real) < 2 or len(feat_gen) < 2:
        return float("nan")
    mu_r, mu_g = feat_real.mean(axis=0), feat_gen.mean(axis=0)
    sig_r = np.cov(feat_real, rowvar=False)
    sig_g = np.cov(feat_gen, rowvar=False)
    sig_r = np.atleast_2d(sig_r)
    sig_g = np.atleast_2d(sig_g)
    diff = mu_r - mu_g
    try:
        covmean = sqrtm(sig_r.dot(sig_g))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fd = diff.dot(diff) + np.trace(sig_r + sig_g - 2.0 * covmean)
        return float(fd)
    except Exception as e:
        logger.debug("Frechet computation failed: %s" % e)
        return float("nan")


def metric_fp_pca_frechet(real_mols, gen_mols, n_components=2, sample_cap=20000, seed=42):
    """FP-PCA-FD: the ORIGINAL fingerprint-PCA Frechet distance, honestly named.

    Kept algorithmically IDENTICAL to the historical calculate_mfd() so previously
    reported values stay reproducible and comparable. This is NOT Frechet ChemNet
    Distance -- it is a Frechet-form distance in a 2-component PCA projection of
    Morgan-fingerprint space, which is a much weaker representation than a
    learned embedding. Reported next to PFD and real FCD, never alone.
    """
    real_mols = [m for m in real_mols if m is not None]
    gen_mols = [m for m in gen_mols if m is not None]
    if not real_mols or not gen_mols:
        return float("inf")
    rng = np.random.RandomState(seed)
    if len(gen_mols) > sample_cap:
        gen_mols = [gen_mols[i] for i in rng.choice(len(gen_mols), sample_cap, replace=False)]
    rf = [np.asarray(f) for f in (morgan_fp(m) for m in real_mols) if f is not None]
    gf = [np.asarray(f) for f in (morgan_fp(m) for m in gen_mols) if f is not None]
    if not rf or not gf:
        return float("inf")
    R, G = np.array(rf), np.array(gf)
    allfp = np.vstack([R, G])
    try:
        k = max(2, min(n_components, allfp.shape[0] - 1, allfp.shape[1]))
        pca = PCA(n_components=k, random_state=seed)
        proj = pca.fit_transform(allfp)
        return _frechet_from_features(proj[:len(R)], proj[len(R):])
    except Exception as e:
        logger.debug("FP-PCA-FD failed: %s" % e)
        return float("nan")


def metric_property_embedding_frechet(train_smiles, gen_smiles, embed_model,
                                      sample_cap=20000, seed=42):
    """PFD -- Property-Embedding Frechet Distance (PRIMARY domain-appropriate distance).

    Frechet distance computed in the penultimate-layer activation space of a
    property model trained on THIS domain (OPV donors/acceptors with device and
    optoelectronic labels). Unlike ChemNet (trained on ChEMBL bioactivity), this
    embedding is anchored to the properties that actually matter for solar cells,
    so a small PFD means "the generated set occupies the same property-relevant
    region as the training set", which is the statement we actually want to make.
    """
    if embed_model is None:
        return float("nan"), 0, 0
    rng = np.random.RandomState(seed)
    gs = list(gen_smiles)
    if len(gs) > sample_cap:
        gs = [gs[i] for i in rng.choice(len(gs), sample_cap, replace=False)]
    try:
        er, _ = embed_model.embed_smiles(list(train_smiles))
        eg, _ = embed_model.embed_smiles(gs)
        if er is None or eg is None:
            return float("nan"), 0, 0
        return _frechet_from_features(er, eg), len(er), len(eg)
    except Exception as e:
        logger.debug("PFD failed: %s" % e)
        return float("nan"), 0, 0


def metric_real_fcd(train_smiles, gen_smiles, sample_cap=10000, seed=42):
    """Real Frechet ChemNet Distance via the `fcd` package.

    CAVEAT (state this in the manuscript): ChemNet is trained on ChEMBL
    bioactivity data, i.e. drug-like chemical space. OPV donors/acceptors
    (fused-thiophene polymers, cyano-indanone acceptors) sit outside that
    domain, so FCD here measures distance in a drug-likeness embedding, not an
    OPV-relevant one. Reported because it is the metric the wider
    generative-chemistry community recognises -- NOT as the primary evidence.
    """
    if not FCD_AVAILABLE:
        return float("nan"), 0, 0
    rng = np.random.RandomState(seed)
    gs = [str(s) for s in gen_smiles if s]
    ts = [str(s) for s in train_smiles if s]

    # ------------------------------------------------------------------
    # HARD CONSTRAINT OF ChemNet: its one-hot encoder allocates a fixed
    # (350, n_char) array, so any SMILES longer than 350 characters writes
    # past the end and raises
    #   IndexError: index 350 is out of bounds for axis 0 with size 350
    # inside fcd/utils.py::get_one_hot. That is exactly what made the pooled
    # ACCEPTOR FCD come back NaN in earlier runs: Y6/ITIC-family acceptors with
    # long solubilising side chains routinely exceed 350 characters, and a
    # single over-length string aborted the whole batch.
    #
    # Fix: drop over-length SMILES BEFORE calling get_fcd and report exactly
    # how many were dropped, so the number that is published is a real FCD over
    # a stated, reproducible subset rather than a silent NaN. Truncating the
    # strings instead would corrupt the molecule, so exclusion is the honest
    # option -- and the exclusion count is returned for the report.
    # ------------------------------------------------------------------
    _MAX_CHEMNET_SMILES_LEN = 350
    n_gs_raw, n_ts_raw = len(gs), len(ts)
    gs = [s for s in gs if len(s) <= _MAX_CHEMNET_SMILES_LEN]
    ts = [s for s in ts if len(s) <= _MAX_CHEMNET_SMILES_LEN]
    n_gs_dropped, n_ts_dropped = n_gs_raw - len(gs), n_ts_raw - len(ts)
    if n_gs_dropped or n_ts_dropped:
        logger.info("    FCD: excluded %d/%d generated and %d/%d training SMILES longer "
                    "than ChemNet's %d-character encoder limit (over-length strings would "
                    "raise IndexError and void the whole computation)."
                    % (n_gs_dropped, n_gs_raw, n_ts_dropped, n_ts_raw,
                       _MAX_CHEMNET_SMILES_LEN))

    if len(gs) > sample_cap:
        gs = [gs[i] for i in rng.choice(len(gs), sample_cap, replace=False)]
    if len(ts) > sample_cap:
        ts = [ts[i] for i in rng.choice(len(ts), sample_cap, replace=False)]
    if len(gs) < 2 or len(ts) < 2:
        logger.warning("    FCD: fewer than 2 usable SMILES on one side after the "
                       "%d-character filter (gen=%d, train=%d). Reported as NaN."
                       % (_MAX_CHEMNET_SMILES_LEN, len(gs), len(ts)))
        return float("nan"), len(ts), len(gs)
    try:
        val = float(fcd_module.get_fcd(ts, gs))
        return val, len(ts), len(gs)
    except Exception as e:
        logger.warning("Real FCD computation failed (%s). Reported as NaN." % e)
        return float("nan"), len(ts), len(gs)


DESCRIPTOR_FUNCS = {
    "MW":       lambda m: Descriptors.ExactMolWt(m),
    "LogP":     lambda m: Descriptors.MolLogP(m),
    "TPSA":     lambda m: Descriptors.TPSA(m),
    "HBA":      lambda m: Descriptors.NumHAcceptors(m),
    "HBD":      lambda m: Descriptors.NumHDonors(m),
    "RotBonds": lambda m: Descriptors.NumRotatableBonds(m),
    "AromRings": lambda m: Descriptors.NumAromaticRings(m),
    "HeavyAtoms": lambda m: float(m.GetNumHeavyAtoms()),
    "FracCSP3": lambda m: Descriptors.FractionCSP3(m),
    "RingCount": lambda m: float(m.GetRingInfo().NumRings()),
}


def descriptor_matrix(mols, keys=None):
    """Descriptor matrix (n_mols, n_desc); rows with any failure are dropped."""
    keys = keys or list(DESCRIPTOR_FUNCS.keys())
    rows = []
    for m in mols:
        if m is None:
            continue
        try:
            rows.append([float(DESCRIPTOR_FUNCS[k](m)) for k in keys])
        except Exception:
            continue
    if not rows:
        return np.zeros((0, len(keys))), keys
    return np.asarray(rows, dtype=np.float64), keys


def metric_property_match_binary(gen_mols, train_mols, keys=None):
    """LEGACY 6-condition binary property-match score (retained for comparability).

    KNOWN LIMITATIONS, stated plainly: it counts how many descriptors fall inside
    the training min-max box, so (a) it can only take discrete values separated
    by 1/n_descriptors -- which is what produced the sharp spikes in the old
    histograms -- (b) it ignores descriptor correlation entirely, and (c) a
    rectangular min-max box is a crude stand-in for the real training manifold.
    The Wasserstein and Mahalanobis metrics below fix (a)-(c); this one is kept
    only so the new report can be cross-checked against the previous numbers.
    """
    keys = keys or ["MW", "LogP", "TPSA", "HBA", "HBD", "RotBonds"]
    tmat, keys = descriptor_matrix(train_mols, keys)
    if tmat.shape[0] == 0:
        return np.asarray([]), keys
    lo, hi = tmat.min(axis=0), tmat.max(axis=0)
    scores = []
    for m in gen_mols:
        if m is None:
            continue
        try:
            vals = np.asarray([float(DESCRIPTOR_FUNCS[k](m)) for k in keys])
            scores.append(float(np.mean((vals >= lo) & (vals <= hi))))
        except Exception:
            continue
    return np.asarray(scores, dtype=np.float64), keys


def metric_descriptor_wasserstein(gen_mols, train_mols, keys=None):
    """Per-descriptor 1-Wasserstein distance between generated and training sets.

    A continuous, distribution-aware replacement for the binary in-range count.
    Reported per descriptor (so a reader can see WHICH property drifted) plus a
    normalised mean. Normalisation is by the training standard deviation so the
    numbers are dimensionless and comparable across descriptors.
    """
    keys = keys or list(DESCRIPTOR_FUNCS.keys())
    gmat, keys = descriptor_matrix(gen_mols, keys)
    tmat, _ = descriptor_matrix(train_mols, keys)
    out = {}
    if gmat.shape[0] == 0 or tmat.shape[0] == 0:
        return out
    for j, k in enumerate(keys):
        try:
            w = float(wasserstein_distance(tmat[:, j], gmat[:, j]))
            sd = float(np.std(tmat[:, j]))
            out["wasserstein_%s" % k] = w
            out["wasserstein_norm_%s" % k] = w / sd if sd > 1e-12 else np.nan
        except Exception:
            out["wasserstein_%s" % k] = np.nan
            out["wasserstein_norm_%s" % k] = np.nan
    norms = [v for kk, v in out.items() if kk.startswith("wasserstein_norm_") and np.isfinite(v)]
    out["wasserstein_norm_mean"] = float(np.mean(norms)) if norms else np.nan
    return out


def metric_mahalanobis_to_training(gen_mols, train_mols, keys=None, ridge=1e-6):
    """Mahalanobis distance of each generated molecule to the training manifold.

    Unlike the min-max box, this RESPECTS descriptor covariance: a molecule can
    sit inside every individual descriptor range yet still be far from the
    training manifold if its combination of descriptors never co-occurs in the
    training data. Returns (distances, keys, summary_dict).
    """
    keys = keys or list(DESCRIPTOR_FUNCS.keys())
    tmat, keys = descriptor_matrix(train_mols, keys)
    gmat, _ = descriptor_matrix(gen_mols, keys)
    if tmat.shape[0] < len(keys) + 2 or gmat.shape[0] == 0:
        return np.asarray([]), keys, {}
    mu = tmat.mean(axis=0)
    cov = np.cov(tmat, rowvar=False) + ridge * np.eye(len(keys))
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        return np.asarray([]), keys, {}
    d = []
    for row in gmat:
        try:
            d.append(float(mahalanobis(row, mu, inv)))
        except Exception:
            continue
    d = np.asarray(d, dtype=np.float64)
    # Reference: Mahalanobis distances of the TRAINING molecules themselves, so
    # "how far is far" has an in-domain yardstick rather than an arbitrary cutoff.
    dt = []
    for row in tmat:
        try:
            dt.append(float(mahalanobis(row, mu, inv)))
        except Exception:
            continue
    dt = np.asarray(dt, dtype=np.float64)
    summary = {
        "mahalanobis_gen_mean": float(np.mean(d)) if d.size else np.nan,
        "mahalanobis_gen_median": float(np.median(d)) if d.size else np.nan,
        "mahalanobis_gen_p95": float(np.percentile(d, 95)) if d.size else np.nan,
        "mahalanobis_train_mean": float(np.mean(dt)) if dt.size else np.nan,
        "mahalanobis_train_p95": float(np.percentile(dt, 95)) if dt.size else np.nan,
        "mahalanobis_frac_gen_beyond_train_p95": (
            float(np.mean(d > np.percentile(dt, 95))) if d.size and dt.size else np.nan),
        "n_descriptors": len(keys),
    }
    return d, keys, summary


def metric_scaffold_diversity(mols):
    """Unique Bemis-Murcko scaffolds / molecule count (structural-novelty axis)."""
    scafs = []
    for m in mols:
        if m is None:
            continue
        try:
            s = MurckoScaffold.GetScaffoldForMol(m)
            scafs.append(Chem.MolToSmiles(s, canonical=True) if s is not None else "NONE")
        except Exception:
            continue
    if not scafs:
        return 0.0, 0, 0
    return float(len(set(scafs))) / len(scafs), len(set(scafs)), len(scafs)


def compute_metric_block(label, gen_smiles, train_smiles, embed_model=None,
                         intdiv_cap=5000, seed=42):
    """Compute the full deterministic metric block for one molecule set.

    Returns a flat dict ready to be written as one CSV row. Every sampling cap
    and seed used is INCLUDED IN THE OUTPUT so the numbers are reproducible.
    """
    logger.info("  Computing metric block: %s (n_generated=%d, n_training=%d)"
                % (label, len(gen_smiles), len(train_smiles)))
    t0 = time.time()

    rec = {"set": label, "n_proposed": len(gen_smiles),
           "n_training_reference": len(train_smiles),
           "intdiv_sample_cap": intdiv_cap, "random_seed": seed}

    validity, n_valid, n_prop = metric_validity(gen_smiles)
    rec.update({"validity": validity, "n_valid": n_valid})

    valid_smiles = [s for s in gen_smiles if s and is_valid_smiles(str(s))]
    uniq, n_uniq, _ = metric_uniqueness(valid_smiles)
    rec.update({"uniqueness": uniq, "n_unique_canonical": n_uniq})

    train_can = canonical_smiles_set(train_smiles)
    nov, n_novel, n_uniq_for_nov = metric_novelty(valid_smiles, train_can)
    rec.update({"novelty": nov, "n_novel": n_novel,
                "n_unique_for_novelty": n_uniq_for_nov,
                "n_training_canonical": len(train_can)})

    gen_mols = [m for m in (Chem.MolFromSmiles(s) for s in valid_smiles) if m is not None]
    train_mols = [m for m in (Chem.MolFromSmiles(s) for s in train_smiles) if m is not None]

    intdiv, sim_std, n_used, n_pairs = metric_internal_diversity(
        gen_mols, sample_cap=intdiv_cap, seed=seed)
    rec.update({"internal_diversity": intdiv,
                "internal_diversity_pairwise_sim_std": sim_std,
                "intdiv_n_molecules_used": n_used, "intdiv_n_pairs": n_pairs,
                "mean_pairwise_tanimoto": (1.0 - intdiv) if intdiv else np.nan})

    scaf_div, n_scaf, n_scaf_mols = metric_scaffold_diversity(gen_mols)
    rec.update({"scaffold_diversity": scaf_div, "n_unique_scaffolds": n_scaf,
                "n_molecules_scaffolded": n_scaf_mols})

    train_fps = fingerprints_from_smiles(train_smiles)
    nn_sims = metric_nearest_neighbour_similarity(gen_mols, train_fps, seed=seed)
    if nn_sims.size:
        rec.update({
            "nn_similarity_mean": float(np.mean(nn_sims)),
            "nn_similarity_std": float(np.std(nn_sims)),
            "nn_similarity_min": float(np.min(nn_sims)),
            "nn_similarity_max": float(np.max(nn_sims)),
            "nn_similarity_frac_below_0p5": float(np.mean(nn_sims < 0.5)),
            "nn_similarity_n": int(nn_sims.size),
        })

    rec["fp_pca_frechet_distance"] = metric_fp_pca_frechet(train_mols, gen_mols, seed=seed)
    pfd, n_pr, n_pg = metric_property_embedding_frechet(
        train_smiles, valid_smiles, embed_model, seed=seed)
    rec.update({"property_embedding_frechet_distance": pfd,
                "pfd_n_train_embedded": n_pr, "pfd_n_gen_embedded": n_pg,
                "pfd_embedding_source": (embed_model.key if embed_model is not None else "unavailable"),
                "pfd_embedding_model_kind": (embed_model.kind if embed_model is not None else "unavailable")})
    fcd_val, n_fr, n_fg = metric_real_fcd(train_smiles, valid_smiles, seed=seed)
    rec.update({"frechet_chemnet_distance": fcd_val,
                "fcd_n_train": n_fr, "fcd_n_gen": n_fg,
                "fcd_available": FCD_AVAILABLE})

    pm, pm_keys = metric_property_match_binary(gen_mols, train_mols)
    if pm.size:
        rec.update({
            "legacy_property_match_mean": float(np.mean(pm)),
            "legacy_property_match_std": float(np.std(pm)),
            "legacy_property_match_n_descriptors": len(pm_keys),
            "legacy_property_match_frac_strong": float(np.mean(pm >= 0.8)),
        })

    rec.update(metric_descriptor_wasserstein(gen_mols, train_mols))
    _, _, mh = metric_mahalanobis_to_training(gen_mols, train_mols)
    rec.update(mh)

    if SASCORER_AVAILABLE:
        sas = [compute_sascore(m) for m in gen_mols[:min(len(gen_mols), 20000)]]
        sas = [s for s in sas if s is not None]
        if sas:
            sas = np.asarray(sas)
            rec.update({
                "sascore_mean": float(np.mean(sas)), "sascore_std": float(np.std(sas)),
                "sascore_min": float(np.min(sas)), "sascore_max": float(np.max(sas)),
                "sascore_frac_le_6": float(np.mean(sas <= 6.0)),
                "sascore_n": int(sas.size),
            })

    rec["metric_block_seconds"] = round(time.time() - t0, 2)
    logger.info("    validity=%.4f uniqueness=%.4f novelty=%.4f IntDiv=%.4f "
                "FP-PCA-FD=%.3f PFD=%.3f FCD=%s (%.1fs)"
                % (rec.get("validity", np.nan), rec.get("uniqueness", np.nan),
                   rec.get("novelty", np.nan), rec.get("internal_diversity", np.nan),
                   rec.get("fp_pca_frechet_distance", np.nan),
                   rec.get("property_embedding_frechet_distance", np.nan),
                   ("%.3f" % fcd_val) if np.isfinite(fcd_val) else "NaN",
                   rec["metric_block_seconds"]))
    return rec


# ============================================================================
# RETROSYNTHESIS AUDIT  (tiered, graceful degradation)
#
# The publishable synthesizability claim is gated HERE, not on the heuristics.
# Backends in order of scientific strength:
#
#   TIER R1  AiZynthFinder MCTS route search  -- the real thing. Requires a
#            stock file + trained expansion policy in models/aizynthfinder/
#            (multi-GB public download; absent by default). Only ever applied
#            to a small top-N shortlist because MCTS costs ~seconds/molecule
#            and the pool has ~107k unique molecules / ~12M pairs.
#
#   TIER R2  RAscore -- an XGB/NN classifier DISTILLED from AiZynthFinder
#            outcomes, so it approximates "can a route be found?" in
#            milliseconds. DOCUMENTED CAVEAT: RAscore was trained on 200k
#            ChEMBL bioactive compounds and its own authors state it does not
#            transfer to exotic chemical space. OPV donors/acceptors (fused
#            thiophenes, cyano-indanones) are outside its training domain, so
#            values are indicative only -- exactly the same class of caveat as
#            ChemNet/FCD.
#
#   TIER R3  SAScore (Ertl & Schuffenhauer fragment-frequency heuristic) --
#            always available, no domain restriction, no learned classifier,
#            but purely structural: it says "this looks complex/rare", never
#            "here is a route".
#
# Whichever tier is available is recorded per molecule so the manuscript can
# state precisely which evidence backs each synthesizability statement.
# ============================================================================


class RetrosynthesisAuditor:
    """Tiered synthesizability auditor with explicit provenance per score."""

    def __init__(self, aizynth_config=AIZYNTH_CONFIG_PATH, tango=None):
        self.backend = "sascore"
        self.aizynth = None
        self.rascore = None
        self.tango = tango

        logger.info("=" * 80)
        logger.info("INITIALISING RETROSYNTHESIS AUDITOR")

        if AIZYNTHFINDER_AVAILABLE and os.path.exists(aizynth_config):
            try:
                self.aizynth = AiZynthFinder(configfile=aizynth_config)
                try:
                    self.aizynth.stock.select(self.aizynth.stock.items[0])
                    self.aizynth.expansion_policy.select(self.aizynth.expansion_policy.items[0])
                    if getattr(self.aizynth, "filter_policy", None) and self.aizynth.filter_policy.items:
                        self.aizynth.filter_policy.select(self.aizynth.filter_policy.items[0])
                except Exception as sel_e:
                    logger.warning("  AiZynthFinder policy/stock selection issue: %s" % sel_e)
                self.backend = "aizynthfinder"
                logger.info("  TIER R1 ACTIVE: AiZynthFinder MCTS route search (config=%s)" % aizynth_config)
            except Exception as e:
                logger.warning("  AiZynthFinder init failed (%s); falling back." % e)
                self.aizynth = None

        if self.aizynth is None and RASCORE_AVAILABLE:
            try:
                try:
                    from RAscore import RAscore_XGB
                    self.rascore = RAscore_XGB.RAScorerXGB()
                except Exception:
                    from RAscore import RAscore_NN
                    self.rascore = RAscore_NN.RAScorerNN()
                self.backend = "rascore"
                logger.info("  TIER R2 ACTIVE: RAscore (distilled AiZynthFinder classifier)")
                logger.warning("  RAscore DOMAIN CAVEAT: trained on 200k ChEMBL bioactives; the "
                               "authors state it does not transfer to exotic chemistry. OPV "
                               "scaffolds are out of domain -- treat values as indicative only.")
            except Exception as e:
                logger.warning("  RAscore init failed (%s); falling back to SAScore." % e)
                self.rascore = None

        if self.aizynth is None and self.rascore is None:
            self.backend = "sascore" if SASCORER_AVAILABLE else "unavailable"
            if self.backend == "sascore":
                logger.info("  TIER R3 ACTIVE: SAScore only (structural heuristic, no route search)")
            else:
                logger.warning("  NO synthesizability backend available; audit columns will be NaN.")

        logger.info("  Selected backend: %s" % self.backend.upper())
        logger.info("=" * 80)

    def audit_molecule(self, smiles, time_limit=60):
        """Return a provenance-tagged audit record for one molecule."""
        rec = {
            "smiles": smiles, "audit_backend": self.backend,
            "route_found": np.nan, "n_route_steps": np.nan,
            "route_score": np.nan, "rascore": np.nan,
            "sascore": np.nan, "sascore_reward": np.nan,
            "tango_inspired": np.nan, "audit_error": "",
        }
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            rec["audit_error"] = "unparseable_smiles"
            return rec

        sa = compute_sascore(mol)
        rec["sascore"] = sa if sa is not None else np.nan
        rec["sascore_reward"] = sascore_to_reward(sa)
        if self.tango is not None:
            try:
                rec["tango_inspired"] = float(self.tango.score(mol))
            except Exception:
                pass

        if self.backend == "aizynthfinder" and self.aizynth is not None:
            try:
                self.aizynth.target_smiles = Chem.MolToSmiles(mol)
                self.aizynth.tree_search(show_progress=False)
                self.aizynth.build_routes()
                stats = self.aizynth.extract_statistics()
                solved = bool(stats.get("is_solved", False))
                rec["route_found"] = 1.0 if solved else 0.0
                rec["n_route_steps"] = float(stats.get("number_of_steps", np.nan))
                rec["route_score"] = float(stats.get("top_score", np.nan))
            except Exception as e:
                rec["audit_error"] = "aizynth_failed:%s" % type(e).__name__
                logger.debug("AiZynthFinder failed on %s: %s" % (str(smiles)[:60], e))
        elif self.backend == "rascore" and self.rascore is not None:
            try:
                v = float(self.rascore.predict(Chem.MolToSmiles(mol)))
                rec["rascore"] = v
                rec["route_found"] = 1.0 if v >= 0.5 else 0.0
                rec["route_score"] = v
            except Exception as e:
                rec["audit_error"] = "rascore_failed:%s" % type(e).__name__
        return rec


# ============================================================================
# STAGE C -- RANK-SHARDED SCORING OF THE GENERATED POOL
# ============================================================================


def find_pairs_csv():
    """Locate generated_molecules_with_predictions.csv across standard locations."""
    if ARGS.pairs_csv and os.path.exists(ARGS.pairs_csv):
        return os.path.abspath(ARGS.pairs_csv)
    candidates = [
        os.path.join(BASE_WORKDIR, "plot_data", "generated_molecules_with_predictions.csv"),
        os.path.join(BASE_WORKDIR, "results", "generated_molecules_with_predictions.csv"),
        os.path.join(BASE_WORKDIR, "results", "results", "generated_molecules_with_predictions.csv"),
        os.path.join(BASE_WORKDIR, "generated_molecules_with_predictions.csv"),
    ]
    candidates += sorted(glob.glob(os.path.join(
        BASE_WORKDIR, "runs", "*", "merged", "results", "generated_molecules_with_predictions.csv")))
    candidates += sorted(glob.glob(os.path.join(
        BASE_WORKDIR, "runs", "*", "merged", "plot_data", "generated_molecules_with_predictions.csv")))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_pairs_csv(path, max_rows=0):
    """Load the pair pool, reading only the columns actually needed."""
    logger.info("Loading generated pair pool: %s" % path)
    want = ["donor_smiles", "acceptor_smiles", "PCE", "Jsc", "FF", "Voc",
            "donor_tango", "acceptor_tango", "pair_tango", "tier",
            "favorable_homo_lumo_alignment",
            "donor_HOMO_donor", "donor_LUMO_donor", "donor_Eg_donor",
            "acceptor_HOMO_acceptor", "acceptor_LUMO_acceptor", "acceptor_Eg_acceptor",
            "source", "donor_source", "acceptor_source", "approach"]
    try:
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in want if c in header.columns]
        if "donor_smiles" not in usecols or "acceptor_smiles" not in usecols:
            logger.warning("Pool is missing donor_smiles/acceptor_smiles; loading all columns.")
            usecols = None
    except Exception as e:
        logger.warning("Could not read pool header (%s); loading all columns." % e)
        usecols = None

    kw = {"usecols": usecols} if usecols else {}
    if max_rows and max_rows > 0:
        kw["nrows"] = max_rows
        logger.info("  --max-pairs active: reading at most %d rows" % max_rows)
    df = pd.read_csv(path, low_memory=False, **kw)
    logger.info("  Loaded pool: shape=%s columns=%d" % (df.shape, df.shape[1]))
    return df


def extract_unique_molecules(pairs_df):
    """Unique donor / acceptor SMILES present in the pool."""
    donors, acceptors = [], []
    if "donor_smiles" in pairs_df.columns:
        donors = sorted(set(pairs_df["donor_smiles"].dropna().astype(str)))
    if "acceptor_smiles" in pairs_df.columns:
        acceptors = sorted(set(pairs_df["acceptor_smiles"].dropna().astype(str)))
    logger.info("Unique molecules in pool: %d donors, %d acceptors" % (len(donors), len(acceptors)))
    return donors, acceptors


def predict_molecule_properties(models, smiles_list, side):
    """Apply every single-molecule model for one side to a SMILES list."""
    if not smiles_list:
        return pd.DataFrame()
    prefix = "don_" if side == "donor" else "acc_"
    guz_prefix = "guz_"
    out = {"smiles": smiles_list}
    applicable = []
    for key, tm in models.items():
        if getattr(tm, "is_pair_level", False):
            continue
        if key.startswith(prefix):
            applicable.append(key)
        elif key.startswith(guz_prefix):
            if side == "donor" and ("_D" in key or "donor" in key):
                applicable.append(key)
            elif side == "acceptor" and ("_A" in key or "acceptor" in key):
                applicable.append(key)
    logger.info("  Predicting %d %s-side properties for %d molecules: %s"
                % (len(applicable), side, len(smiles_list), ", ".join(sorted(applicable))))
    for key in sorted(applicable):
        t0 = time.time()
        try:
            out[key] = models[key].predict_smiles(smiles_list)
            n_ok = int(np.isfinite(out[key]).sum())
            logger.info("    %-30s -> %d/%d predicted (%.1fs, model=%s)"
                        % (key, n_ok, len(smiles_list), time.time() - t0, models[key].kind))
        except Exception as e:
            logger.warning("    %-30s FAILED: %s" % (key, e))
            out[key] = np.full(len(smiles_list), np.nan)
    return pd.DataFrame(out)


def predict_pair_properties(models, pairs_df, batch_size=20000):
    """Apply pair-level models (device props, E_loss, mobilities) to the shard."""
    pair_keys = [k for k, tm in models.items() if getattr(tm, "is_pair_level", False)]
    if not pair_keys or pairs_df.empty:
        return pd.DataFrame(index=pairs_df.index)
    logger.info("  Predicting %d pair-level targets for %d pair rows: %s"
                % (len(pair_keys), len(pairs_df), ", ".join(sorted(pair_keys))))

    donors = pairs_df["donor_smiles"].astype(str).values
    acceptors = pairs_df["acceptor_smiles"].astype(str).values

    # Featurisation is cached per unique molecule: the pool is a cross-product,
    # so the same donor appears in thousands of rows and re-fingerprinting it
    # every time would dominate the runtime.
    logger.info("    Building fingerprint cache over unique molecules...")
    uniq_d = {s: featurize_smiles(s) for s in set(donors)}
    uniq_a = {s: featurize_smiles(s) for s in set(acceptors)}
    logger.info("    Cached %d donor + %d acceptor fingerprints"
                % (len(uniq_d), len(uniq_a)))

    results = {k: np.full(len(pairs_df), np.nan) for k in pair_keys}
    n = len(pairs_df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats, idx = [], []
        for i in range(start, end):
            fd = uniq_d.get(donors[i])
            fa = uniq_a.get(acceptors[i])
            if fd is None or fa is None:
                continue
            feats.append(np.concatenate([fd, fa]))
            idx.append(i)
        if not feats:
            continue
        Xb = np.asarray(feats, dtype=np.float32)
        ib = np.asarray(idx)
        for k in pair_keys:
            try:
                results[k][ib] = models[k].model.predict(Xb)
            except Exception as e:
                logger.debug("pair predict %s failed on batch %d: %s" % (k, start, e))
        if (start // batch_size) % 10 == 0:
            logger.info("      pair-level progress: %d/%d rows" % (end, n))

    for k in pair_keys:
        n_ok = int(np.isfinite(results[k]).sum())
        logger.info("    %-30s -> %d/%d predicted" % (k, n_ok, n))
    return pd.DataFrame(results, index=pairs_df.index)


def apply_eloss_physics_gate(pair_pred_df):
    """Flag predicted E_loss values that violate the physical floor.

    Prediction is a model output, so it can land below the detailed-balance
    floor even when trained only on physical rows. Those rows are FLAGGED (never
    silently rewritten): the primary reported E_loss column excludes them.
    """
    if "dev_E_loss" not in pair_pred_df.columns:
        return pair_pred_df
    out = pair_pred_df.copy()
    e = pd.to_numeric(out["dev_E_loss"], errors="coerce")
    out["eloss_pred_raw"] = e
    out["eloss_pred_negative"] = e < 0
    out["eloss_pred_unphysical"] = e < DETAILED_BALANCE_ELOSS_FLOOR_EV
    out["eloss_pred_physical"] = e.where(e >= DETAILED_BALANCE_ELOSS_FLOOR_EV, np.nan)
    n_neg = int((e < 0).sum())
    n_unphys = int((e < DETAILED_BALANCE_ELOSS_FLOOR_EV).sum())
    n_tot = int(e.notna().sum())
    logger.info("  E_loss physics gate on PREDICTED values: %d/%d negative, "
                "%d/%d below %.2f eV detailed-balance floor -> excluded from "
                "'eloss_pred_physical'" % (n_neg, n_tot, n_unphys, n_tot,
                                           DETAILED_BALANCE_ELOSS_FLOOR_EV))
    return out


# ============================================================================
# STAGE D -- OPTIONAL RL FINE-TUNING  (OFF by default: --run-rl-finetune)
#
# DESIGN RATIONALE (why the reward looks like this):
#   * SAScore is the PRIMARY always-on reward term. It is a pure RDKit
#     fragment-frequency heuristic: microseconds per molecule, no learned
#     classifier, and therefore NO chemical-domain restriction. That makes it
#     the only synthesizability signal cheap enough to evaluate every RL step.
#   * The TANGO-inspired heuristic is the SECONDARY term because it is the only
#     signal that knows about OPV-specific building blocks, which SAScore does
#     not. Blending the two gives a reward that is both generic-complexity aware
#     and domain aware.
#   * AiZynthFinder MCTS is deliberately NOT in the loop: at ~seconds/molecule
#     it is 3-6 orders of magnitude too slow for a per-step reward. It is used
#     as a post-hoc audit instead (Stage C).
#   * Optuna tunes the blend weight and the learning rate -- the two parameters
#     that genuinely change the policy's behaviour.
#
# ISOLATION GUARANTEE: output goes to rl_finetune_candidates.csv ONLY. It is
# never merged into the 13-approach pool, so previously reported per-approach
# totals, diversity numbers and distances are not retroactively altered.
# ============================================================================


def _build_rl_generator(latent_dim, n_bits, hidden=(256, 512)):
    """Small fingerprint-space generator used for the RL fine-tune."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow unavailable; RL fine-tune not possible.")
    inp = Input(shape=(latent_dim,))
    x = layers.Dense(hidden[0], activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden[1], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(n_bits, activation="sigmoid")(x)
    return Model(inp, out, name="RL_FP_Generator")


def _decode_by_nearest_template(fp_vec, template_fps, template_smiles, tango, top_k=6):
    """Map a generated fingerprint to a real molecule via template retrieval + mutation.

    A fingerprint is not invertible to a molecule, so the policy's output is
    used as a RETRIEVAL KEY: find the closest training templates, then apply the
    same BRICS/SMARTS mutation operators the generation pipeline uses to produce
    a NOVEL neighbour. This keeps every emitted molecule chemically valid.
    """
    if not template_fps:
        return None
    v = (fp_vec > 0.5).astype(np.float32)
    if v.sum() < 1:
        return None
    sims = []
    for i, tf_ in enumerate(template_fps):
        inter = float(np.dot(v, tf_))
        denom = float(v.sum() + tf_.sum() - inter)
        sims.append(inter / denom if denom > 0 else 0.0)
    order = np.argsort(sims)[::-1][:top_k]
    for i in order:
        base = template_smiles[i]
        mol = Chem.MolFromSmiles(base)
        if mol is None:
            continue
        cands = []
        try:
            frags = list(BRICS.BRICSDecompose(mol))
            if len(frags) >= 2:
                built = BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in frags if Chem.MolFromSmiles(f)])
                for j, m2 in enumerate(built):
                    if j >= 4:
                        break
                    try:
                        Chem.SanitizeMol(m2)
                        cands.append(Chem.MolToSmiles(m2))
                    except Exception:
                        continue
        except Exception:
            pass
        for smi in cands:
            if smi and smi != base and is_valid_smiles(smi):
                return smi
    return None


def stage_d_rl_finetune(train_smiles, tango, molecule_type="Donor"):
    """Small-scale RL fine-tune with an SAScore+TANGO blended reward."""
    logger.info("=" * 80)
    logger.info("STAGE D -- OPTIONAL RL FINE-TUNING (%s)" % molecule_type)
    logger.info("=" * 80)
    if not TF_AVAILABLE:
        logger.warning("TensorFlow unavailable -- RL fine-tune SKIPPED.")
        return pd.DataFrame()
    if not SASCORER_AVAILABLE:
        logger.warning("SAScore unavailable -- the primary reward term is missing; "
                       "RL fine-tune SKIPPED to avoid an unjustified reward.")
        return pd.DataFrame()

    n_bits = 2048
    latent_dim = 64
    templates = balanced_trait_order(list(train_smiles), molecule_type)[:400]
    tmpl_fps, tmpl_smiles = [], []
    for s in templates:
        f = featurize_smiles(s, n_bits=n_bits)
        if f is not None:
            tmpl_fps.append(f)
            tmpl_smiles.append(s)
    if len(tmpl_fps) < 10:
        logger.warning("Too few featurisable templates (%d) -- RL fine-tune SKIPPED." % len(tmpl_fps))
        return pd.DataFrame()
    logger.info("  Templates for retrieval-decode: %d" % len(tmpl_fps))

    train_can = canonical_smiles_set(train_smiles)

    def run_rl(alpha_sa, learning_rate, epochs, batch, seed=ARGS.seed, collect=False):
        """One RL run. Returns (mean_final_reward, collected_records)."""
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        gen = _build_rl_generator(latent_dim, n_bits)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        baseline = 0.0
        records = []
        epoch_means = []

        for ep in range(epochs):
            noise = tf.random.normal((batch, latent_dim))
            with tf.GradientTape() as tape:
                probs = gen(noise, training=True)
                # Bernoulli sample with a straight-through style log-prob so the
                # REINFORCE gradient is well-defined on a discrete action.
                sample = tf.cast(tf.random.uniform(tf.shape(probs)) < probs, tf.float32)
                eps = 1e-7
                logp = tf.reduce_sum(
                    sample * tf.math.log(probs + eps) +
                    (1.0 - sample) * tf.math.log(1.0 - probs + eps), axis=1)

                sample_np = sample.numpy()
                rewards = np.zeros(batch, dtype=np.float32)
                for b in range(batch):
                    smi = _decode_by_nearest_template(sample_np[b], tmpl_fps, tmpl_smiles, tango)
                    if not smi:
                        rewards[b] = 0.0
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        rewards[b] = 0.0
                        continue
                    sa = compute_sascore(mol)
                    r_sa = sascore_to_reward(sa)
                    r_tango = float(tango.score(mol)) if tango is not None else 0.0
                    can = canonical_smiles(smi)
                    novelty_bonus = 0.10 if (can and can not in train_can) else 0.0
                    r = alpha_sa * r_sa + (1.0 - alpha_sa) * r_tango + novelty_bonus
                    rewards[b] = float(np.clip(r, 0.0, 1.2))
                    if collect and can and can not in train_can:
                        records.append({
                            "molecule_type": molecule_type, "smiles": smi,
                            "canonical_smiles": can, "epoch": ep,
                            "sascore": sa if sa is not None else np.nan,
                            "sascore_reward": r_sa, "tango_inspired": r_tango,
                            "blended_reward": rewards[b],
                            "alpha_sascore": alpha_sa, "learning_rate": learning_rate,
                            "source": "RL_Finetune_SAScore_TANGO",
                        })

                mean_r = float(np.mean(rewards))
                baseline = 0.9 * baseline + 0.1 * mean_r if ep > 0 else mean_r
                advantage = tf.constant(rewards - baseline, dtype=tf.float32)
                loss = -tf.reduce_mean(logp * advantage)

            grads = tape.gradient(loss, gen.trainable_variables)
            if grads is not None and any(g is not None for g in grads):
                opt.apply_gradients([(g, v) for g, v in zip(grads, gen.trainable_variables)
                                     if g is not None])
            epoch_means.append(mean_r)
            if collect or ep % max(1, epochs // 5) == 0:
                logger.info("    epoch %3d/%d | mean reward %.4f | baseline %.4f | loss %.4f"
                            % (ep, epochs, mean_r, baseline, float(loss.numpy())))
        return float(np.mean(epoch_means[-3:])) if epoch_means else 0.0, records

    # --- Optuna over the two parameters that actually matter ---------------
    best_alpha, best_lr = 0.6, 1e-3
    if OPTUNA_AVAILABLE:
        logger.info("  Optuna: tuning reward blend weight (alpha_SAScore) and learning rate...")

        def objective(trial):
            a = trial.suggest_float("alpha_sascore", 0.2, 0.9)
            lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
            score, _ = run_rl(a, lr, epochs=max(4, ARGS.rl_epochs // 3),
                              batch=max(16, ARGS.rl_batch // 2), seed=ARGS.seed)
            return score

        try:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=ARGS.seed))
            study.optimize(objective, n_trials=max(3, ARGS.optuna_trials // 3),
                           show_progress_bar=False)
            best_alpha = float(study.best_params.get("alpha_sascore", best_alpha))
            best_lr = float(study.best_params.get("learning_rate", best_lr))
            logger.info("  Optuna best: alpha_SAScore=%.3f, learning_rate=%.5f (reward=%.4f)"
                        % (best_alpha, best_lr, study.best_value))
        except Exception as e:
            logger.warning("  Optuna RL tuning failed (%s); using defaults." % e)
    else:
        logger.info("  Optuna unavailable -- using default alpha=0.6, lr=1e-3")

    logger.info("  Final RL run: epochs=%d batch=%d alpha=%.3f lr=%.5f"
                % (ARGS.rl_epochs, ARGS.rl_batch, best_alpha, best_lr))
    _, records = run_rl(best_alpha, best_lr, ARGS.rl_epochs, ARGS.rl_batch,
                        seed=ARGS.seed, collect=True)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("blended_reward", ascending=False)
        df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    logger.info("  RL fine-tune produced %d unique novel candidates" % len(df))
    logger.info("  ISOLATION: written only to rl_finetune_candidates.csv; the 13-approach "
                "pool and all previously reported totals remain unchanged.")
    return df


# ============================================================================
# STAGE C ORCHESTRATION + STAGE E WRITE-OUT
# ============================================================================


def approach_column(pairs_df, side):
    """Best-effort per-molecule approach/source label column name."""
    for c in ("%s_source" % side, "source", "approach"):
        if c in pairs_df.columns:
            return c
    return None


def stage_c_score(models, pairs_df, dmpnn_df, guzik_df, tango, auditor):
    """Score the shard: molecule properties, pair properties, metrics, audit."""
    logger.info("=" * 80)
    logger.info("STAGE C -- SCORING THE GENERATED POOL (rank %d shard)" % RANK)
    logger.info("=" * 80)

    outputs = {}
    donors, acceptors = extract_unique_molecules(pairs_df)

    # ---- shard the unique-molecule work across ranks ----
    my_donors = shard_list(donors, RANK, WORLD_SIZE) if IS_DISTRIBUTED else donors
    my_acceptors = shard_list(acceptors, RANK, WORLD_SIZE) if IS_DISTRIBUTED else acceptors
    logger.info("  This rank's molecule shard: %d donors, %d acceptors"
                % (len(my_donors), len(my_acceptors)))

    # ---- C1: individual molecule property prediction ----
    logger.info("-" * 80)
    logger.info("  [C1] Individual molecule property prediction")
    don_pred = predict_molecule_properties(models, my_donors, "donor")
    acc_pred = predict_molecule_properties(models, my_acceptors, "acceptor")

    # ---- C2: SAScore + TANGO-inspired screen per molecule ----
    logger.info("-" * 80)
    logger.info("  [C2] SAScore + TANGO-inspired synthesizability screen")
    for df_, smis, label in ((don_pred, my_donors, "donor"), (acc_pred, my_acceptors, "acceptor")):
        if df_.empty:
            continue
        sas, tgs = [], []
        for s in smis:
            m = Chem.MolFromSmiles(str(s))
            sa = compute_sascore(m) if m is not None else None
            sas.append(sa if sa is not None else np.nan)
            tgs.append(float(tango.score(m)) if (m is not None and tango is not None) else np.nan)
        df_["sascore"] = sas
        df_["sascore_reward"] = [sascore_to_reward(x if np.isfinite(x) else None) for x in sas]
        df_["tango_inspired_screen"] = tgs
        finite = np.asarray([x for x in sas if np.isfinite(x)])
        if finite.size:
            logger.info("    %s SAScore: mean=%.3f min=%.3f max=%.3f (n=%d)"
                        % (label, finite.mean(), finite.min(), finite.max(), finite.size))
    outputs["donor_molecule_predictions"] = don_pred
    outputs["acceptor_molecule_predictions"] = acc_pred

    # ---- C3: pair-level prediction on the pair shard ----
    logger.info("-" * 80)
    logger.info("  [C3] Pair-level device + E_loss prediction")
    my_pairs = shard_dataframe(pairs_df, RANK, WORLD_SIZE) if IS_DISTRIBUTED else pairs_df
    logger.info("    This rank's pair shard: %d rows" % len(my_pairs))
    pair_pred = predict_pair_properties(models, my_pairs)
    pair_pred = apply_eloss_physics_gate(pair_pred)
    keep = [c for c in ("donor_smiles", "acceptor_smiles", "PCE", "Jsc", "FF", "Voc",
                        "pair_tango", "tier", "favorable_homo_lumo_alignment")
            if c in my_pairs.columns]
    pair_out = pd.concat([my_pairs[keep].reset_index(drop=True),
                          pair_pred.reset_index(drop=True)], axis=1)
    outputs["pair_predictions"] = pair_out

    # ---- C4: generative metrics (rank 0 only -- they are pool-level) ----
    if IS_MASTER:
        logger.info("-" * 80)
        logger.info("  [C4] Generative metrics (MOSES quartet + 3 distances + Wasserstein/Mahalanobis)")
        train_donors = [s for s in dmpnn_df.get("donor(SMILES)", pd.Series(dtype=str)).dropna().astype(str)]
        train_acceptors = [s for s in dmpnn_df.get("acceptor(SMILES)", pd.Series(dtype=str)).dropna().astype(str)]
        if guzik_df is not None:
            train_donors += [s for s in guzik_df.get("Donor SMILES", pd.Series(dtype=str)).dropna().astype(str)]
            train_acceptors += [s for s in guzik_df.get("Acceptor SMILES", pd.Series(dtype=str)).dropna().astype(str)]

        # PFD embedding model: prefer a winning ACCEPTOR property model (largest N)
        embed_donor = _pick_embedding_model(models, "don_")
        embed_acceptor = _pick_embedding_model(models, "acc_")
        logger.info("    PFD embedding (donor side)   : %s"
                    % (embed_donor.key if embed_donor else "unavailable"))
        logger.info("    PFD embedding (acceptor side): %s"
                    % (embed_acceptor.key if embed_acceptor else "unavailable"))

        metric_rows = []
        metric_rows.append(compute_metric_block(
            "donor_pooled_all_approaches", donors, train_donors, embed_donor, seed=ARGS.seed))
        metric_rows.append(compute_metric_block(
            "acceptor_pooled_all_approaches", acceptors, train_acceptors, embed_acceptor, seed=ARGS.seed))

        # Per-approach + "excluding Training_Set_Augmentation" variants.
        for side, smis, train_ref, emb in (("donor", donors, train_donors, embed_donor),
                                           ("acceptor", acceptors, train_acceptors, embed_acceptor)):
            col = approach_column(pairs_df, side)
            if col is None:
                logger.warning("    No per-molecule approach column for %s side; per-approach "
                               "metric breakdown unavailable in this pool. The merge stage "
                               "computes it from the per-rank source files instead." % side)
                continue
            smi_col = "%s_smiles" % side
            sub = pairs_df[[smi_col, col]].dropna().drop_duplicates(subset=[smi_col])
            groups = sub.groupby(col)[smi_col].apply(lambda s: sorted(set(s.astype(str))))
            excl = []
            for approach, mols in groups.items():
                metric_rows.append(compute_metric_block(
                    "%s_approach_%s" % (side, approach), list(mols), train_ref, emb, seed=ARGS.seed))
                if str(approach) != "Training_Set_Augmentation":
                    excl.extend(mols)
            if excl:
                metric_rows.append(compute_metric_block(
                    "%s_pooled_excluding_Training_Set_Augmentation" % side,
                    sorted(set(excl)), train_ref, emb, seed=ARGS.seed))

        outputs["generative_metrics"] = pd.DataFrame(metric_rows)

    # ---- C5: retrosynthesis audit of the Tier-1 shortlist ----
    logger.info("-" * 80)
    logger.info("  [C5] Retrosynthesis audit (backend=%s)" % auditor.backend)
    audit_df = _audit_shortlist(my_pairs, auditor)
    outputs["retrosynthesis_audit"] = audit_df

    return outputs


def _pick_embedding_model(models, prefix):
    """Choose the property model whose embedding space defines PFD.

    Selection rule (deterministic and reported): among single-molecule models on
    the requested side, take the one with the most training rows, breaking ties
    on held-out R^2. Using the best-supported model means the PFD embedding is
    the most trustworthy property space available for that side.
    """
    cands = [tm for k, tm in models.items()
             if k.startswith(prefix) and not getattr(tm, "is_pair_level", False)
             and tm.embedding_available]
    if not cands:
        return None
    def _key(tm):
        r2 = tm.cv_scores.get("r2", -np.inf)
        r2 = r2 if np.isfinite(r2) else -np.inf
        return (tm.n_train, r2)
    return sorted(cands, key=_key, reverse=True)[0]


def _audit_shortlist(pairs_df, auditor):
    """Audit the top-N pairs per approach (or globally) from the Tier-1 shortlist."""
    if pairs_df.empty:
        return pd.DataFrame()
    df = pairs_df.copy()
    if "tier" in df.columns:
        t1 = df[df["tier"].astype(str).str.contains("Tier1", na=False)]
        if not t1.empty:
            df = t1
            logger.info("    Restricting audit to Tier-1 rows: %d" % len(df))
        else:
            logger.info("    No Tier-1 rows in this shard; auditing the top of the shard instead.")
    sort_col = "PCE" if "PCE" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    top_n = max(1, ARGS.audit_top_n)
    per_rank_n = max(1, top_n // max(1, WORLD_SIZE)) if IS_DISTRIBUTED else top_n
    df = df.head(per_rank_n)

    mols = set()
    if "donor_smiles" in df.columns:
        mols.update(df["donor_smiles"].dropna().astype(str).tolist())
    if "acceptor_smiles" in df.columns:
        mols.update(df["acceptor_smiles"].dropna().astype(str).tolist())
    mols = sorted(mols)
    logger.info("    Auditing %d unique molecules from %d shortlisted pairs (backend=%s)"
                % (len(mols), len(df), auditor.backend))

    recs = []
    t0 = time.time()
    for i, s in enumerate(mols):
        recs.append(auditor.audit_molecule(s))
        if (i + 1) % 100 == 0:
            logger.info("      audited %d/%d (%.1fs elapsed)" % (i + 1, len(mols), time.time() - t0))
    out = pd.DataFrame(recs)
    if not out.empty and "route_found" in out.columns:
        rf = pd.to_numeric(out["route_found"], errors="coerce")
        n_found = int((rf == 1.0).sum())
        n_eval = int(rf.notna().sum())
        if n_eval:
            logger.info("    Audit result: %d/%d molecules with a route found (%.1f%%) [backend=%s]"
                        % (n_found, n_eval, 100.0 * n_found / n_eval, auditor.backend))
            if auditor.backend == "rascore":
                logger.warning("    REMINDER: RAscore is ChEMBL-domain; these OPV scaffolds are "
                               "out of domain. Do NOT report as validated synthesizability.")
    return out


def stage_e_write(outputs, race_df=None, rl_df=None):
    """Write per-rank shard outputs (Stage E)."""
    logger.info("=" * 80)
    logger.info("STAGE E -- WRITING RANK %d SHARD OUTPUTS" % RANK)
    logger.info("=" * 80)
    safe_makedirs(RANK_OUTDIR)
    written = []
    for name, df in outputs.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.info("  %-34s : empty, not written" % name)
            continue
        p = os.path.join(RANK_OUTDIR, "%s.csv" % name)
        try:
            df.to_csv(p, index=False)
            written.append(p)
            logger.info("  %-34s : %d rows -> %s" % (name, len(df), p))
        except Exception as e:
            logger.error("  Failed writing %s: %s" % (name, e))

    if IS_MASTER and race_df is not None and not race_df.empty:
        p = os.path.join(OUT_ROOT, "model_selection_horse_race.csv")
        try:
            race_df.to_csv(p, index=False)
            logger.info("  model_selection_horse_race.csv -> %s" % p)
        except Exception as e:
            logger.warning("  Could not write horse race: %s" % e)

    if IS_MASTER and rl_df is not None and not rl_df.empty:
        p = os.path.join(OUT_ROOT, "rl_finetune_candidates.csv")
        try:
            rl_df.to_csv(p, index=False)
            logger.info("  rl_finetune_candidates.csv -> %s (%d rows)" % (p, len(rl_df)))
        except Exception as e:
            logger.warning("  Could not write RL candidates: %s" % e)

    logger.info("  Rank %d wrote %d files." % (RANK, len(written)))
    return written


def load_saved_models():
    """Reload persisted models (used by --score-only and by non-master ranks)."""
    models = {}
    if not os.path.isdir(POSTGEN_MODELS_DIR):
        logger.warning("No saved models directory at %s" % POSTGEN_MODELS_DIR)
        return models
    for path in sorted(glob.glob(os.path.join(POSTGEN_MODELS_DIR, "*.pkl"))):
        base = os.path.basename(path)
        if base.endswith(".meta.pkl"):
            continue
        key = base[:-4]
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            tm = TargetModel(key, "classical", blob["model"], blob.get("cv", {}),
                             blob.get("params", {}), blob.get("n_train", 0),
                             blob.get("n_groups", 0), blob.get("group_kind", ""),
                             blob.get("source", ""), blob.get("column", ""),
                             blob.get("needs_abs", False), embedding_available=True)
            tm.is_pair_level = blob.get("is_pair_level", False)
            models[key] = tm
        except Exception as e:
            logger.warning("Could not load model %s: %s" % (key, e))
    for meta in sorted(glob.glob(os.path.join(POSTGEN_MODELS_DIR, "*.meta.pkl"))):
        key = os.path.basename(meta)[:-len(".meta.pkl")]
        kpath = os.path.join(POSTGEN_MODELS_DIR, "%s_dmpnn.keras" % key)
        if not (TF_AVAILABLE and os.path.exists(kpath)):
            continue
        try:
            with open(meta, "rb") as fh:
                blob = pickle.load(fh)
            model = tf.keras.models.load_model(kpath, compile=False)
            tm = TargetModel(key, "dmpnn", model, blob.get("cv", {}), blob.get("params", {}),
                             blob.get("n_train", 0), blob.get("n_groups", 0),
                             blob.get("group_kind", ""), blob.get("source", ""),
                             blob.get("column", ""), blob.get("needs_abs", False),
                             embedding_available=True)
            tm._y_mu = blob.get("y_mu", 0.0)
            tm._y_sd = blob.get("y_sd", 1.0)
            tm.is_pair_level = False
            models[key] = tm
        except Exception as e:
            logger.warning("Could not load D-MPNN model %s: %s" % (key, e))
    logger.info("Reloaded %d saved models from %s" % (len(models), POSTGEN_MODELS_DIR))
    return models


def wait_for_models(poll=15):
    """Non-master ranks wait for rank 0 to publish the DONE sentinel.

    NO TIMEOUT: waits indefinitely. Stage B's full Optuna/CV grid across ~25
    targets legitimately took ~9h in a prior real run (see
    property_model_cv_report.csv timing), and there is no safe upper bound to
    guess at -- a timeout here means every non-master rank gives up early and
    proceeds with EMPTY models, silently producing all-NaN predictions for
    239/240 of the pool while still looking like a normal completion in the
    log. Logs a heartbeat every 10 minutes so a long wait is visible, not
    silent. This function only returns True (once the sentinel appears); it
    never returns False.
    """
    sentinel = os.path.join(POSTGEN_MODELS_DIR, "TRAINING_DONE")
    t0 = time.time()
    logger.info("Rank %d waiting for rank-0 training sentinel: %s (no timeout -- "
                "will wait as long as required)" % (RANK, sentinel))
    last_heartbeat = t0
    while True:
        if os.path.exists(sentinel):
            logger.info("Sentinel found after %.1f s" % (time.time() - t0))
            time.sleep(3)
            return True
        if time.time() - last_heartbeat > 600:
            logger.info("Rank %d still waiting for rank-0 training sentinel (%.0f s elapsed)..."
                        % (RANK, time.time() - t0))
            last_heartbeat = time.time()
        time.sleep(poll)


def main():
    t_start = time.time()
    logger.info("#" * 80)
    logger.info("POST-GENERATION PIPELINE :: MAIN")
    logger.info("#" * 80)

    # ---------------- STAGE A ----------------
    dmpnn_df, guzik_df, diag = stage_a_load_data()

    all_train_smiles = []
    if "donor(SMILES)" in dmpnn_df.columns:
        all_train_smiles += dmpnn_df["donor(SMILES)"].dropna().astype(str).tolist()
    if "acceptor(SMILES)" in dmpnn_df.columns:
        all_train_smiles += dmpnn_df["acceptor(SMILES)"].dropna().astype(str).tolist()
    if guzik_df is not None:
        for c in ("Donor SMILES", "Acceptor SMILES"):
            if c in guzik_df.columns:
                all_train_smiles += guzik_df[c].dropna().astype(str).tolist()

    tango = TangoInspiredHeuristic(training_smiles=all_train_smiles, alpha=0.5)

    # ---------------- STAGE B ----------------
    race_df = pd.DataFrame()
    if ARGS.score_only:
        logger.info("--score-only: skipping Stage B; loading persisted models.")
        models = load_saved_models()
    elif IS_MASTER:
        # ================================================================
        # CRITICAL: remove the stale TRAINING_DONE sentinel from any prior
        # run BEFORE starting training. If we don't, every non-master rank
        # finds the old sentinel immediately (within milliseconds of
        # starting) and loads whatever old models happen to sit in
        # models/postgen/ -- completing with the WRONG model set long before
        # THIS run's training finishes. This is exactly what happened in
        # job 70355 (Aug 7-8 2026) and produced the 1-valid-shard problem
        # for a THIRD time with a different mechanism than the timeout.
        # ================================================================
        _sentinel_path = os.path.join(POSTGEN_MODELS_DIR, "TRAINING_DONE")
        if os.path.exists(_sentinel_path):
            os.remove(_sentinel_path)
            logger.info("Removed stale TRAINING_DONE sentinel from a prior run.")
        models, race_df = stage_b_train_models(dmpnn_df, guzik_df)
        save_models(models, race_df)
        try:
            with open(os.path.join(POSTGEN_MODELS_DIR, "TRAINING_DONE"), "w") as fh:
                fh.write(datetime.datetime.now().isoformat())
            logger.info("Published training sentinel for other ranks.")
        except Exception as e:
            logger.warning("Could not write sentinel: %s" % e)
    else:
        # Non-master, non-score-only: wait for rank 0's FRESH sentinel.
        # The sentinel is deleted by rank 0 at the START of its training
        # (see the IS_MASTER branch above), so if we see it exist right now
        # before rank 0 has had time to delete it, we must wait until it
        # disappears and reappears rather than trusting a stale one. The
        # simplest correct logic: wait until the sentinel has a timestamp
        # AFTER this process's own start time (t_start).
        _sentinel_path = os.path.join(POSTGEN_MODELS_DIR, "TRAINING_DONE")
        if os.path.exists(_sentinel_path):
            # A sentinel exists from a prior run. Wait for rank 0 to delete
            # and recreate it. Give rank 0 up to 120s to start and delete it.
            logger.info("Rank %d: stale TRAINING_DONE sentinel detected from a prior run; "
                        "waiting for rank 0 to delete and recreate it..." % RANK)
            _wait_t0 = time.time()
            while os.path.exists(_sentinel_path) and time.time() - _wait_t0 < 120:
                time.sleep(5)
            if os.path.exists(_sentinel_path):
                # Rank 0 didn't delete it within 120s -- maybe this IS a --score-only
                # scenario where rank 0 also just loaded and exited. Trust it.
                logger.warning("Rank %d: sentinel still present after 120s; trusting it "
                               "(rank 0 may be in score-only mode)." % RANK)
        wait_for_models()  # blocks indefinitely until the (fresh) sentinel appears
        models = load_saved_models()

    if ARGS.train_only:
        logger.info("--train-only: Stage B finished; exiting before scoring.")
        logger.info("Total elapsed: %.1f s" % (time.time() - t_start))
        return

    # ---------------- STAGE C ----------------
    pairs_path = find_pairs_csv()
    if pairs_path is None:
        logger.error("Could not locate generated_molecules_with_predictions.csv. "
                     "Pass it explicitly with --pairs-csv. Stage C skipped.")
        outputs = {}
    else:
        pairs_df = load_pairs_csv(pairs_path, max_rows=ARGS.max_pairs)
        auditor = RetrosynthesisAuditor(tango=tango)
        outputs = stage_c_score(models, pairs_df, dmpnn_df, guzik_df, tango, auditor)

    # ---------------- STAGE D (optional) ----------------
    rl_df = pd.DataFrame()
    if ARGS.run_rl_finetune and IS_MASTER:
        try:
            d_train = dmpnn_df.get("donor(SMILES)", pd.Series(dtype=str)).dropna().astype(str).tolist()
            a_train = dmpnn_df.get("acceptor(SMILES)", pd.Series(dtype=str)).dropna().astype(str).tolist()
            rl_d = stage_d_rl_finetune(d_train, tango, "Donor")
            rl_a = stage_d_rl_finetune(a_train, tango, "Acceptor")
            rl_df = pd.concat([rl_d, rl_a], ignore_index=True) \
                if (not rl_d.empty or not rl_a.empty) else pd.DataFrame()
        except Exception as e:
            logger.error("RL fine-tune failed: %s" % e)
    elif ARGS.run_rl_finetune:
        logger.info("RL fine-tune requested but this is rank %d; only rank 0 runs it." % RANK)

    # ---------------- STAGE E ----------------
    stage_e_write(outputs, race_df=race_df, rl_df=rl_df)

    logger.info("#" * 80)
    logger.info("POST-GENERATION PIPELINE COMPLETE (rank %d)" % RANK)
    logger.info("  Elapsed: %.1f s" % (time.time() - t_start))
    logger.info("  NEXT: python merge_rank_outputs_final.py --stage postgen "
                "--indir %s --outdir <merged_dir>" % OUT_ROOT)
    logger.info("#" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("FATAL: post-generation pipeline aborted: %s" % exc)
        raise
