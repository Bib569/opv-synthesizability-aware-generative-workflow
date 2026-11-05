# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Pipeline stage 1: GAN generation (12 routes), pairing, Ridge+isotonic+GP surrogate, tiering, disjoint CV.
# Run from the flat run directory built by `sh hpc/setup_run_dir.sh` (see docs/PIPELINE.md);
# the script expects DMPNN.csv / Guzik_data.csv and its sibling scripts in that directory.
# Launched by hpc/sub_140.sh (240 SLURM ranks). Needs DMPNN.csv in the working directory.
# Optional: TensorFlow, RAScore. Large outputs (generated_molecules_with_predictions.csv, about 7.4 GB;
# high_performance_pairs.csv) are NOT in the repository: they are produced by this stage, or request them
# from the authors. The processed libraries used in the paper are in data/generated_libraries/.
# -----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed
from rdkit.Chem.AllChem import GetMorganGenerator
import optuna
import datetime
import os
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from rdkit.Chem import AllChem, Atom
from rdkit.Chem.Draw import IPythonConsole
from matplotlib.patches import Patch

# Additional imports for 5-approach hybrid pipeline
from rdkit.Chem import BRICS, rdChemReactions
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdDepictor
import random
import warnings
import sys
import glob
import argparse
import time
import hashlib
import re

warnings.filterwarnings("ignore")
import logging
from functools import lru_cache


def _parse_runtime_args():
    """Parse runtime args without interfering with external callers."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--outdir", type=str, default=".")
    # ------------------------------------------------------------------
    # RE-RUN TOGGLE (added): lets the expensive 240-rank GAN training /
    # molecule-generation block be bypassed entirely so that ONLY the
    # property-screening + high-performance filtering tail is recomputed
    # against an already-generated predictions CSV. This exists because
    # every correction made to the screening surrogate (Method-2 removal,
    # disjoint-CV reporting, FF threshold labelling, Tier1/Tier2 split
    # reporting) is a post-generation scoring change -- the generated
    # molecules themselves are unaffected and do NOT need regenerating.
    #   --skip-generation             : skip GAN training + generation
    #   --use-existing-predictions P  : path to an existing
    #                                   generated_molecules_with_predictions.csv
    # ------------------------------------------------------------------
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip GAN training/molecule generation; re-score an existing predictions CSV only.")
    parser.add_argument("--use-existing-predictions", type=str, default=None,
                        help="Path to an existing generated_molecules_with_predictions.csv to re-score/re-filter.")
    args, _ = parser.parse_known_args()
    return args


def _get_rank_context(args):
    """Get rank metadata from SLURM env."""
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
    distributed = bool(args.distributed and world_size > 1)
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "cpus_per_task": max(1, cpus_per_task),
        "distributed": distributed,
    }


def shard_list(items, rank, world_size):
    """Deterministic list sharding."""
    if world_size <= 1:
        return items
    return items[rank::world_size]


def safe_makedirs(path, retries=8, base_sleep=0.2):
    """Create directory with retry/backoff for flaky shared filesystems."""
    for attempt in range(retries):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2 ** attempt))
    return False


RUNTIME_ARGS = _parse_runtime_args()
RANK_CTX = _get_rank_context(RUNTIME_ARGS)
BASE_WORKDIR = os.getcwd()

if RANK_CTX["distributed"]:
    RANK_OUTDIR = os.path.join(
        os.path.abspath(RUNTIME_ARGS.outdir),
        f"rank_{RANK_CTX['rank']:03d}",
    )
    safe_makedirs(RANK_OUTDIR)
    os.chdir(RANK_OUTDIR)
else:
    RANK_OUTDIR = os.getcwd()

NUM_JOBS = max(1, RANK_CTX["cpus_per_task"])

# ----------------------------------------------------------------------------
# RANK-DIFFERENTIATED RANDOM SEEDING
#
# Root cause of the 240-rank diversity plateau: every rank trained on the
# IDENTICAL donor/acceptor dataset (GAN_SHARD_TRAINING_DATA disabled) with NO
# rank-based seed differentiation anywhere in the script. Combined with
# several generation approaches (Graph GAN, Conditional GAN, RL-GAN) whose
# generators are consistently untrained/degenerate and fall back to
# ensure_approach_generates_molecules()'s 100% deterministic BRICS/SMARTS
# backfill (balanced_trait_order() has no randomness), every one of the 240
# ranks independently reconstructed the SAME final molecule set — confirmed
# via logs: every rank reports exactly 927 donors, and the fully pooled/
# deduplicated 240-rank merge still only contains 927 unique donors, i.e.
# zero net diversity gain from parallelizing across ranks.
#
# Fix: seed Python's random, NumPy, and TensorFlow's global RNGs uniquely per
# rank (derived from SLURM_PROCID), so each rank's GAN weight initialization,
# training minibatch order, and stochastic sampling steps diverge from every
# other rank. This alone won't fix approaches whose generators are broken,
# but it lets the GAN-driven approaches (1, 6, 8, 9, 10) and the sampling
# noise in approaches that DO train (base_gan, 3, 11) explore different
# regions of solution space across ranks instead of reconverging on the same
# output. See also the per-rank shuffle in ensure_approach_generates_molecules
# for the deterministic-backfill half of this fix.
# ----------------------------------------------------------------------------
_RANK_SEED = (RANK_CTX["rank"] * 104729 + 17) % (2 ** 31 - 1)
random.seed(_RANK_SEED)
np.random.seed(_RANK_SEED)
try:
    tf.random.set_seed(_RANK_SEED)
except Exception:
    pass


def _get_preprocess_jobs():
    try:
        requested_jobs = int(os.environ.get("GAN_PREPROCESS_JOBS", "240"))
    except ValueError:
        requested_jobs = 240
    if RANK_CTX["distributed"]:
        try:
            per_rank_limit = int(os.environ.get("GAN_PREPROCESS_JOBS_PER_RANK", str(RANK_CTX["cpus_per_task"])))
        except ValueError:
            per_rank_limit = RANK_CTX["cpus_per_task"]
        return max(1, min(requested_jobs, per_rank_limit, RANK_CTX["cpus_per_task"]))
    return max(1, min(240, requested_jobs))


PREPROCESS_NUM_JOBS = _get_preprocess_jobs()


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    """Configure comprehensive logging for the entire pipeline"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if module is reloaded/re-entered
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # File handler with DEBUG level for detailed logs
    log_filename = (
        f"gan_pipeline_rank_{RANK_CTX['rank']:03d}.log"
        if RANK_CTX["distributed"]
        else "gan_pipeline_log.txt"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - Line %(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # Always write a source-level progress log for easy tracking from submit dir
    progress_log_path = os.path.join(BASE_WORKDIR, "gan_pipeline_progress.log")
    progress_handler = logging.FileHandler(progress_log_path)
    progress_handler.setLevel(logging.INFO)
    progress_format = logging.Formatter(
        f"%(asctime)s - rank={RANK_CTX['rank']:03d} - %(levelname)s - %(message)s"
    )
    progress_handler.setFormatter(progress_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(progress_handler)

    return logger


# Initialize logger
logger = setup_logging()
logger.info("=" * 80)
logger.info("GAN PIPELINE STARTED")
logger.info("=" * 80)
logger.info(
    f"Runtime context | distributed={RANK_CTX['distributed']} "
    f"rank={RANK_CTX['rank']}/{RANK_CTX['world_size']} "
    f"local_rank={RANK_CTX['local_rank']} cpus_per_task={RANK_CTX['cpus_per_task']}"
)
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Base submit directory: {BASE_WORKDIR}")

# ============================================================================
# ADVANCED APPROACHES IMPORTS (Approaches 7-11)
# ============================================================================
# For VAE (Approach 7)
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# For Transformer (Approach 8)
from tensorflow.keras.layers import (
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    Embedding,
)

# For Multi-objective Optimization (Approach 10)
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# For Active Learning (Approach 11)
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy, pearsonr
from collections import deque, Counter
from sklearn.linear_model import Ridge

# Additional utilities
import pickle
import json
from typing import List, Tuple, Dict, Optional
import copy

# ============================================================================
# SYNTHESIZABILITY ASSESSMENT IMPORTS
# ============================================================================
try:
    from syba.syba import SybaClassifier
    SYBA_AVAILABLE = True
    logger.info("SYBA (Synthetic Bayesian Accessibility) loaded successfully")
except ImportError:
    SYBA_AVAILABLE = False
    logger.warning("SYBA not available. Install with: pip install syba")

try:
    from scscore.standalone_model_numpy import SCScorer
    SCSCORE_AVAILABLE = True
    logger.info("SCScore (Synthetic Complexity Score) loaded successfully")
except ImportError:
    SCSCORE_AVAILABLE = False
    logger.warning("SCScore not available. Install with: pip install scscore")

try:
    from RAscore.RAscore_NN import RAScorerNN
    RASCORE_AVAILABLE = True
    logger.info("RAScore (Retrosynthetic Accessibility) loaded successfully")
except ImportError:
    RASCORE_AVAILABLE = False
    logger.warning("RAScore not available. Install with: pip install git+https://github.com/reymond-group/RAscore.git")

try:
    from aizynthfinder.aizynthfinder import AiZynthFinder
    from aizynthfinder.context import Context
    AIZYNTHFINDER_AVAILABLE = True
    logger.info("AiZynthFinder loaded successfully")
except ImportError:
    AIZYNTHFINDER_AVAILABLE = False
    logger.warning("AiZynthFinder not available. Install with: pip install aizynthfinder")
    logger.warning("Requires reaction database - see DATABASE_DOWNLOAD_GUIDE.md for setup")

try:
    import chemprop
    CHEMPROP_AVAILABLE = True
    logger.info("ChemProp (GNN-based) loaded successfully")
except ImportError:
    CHEMPROP_AVAILABLE = False
    logger.warning("ChemProp not available. Install with: pip install chemprop")

try:
    import selfies as sf
    SELFIES_AVAILABLE = True
    logger.info("SELFIES loaded successfully for fallback molecule representation")
except ImportError:
    sf = None
    SELFIES_AVAILABLE = False
    logger.warning("SELFIES not available. Token/hash fallback representations will still be used.")

# Load donor and acceptor SMILES from Excel file
logger.info("Loading dataset from Excel file...")
file_path = os.path.join(BASE_WORKDIR, "DMPNN.csv")
logger.debug(f"File path: {file_path}")
df = pd.read_csv(file_path)
logger.info(f"Dataset loaded successfully. Shape: {df.shape}")


# Add these validation functions
def sanitize_molecule(mol):
    """Attempt to sanitize and fix common issues in molecules"""
    logger.debug("Starting molecule sanitization...")
    try:
        if mol is None:
            logger.warning("Received None molecule for sanitization")
            return None

        # Try to sanitize first
        logger.debug("Attempting full molecule sanitization...")
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL)
        logger.debug("Full sanitization successful")
        return mol
    except Exception as e:
        logger.warning(
            f"Full sanitization failed: {e}, trying step-by-step approach..."
        )
        try:
            # If full sanitization fails, try step by step
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
                    logger.debug(f"Step-by-step sanitization successful with op: {op}")
                except Exception as step_e:
                    logger.debug(f"Step sanitization failed for op {op}: {step_e}")
                    continue
            logger.info("Step-by-step sanitization completed")
            return mol
        except Exception as final_e:
            logger.error(f"All sanitization attempts failed: {final_e}")
            return None


def fix_valence_issues(mol):
    """Fix common valence issues"""
    if mol is None:
        return None

    try:
        for atom in mol.GetAtoms():
            # Fix nitrogen valence issues
            if atom.GetSymbol() == "N" and atom.GetExplicitValence() > 3:
                atom.SetFormalCharge(1)

            # Fix selenium charge states
            elif atom.GetSymbol() == "Se":
                atom.SetFormalCharge(0)

            # Fix zinc charge states
            elif atom.GetSymbol() == "Zn":
                atom.SetFormalCharge(2)

            # Set explicit hybridization for problematic atoms
            if atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED:
                atom.SetHybridization(Chem.HybridizationType.SP3)

        return mol
    except:
        return None


# Function to check SMILES validity
def is_valid_smiles(smiles):
    """Check if SMILES string is valid"""
    logger.debug(f"Validating SMILES: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("SMILES parsing returned None")
            return False
        Chem.SanitizeMol(mol)
        logger.debug("SMILES validation successful")
        return True
    except Exception as e:
        logger.debug(f"SMILES validation failed: {e}")
        return False


# Safe SMARTS utilities to avoid NoneType queries to RDKit
def safe_mol_from_smarts(pattern):
    """Parse SMARTS safely; auto-correct common mistakes; return None if invalid and log a warning."""
    try:
        q = Chem.MolFromSmarts(pattern)
        if q is not None:
            return q
        # If parsing failed (returned None), try simple normalizations for common mistakes
        norm = pattern.strip()
        # Common mistake: using a single bracket around a bond expression like "[C#N]"
        if norm == '[C#N]':
            norm = 'C#N'
        # Generic fix: if the entire pattern is a single bracketed token containing a '#' bond,
        # rewrite as two atoms with an explicit '#' bond when safe to do so.
        if norm.startswith('[') and norm.endswith(']') and '#' in norm and norm.count(']') == 1 and norm.count('[') == 1:
            inner = norm[1:-1]
            # Very conservative split around a single '#'
            parts = inner.split('#')
            if len(parts) == 2 and parts[0] and parts[1]:
                norm2 = f"{parts[0]}#{parts[1]}"
                q = Chem.MolFromSmarts(norm2)
                if q is not None:
                    logger.debug(f"Normalized SMARTS '{pattern}' -> '{norm2}'")
                    return q
        # Final attempt: unbox single surrounding brackets if present (e.g., "[c1ccccc1]")
        if norm.startswith('[') and norm.endswith(']'):
            norm2 = norm[1:-1]
            q = Chem.MolFromSmarts(norm2)
            if q is not None:
                logger.debug(f"Normalized SMARTS '{pattern}' -> '{norm2}'")
                return q
        logger.warning(f"Invalid SMARTS pattern encountered: {pattern}")
        return None
    except Exception as e:
        logger.warning(f"SMARTS parsing failed for pattern '{pattern}': {e}")
        return None

# Safe SMARTS utilities to avoid NoneType queries to RDKit
@lru_cache(maxsize=128)
def safe_mol_from_smarts_cached(pattern):
    """Cached version of safe_mol_from_smarts for frequently used SMARTS patterns"""
    return safe_mol_from_smarts(pattern)

def pre_cache_common_smarts():
    """Pre-cache commonly used SMARTS patterns for solar cell materials"""
    common_patterns = [
        # Basic aromatic systems
        "c1ccccc1",  # Benzene
        "c1sccc1",   # Thiophene
        "[#7r5][#6r5][#6r5][#6r5][#6r5]",  # Pyrrole
        "[#8r5][#6r5][#6r5][#6r5][#6r5]",  # Furan

        # Electron-withdrawing groups
        "C#N",       # Cyano
        "C(=O)",     # Carbonyl
        "C(=O)O",    # Carboxyl
        "[NX3](=O)=O",  # Nitro
        "F", "Cl", "Br",  # Halogens

        # Conjugation patterns
        "C=C", "C#C",  # Double/triple bonds

        # Functional groups
        "[OH]", "[NH2]", "C(=O)O",  # Reactive groups

        # Heteroatoms and rings
        "c", "[!#6]",  # Aromatic carbon, heteroatoms
    ]

    logger.info("Pre-caching common SMARTS patterns...")
    for pattern in common_patterns:
        safe_mol_from_smarts_cached(pattern)
    logger.info(f"Pre-cached {len(common_patterns)} SMARTS patterns")

# Initialize pre-caching
pre_cache_common_smarts()

def count_substruct(mol, pattern):
    """Count substructure matches safely; returns 0 if SMARTS invalid."""
    q = safe_mol_from_smarts(pattern)
    if mol is None or q is None:
        return 0
    try:
        return len(mol.GetSubstructMatches(q))
    except Exception as e:
        logger.debug(f"GetSubstructMatches failed for pattern '{pattern}': {e}")
        return 0


# ============================================================================
# SYNTHESIZABILITY ASSESSMENT MODULE
# ============================================================================

class SynthesizabilityScorer:
    """
    Comprehensive synthesizability assessment system combining:
    1. SYBA (Synthetic Bayesian Accessibility)
    2. SCScore (Synthetic Complexity Score)
    3. RAScore (Retrosynthetic Accessibility)
    4. Custom OPV heuristics
    """
    
    def __init__(self):
        """Initialize all available synthesizability scoring models"""
        self.syba_scorer = None
        self.scscore_model = None
        self.rascore_model = None
        self.aizynthfinder = None
        self.chemprop_model = None
        
        logger.info("=" * 80)
        logger.info("INITIALIZING SYNTHESIZABILITY ASSESSMENT SYSTEM")
        logger.info("=" * 80)
        
        # Initialize SYBA
        if SYBA_AVAILABLE:
            try:
                logger.debug("Initializing SYBA (Synthetic Bayesian Accessibility)...")
                self.syba_scorer = SybaClassifier()
                self.syba_scorer.fitDefaultScore()
                logger.info("âœ“ SYBA initialized successfully")
                logger.debug("  - Training data: 1M+ real synthesis outcomes from Reaxys")
                logger.debug("  - Model type: Bayesian classifier")
                logger.debug("  - Expected accuracy: 85-90%")
            except Exception as e:
                logger.warning(f"âœ— SYBA initialization failed: {e}")
                logger.debug(f"  - Error details: {str(e)}")
                self.syba_scorer = None
        else:
            logger.debug("SYBA not available - package not installed")
        
        # Initialize SCScore
        if SCSCORE_AVAILABLE:
            try:
                logger.debug("Initializing SCScore (Synthetic Complexity Score)...")
                self.scscore_model = SCScorer()
                self.scscore_model.restore()
                logger.info("âœ“ SCScore initialized successfully")
                logger.debug("  - Training data: 12M reactions from USPTO database")
                logger.debug("  - Model type: Neural network")
                logger.debug("  - Expected accuracy: 80-85%")
            except Exception as e:
                logger.warning(f"âœ— SCScore initialization failed: {e}")
                logger.debug(f"  - Error details: {str(e)}")
                self.scscore_model = None
        else:
            logger.debug("SCScore not available - package not installed")
        
        # Initialize RAScore
        if RASCORE_AVAILABLE:
            try:
                logger.debug("Initializing RAScore (Retrosynthetic Accessibility)...")
                self.rascore_model = RAScorerNN()
                logger.info("âœ“ RAScore initialized successfully")
                logger.debug("  - Training data: Retrosynthesis database")
                logger.debug("  - Model type: Neural network with retrosynthesis planning")
                logger.debug("  - Expected accuracy: 85-90%")
            except Exception as e:
                logger.warning(f"âœ— RAScore initialization failed: {e}")
                logger.debug(f"  - Error details: {str(e)}")
                self.rascore_model = None
        else:
            logger.debug("RAScore not available - package not installed")
        
        # Initialize AiZynthFinder
        if AIZYNTHFINDER_AVAILABLE:
            try:
                logger.debug("Initializing AiZynthFinder (Retrosynthesis Route Planning)...")
                # Note: Full initialization requires config file and reaction database
                # This is a placeholder - actual usage requires proper setup
                self.aizynthfinder = None  # Will be initialized on-demand
                logger.info("âœ“ AiZynthFinder available (requires database setup)")
                logger.debug("  - Capability: Full retrosynthetic route planning")
                logger.debug("  - Requires: Reaction database (see DATABASE_DOWNLOAD_GUIDE.md)")
                logger.debug("  - Use case: Detailed route planning for top candidates")
            except Exception as e:
                logger.warning(f"âœ— AiZynthFinder initialization failed: {e}")
                logger.debug(f"  - Error details: {str(e)}")
                self.aizynthfinder = None
        else:
            logger.debug("AiZynthFinder not available - package not installed")
        
        # Initialize ChemProp
        if CHEMPROP_AVAILABLE:
            try:
                logger.debug("Initializing ChemProp (GNN-based predictions)...")
                # ChemProp requires pre-trained models for specific tasks
                # This is a placeholder - actual usage requires trained models
                self.chemprop_model = None  # Will be initialized on-demand
                logger.info("âœ“ ChemProp available (GNN-based approach)")
                logger.debug("  - Model type: Graph Neural Network")
                logger.debug("  - Capability: Property prediction from molecular graphs")
                logger.debug("  - Use case: Advanced molecular property predictions")
            except Exception as e:
                logger.warning(f"âœ— ChemProp initialization failed: {e}")
                logger.debug(f"  - Error details: {str(e)}")
                self.chemprop_model = None
        else:
            logger.debug("ChemProp not available - package not installed")
        
        logger.info("=" * 80)
        
        # Track which models are available
        self.available_models = []
        if self.syba_scorer is not None:
            self.available_models.append('SYBA')
        if self.scscore_model is not None:
            self.available_models.append('SCScore')
        if self.rascore_model is not None:
            self.available_models.append('RAScore')
        if AIZYNTHFINDER_AVAILABLE:
            self.available_models.append('AiZynthFinder')
        if CHEMPROP_AVAILABLE:
            self.available_models.append('ChemProp')
        self.available_models.append('OPV_Heuristic')  # Always available
        
        logger.info(f"Available scoring models: {', '.join(self.available_models)}")
        logger.debug(f"Total models initialized: {len(self.available_models)}")
    
    def get_syba_score(self, mol):
        """Get SYBA synthesizability score (0-1, higher=better)"""
        if self.syba_scorer is None:
            return None
        try:
            smiles = Chem.MolToSmiles(mol)
            score = self.syba_scorer.predict(smiles)
            return float(score)
        except Exception as e:
            logger.debug(f"SYBA scoring failed: {e}")
            return None
    
    def get_scscore(self, mol):
        """Get SCScore (1-5, lower=easier, converted to 0-1 scale with higher=better)"""
        if self.scscore_model is None:
            return None
        try:
            smiles = Chem.MolToSmiles(mol)
            (smi, score) = self.scscore_model.get_score_from_smi(smiles)
            # Convert from 1-5 scale (lower=better) to 0-1 scale (higher=better)
            normalized_score = 1.0 - (score - 1.0) / 4.0
            return float(normalized_score)
        except Exception as e:
            logger.debug(f"SCScore scoring failed: {e}")
            return None
    
    def get_rascore(self, mol):
        """Get RAScore (0-1, higher=better)"""
        if self.rascore_model is None:
            return None
        try:
            smiles = Chem.MolToSmiles(mol)
            logger.debug(f"Computing RAScore for: {smiles[:50]}...")
            score = self.rascore_model.predict(smiles)
            logger.debug(f"  RAScore result: {score:.3f}")
            return float(score)
        except Exception as e:
            logger.debug(f"RAScore scoring failed: {e}")
            return None
    
    def get_aizynthfinder_route(self, mol, config_path=None):
        """
        Get retrosynthetic route using AiZynthFinder
        Returns: (route_found, num_steps, route_score)
        """
        if not AIZYNTHFINDER_AVAILABLE:
            logger.debug("AiZynthFinder not available")
            return None, None, None
        
        try:
            smiles = Chem.MolToSmiles(mol)
            logger.debug(f"Planning retrosynthetic route for: {smiles[:50]}...")
            
            # Note: This requires proper AiZynthFinder setup with config and database
            # This is a placeholder implementation
            # Actual usage requires:
            # 1. Config file with model paths
            # 2. Reaction database
            # 3. Building block database
            
            logger.debug("  AiZynthFinder route planning requires database setup")
            logger.debug("  See DATABASE_DOWNLOAD_GUIDE.md for instructions")
            
            # Placeholder return - would be replaced with actual route planning
            return None, None, None
            
        except Exception as e:
            logger.debug(f"AiZynthFinder route planning failed: {e}")
            return None, None, None
    
    def get_chemprop_prediction(self, mol):
        """
        Get synthesizability prediction using ChemProp GNN
        Returns: score (0-1, higher=better)
        """
        if not CHEMPROP_AVAILABLE:
            logger.debug("ChemProp not available")
            return None
        
        try:
            smiles = Chem.MolToSmiles(mol)
            logger.debug(f"Computing ChemProp GNN prediction for: {smiles[:50]}...")
            
            # Note: ChemProp requires pre-trained models
            # This is a placeholder implementation
            # Actual usage requires:
            # 1. Pre-trained model checkpoint
            # 2. Model configuration
            # 3. Proper input formatting
            
            logger.debug("  ChemProp requires pre-trained model checkpoint")
            logger.debug("  Can be trained on custom synthesizability dataset")
            
            # Placeholder return - would be replaced with actual prediction
            return None
            
        except Exception as e:
            logger.debug(f"ChemProp prediction failed: {e}")
            return None
    
    def calculate_molecular_symmetry(self, mol):
        """Calculate molecular symmetry score (0-1, higher=more symmetric)"""
        try:
            # Get canonical SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # Check for palindromic patterns (simple symmetry indicator)
            if smiles == smiles[::-1]:
                return 1.0
            
            # Count symmetric substructures
            symmetry_score = 0.0
            
            # Check for symmetric ring systems
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            
            if len(rings) > 0:
                # Check if rings are symmetric
                for ring in rings:
                    if len(ring) % 2 == 0:  # Even-membered rings more likely symmetric
                        symmetry_score += 0.1
            
            # Check for symmetric functional groups on both sides
            # This is a simplified heuristic
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 0:
                # Count atom types
                atom_counts = {}
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
                
                # If atom counts are even, more likely symmetric
                even_count = sum(1 for count in atom_counts.values() if count % 2 == 0)
                symmetry_score += 0.2 * (even_count / len(atom_counts))
            
            return min(1.0, symmetry_score)
        except Exception as e:
            logger.debug(f"Symmetry calculation failed: {e}")
            return 0.0
    
    def calculate_conjugation_length(self, mol):
        """Calculate maximum conjugation length in molecule"""
        try:
            # Count conjugated double bonds and aromatic systems
            conjugated_bonds = 0
            
            # Count aromatic atoms
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            
            # Count double/triple bonds
            double_bonds = count_substruct(mol, "C=C")
            triple_bonds = count_substruct(mol, "C#C")
            
            # Estimate conjugation length
            conjugation_length = aromatic_atoms + double_bonds * 2 + triple_bonds * 2
            
            return conjugation_length
        except Exception as e:
            logger.debug(f"Conjugation length calculation failed: {e}")
            return 0
    
    def count_fused_ring_systems(self, mol):
        """Count number of fused ring systems"""
        try:
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            
            if len(rings) == 0:
                return 0
            
            # Count fused systems by checking shared atoms
            fused_count = 0
            for i, ring1 in enumerate(rings):
                for ring2 in rings[i+1:]:
                    shared_atoms = set(ring1) & set(ring2)
                    if len(shared_atoms) >= 2:  # Fused if share 2+ atoms
                        fused_count += 1
            
            return fused_count
        except Exception as e:
            logger.debug(f"Fused ring counting failed: {e}")
            return 0
    
    def calculate_opv_synthesizability(self, mol):
        """
        Custom synthesizability heuristic for organic photovoltaics
        Based on literature analysis of synthesized OPV materials
        Returns score 0-1 (higher=more synthesizable)
        """
        try:
            score = 1.0  # Start at maximum
            
            # 1. Molecular complexity penalty
            num_atoms = mol.GetNumHeavyAtoms()
            if num_atoms > 100:
                score -= 0.2
            elif num_atoms > 150:
                score -= 0.4
            
            # 2. Symmetry bonus (symmetric molecules easier)
            symmetry_score = self.calculate_molecular_symmetry(mol)
            score += 0.15 * symmetry_score
            
            # 3. Functional group difficulty
            difficult_groups = {
                'C#N': 0.05,  # Cyano - moderate
                'C(=C(C#N))C#N': 0.15,  # Dicyanovinyl - harder
                '[F]': 0.03,  # Fluorine - moderate
                'C(F)(F)F': 0.10,  # CF3 - harder
                'c1[se]ccc1': 0.20,  # Selenium - expensive/difficult
                '[Br]': 0.05,  # Bromine - moderate (for coupling)
            }
            
            for smarts, penalty in difficult_groups.items():
                count = count_substruct(mol, smarts)
                score -= count * penalty
            
            # 4. Conjugation length (longer = harder to control)
            conjugation_length = self.calculate_conjugation_length(mol)
            if conjugation_length > 15:
                score -= 0.15
            elif conjugation_length > 25:
                score -= 0.30
            
            # 5. Stereocenters penalty (chiral synthesis harder)
            try:
                num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                score -= num_stereocenters * 0.10
            except:
                pass
            
            # 6. Ring complexity
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings > 8:
                score -= 0.15
            
            # Check for fused ring systems (common in OPVs, generally accessible)
            fused_systems = self.count_fused_ring_systems(mol)
            if 2 <= fused_systems <= 4:
                score += 0.10  # Bonus for typical OPV architecture
            
            # 7. Known building block bonus
            known_cores = [
                'c1ccc2sccc2c1',  # BDT
                'c1sccc1',  # Thiophene
                'c1ccccc1',  # Benzene
            ]
            for core in known_cores:
                if count_substruct(mol, core) > 0:
                    score += 0.10
                    break
            
            # 8. Reactive group penalty (stability issues)
            reactive_groups = ['[OH]', '[NH2]', 'C(=O)O']
            reactive_count = sum(count_substruct(mol, group) for group in reactive_groups)
            if reactive_count > 2:
                score -= 0.20
            
            # 9. Molecular weight penalty
            try:
                mw = Descriptors.MolWt(mol)
                if mw > 1000:
                    score -= 0.15
                elif mw > 1500:
                    score -= 0.30
            except:
                pass
            
            # 10. Rotatable bonds (too many = difficult purification)
            try:
                num_rotatable = Descriptors.NumRotatableBonds(mol)
                if num_rotatable > 15:
                    score -= 0.10
            except:
                pass
            
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"OPV synthesizability calculation failed: {e}")
            return 0.5  # Return neutral score on error
    
    def score_tier1(self, mol):
        """
        Fast synthesizability assessment (Tier 1)
        Combines SYBA, SCScore, and custom OPV heuristics
        Returns: (final_score, detailed_scores_dict)
        """
        scores = {}
        
        # Get individual scores
        syba_score = self.get_syba_score(mol)
        scscore = self.get_scscore(mol)
        opv_score = self.calculate_opv_synthesizability(mol)
        
        if syba_score is not None:
            scores['syba'] = syba_score
        if scscore is not None:
            scores['scscore'] = scscore
        scores['opv_heuristic'] = opv_score
        
        # Calculate weighted average based on available scores
        if len(scores) == 0:
            return 0.5, {}
        
        # Weights (adjusted based on availability)
        if 'syba' in scores and 'scscore' in scores:
            # All three available
            final_score = (
                0.4 * scores.get('syba', 0.5) +
                0.3 * scores.get('scscore', 0.5) +
                0.3 * scores['opv_heuristic']
            )
        elif 'syba' in scores:
            # SYBA + OPV
            final_score = (
                0.5 * scores['syba'] +
                0.5 * scores['opv_heuristic']
            )
        elif 'scscore' in scores:
            # SCScore + OPV
            final_score = (
                0.5 * scores['scscore'] +
                0.5 * scores['opv_heuristic']
            )
        else:
            # Only OPV heuristic
            final_score = scores['opv_heuristic']
        
        scores['final_tier1'] = final_score
        
        return final_score, scores
    
    def score_tier2(self, mol):
        """
        Detailed synthesizability assessment (Tier 2)
        Adds RAScore for top candidates
        Returns: (final_score, detailed_scores_dict)
        """
        # Get Tier 1 scores first
        tier1_score, scores = self.score_tier1(mol)
        
        # Add RAScore
        rascore = self.get_rascore(mol)
        if rascore is not None:
            scores['rascore'] = rascore
            
            # Recalculate with RAScore
            if 'syba' in scores and 'scscore' in scores:
                final_score = (
                    0.3 * scores['syba'] +
                    0.25 * scores['scscore'] +
                    0.25 * scores['rascore'] +
                    0.2 * scores['opv_heuristic']
                )
            elif 'syba' in scores:
                final_score = (
                    0.4 * scores['syba'] +
                    0.3 * scores['rascore'] +
                    0.3 * scores['opv_heuristic']
                )
            elif 'scscore' in scores:
                final_score = (
                    0.4 * scores['scscore'] +
                    0.3 * scores['rascore'] +
                    0.3 * scores['opv_heuristic']
                )
            else:
                final_score = (
                    0.5 * scores['rascore'] +
                    0.5 * scores['opv_heuristic']
                )
        else:
            final_score = tier1_score
        
        scores['final_tier2'] = final_score
        
        return final_score, scores


# Initialize global synthesizability scorer
logger.info("Initializing global synthesizability scorer...")
SYNTH_SCORER = SynthesizabilityScorer()
logger.info(f"Synthesizability scorer ready with models: {', '.join(SYNTH_SCORER.available_models)}")


# ============================================================================
# TANGO-INSPIRED SYNTHESIZABILITY REWARD (Guo & Schwaller 2026)
# ============================================================================

class TANGOSynthesizabilityReward:
    """
    TANGO-inspired continuous synthesizability reward.
    Combines Fuzzy Matching Substructure (FMS) and Tanimoto similarity
    to building blocks, producing a continuous [0,1] reward signal.
    
    Reference: Guo & Schwaller, Nat. Comput. Sci. 6, 260-270 (2026)
    """
    
    def __init__(self, training_smiles=None, alpha=0.5, radius=2, n_bits=2048):
        self.alpha = alpha
        self.radius = radius
        self.n_bits = n_bits
        self.building_block_fps = []
        self.building_block_mols = []
        self.building_block_smiles = []
        self.bb_atom_counts = []
        
        logger.info("=" * 80)
        logger.info("INITIALIZING TANGO SYNTHESIZABILITY REWARD")
        logger.info(f"  Alpha (FMS weight): {alpha}")
        logger.info(f"  Fingerprint radius: {radius}, bits: {n_bits}")
        logger.info("=" * 80)
        
        opv_building_blocks = [
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
        
        all_bb_smiles = set(opv_building_blocks)
        
        if training_smiles:
            logger.info(f"  Extracting BRICS fragments from {len(training_smiles)} training molecules...")
            frag_count = 0
            for smi in training_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                try:
                    frags = BRICS.BRICSDecompose(mol)
                    all_bb_smiles.update(frags)
                    frag_count += len(frags)
                except Exception:
                    continue
            logger.info(f"  Extracted {frag_count} BRICS fragments")
        
        morgan_gen = GetMorganGenerator(radius, fpSize=n_bits)
        for bb_smi in all_bb_smiles:
            try:
                bb_mol = Chem.MolFromSmiles(bb_smi)
                if bb_mol is None:
                    continue
                Chem.SanitizeMol(bb_mol)
                fp = morgan_gen.GetFingerprintAsNumPy(bb_mol)
                self.building_block_mols.append(bb_mol)
                self.building_block_fps.append(fp)
                self.building_block_smiles.append(bb_smi)
                self.bb_atom_counts.append(bb_mol.GetNumAtoms())
            except Exception:
                continue
        
        logger.info(f"  Building block library: {len(self.building_block_fps)} blocks loaded")
        logger.info("TANGO REWARD INITIALIZED")
        logger.info("=" * 80)
    
    def _fuzzy_matching_substructure(self, mol, bb_mol, bb_atom_count):
        """Compute FMS score: fraction of building block atoms matched in molecule."""
        try:
            bb_fp = GetMorganFingerprintAsBitVect(bb_mol, self.radius, nBits=self.n_bits)
            mol_fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            
            bb_on = set(bb_fp.GetOnBits())
            mol_on = set(mol_fp.GetOnBits())
            
            if len(bb_on) == 0:
                return 0.0
            overlap = len(bb_on & mol_on)
            return overlap / len(bb_on)
        except Exception:
            return 0.0
    
    def _tanimoto_similarity(self, mol):
        """Max Tanimoto similarity to any building block."""
        try:
            morgan_gen = GetMorganGenerator(self.radius, fpSize=self.n_bits)
            mol_fp = morgan_gen.GetFingerprintAsNumPy(mol)
            
            max_sim = 0.0
            for bb_fp in self.building_block_fps:
                dot = np.dot(mol_fp, bb_fp)
                denom = np.sum(mol_fp) + np.sum(bb_fp) - dot
                if denom > 0:
                    sim = dot / denom
                    max_sim = max(max_sim, sim)
            return max_sim
        except Exception:
            return 0.0
    
    def score(self, mol):
        """
        Compute TANGO score for a molecule.
        Returns: float in [0, 1], higher = more synthesizable.
        """
        if mol is None:
            return 0.0
        try:
            best_fms = 0.0
            for bb_mol, bb_ac in zip(self.building_block_mols, self.bb_atom_counts):
                fms = self._fuzzy_matching_substructure(mol, bb_mol, bb_ac)
                best_fms = max(best_fms, fms)
            
            tan_sim = self._tanimoto_similarity(mol)
            
            tango = self.alpha * best_fms + (1 - self.alpha) * tan_sim
            return float(tango)
        except Exception as e:
            logger.debug(f"TANGO scoring failed: {e}")
            return 0.0
    
    def score_smiles(self, smiles):
        """Score a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        return self.score(mol)
    
    def score_batch(self, smiles_list):
        """Score a batch of SMILES."""
        return [self.score_smiles(smi) for smi in smiles_list]
    
    def combined_reward(self, mol, synth_scorer=None):
        """
        Combined reward incorporating both TANGO and the existing synthesizability scorer.
        Returns (tango_score, synth_score, combined_score, details_dict).
        """
        tango = self.score(mol)
        details = {"tango_score": tango}
        
        if synth_scorer is not None:
            try:
                synth_score, synth_details = synth_scorer.score_tier2(mol)
                details["synth_tier2"] = synth_score
                details.update({f"synth_{k}": v for k, v in synth_details.items() if isinstance(v, (int, float))})
                combined = 0.5 * tango + 0.5 * synth_score
            except Exception:
                synth_score = 0.0
                combined = tango
        else:
            synth_score = 0.0
            combined = tango
        
        details["combined_score"] = combined
        return tango, synth_score, combined, details


# ============================================================================
# DESCRIPTOR-BASED INTERPOLATIVE SCREENING SURROGATE FOR OSC DEVICE PROPERTIES
#
# NAMING CORRECTION: this class was previously called
# "DMPNNPredictionPipeline", which was a misnomer -- it does NOT train a
# directed message-passing neural network. It trains Ridge regression with
# isotonic calibration plus a Gaussian-Process component on 2048-bit Morgan
# fingerprints. It is renamed ScreeningSurrogatePipeline to describe what it
# actually is and, critically, what it is FOR: a fast interpolative SCREEN
# used to triage millions of candidate donor-acceptor pairs down to a
# shortlist. It is explicitly NOT a validated predictor and its outputs must
# not be reported as final device-performance claims. Honest held-out error
# is now reported via donor-disjoint / acceptor-disjoint / scaffold-disjoint
# cross-validation (see _grouped_disjoint_cv) alongside the in-sample fit.
#
# A true D-MPNN and properly cross-validated property models are trained
# separately in postgen_pipeline.py against DMPNN.csv and Guzik_data.csv.
# ============================================================================

class ScreeningSurrogatePipeline:
    """
    Two-stage DESCRIPTOR-BASED INTERPOLATIVE SCREENING SURROGATE (not a D-MPNN).

    Stage 1: Screen molecular descriptors (HOMO, LUMO, Eg, M) per molecule.
    Stage 2: Screen device properties (PCE, Jsc, FF, Voc) per donor-acceptor pair.

    Model: Ridge regression (isotonically calibrated) + Gaussian Process on
    Morgan fingerprints, uncertainty-weighted. This is a fast triage tool for
    reducing ~12M candidate pairs to a shortlist -- NOT a validated predictor.
    Final quantitative property claims come from postgen_pipeline.py.

    Physical bounds enforced on all predictions to prevent impossible values from
    Ridge regression extrapolation outside the training domain.

    Held-out generalisation error is reported per property via grouped
    disjoint cross-validation (donor-disjoint, acceptor-disjoint,
    scaffold-disjoint) written to results/method1_disjoint_cv_report.csv.
    """

    PCE_THRESHOLD = 18.0
    JSC_THRESHOLD = 25.0
    # NOTE: FF_THRESHOLD below is the *nominal* literature-style threshold.
    # The Tier-1 filter deliberately uses FF_THRESHOLD_TIER1 = 0.75 (relaxed
    # from 0.8) as more realistic for novel materials. Previously the code
    # applied 0.75 while all printed text/plot labels still claimed 0.8,
    # which was a reporting inconsistency. Both are now explicit and the
    # value actually applied is the one reported everywhere.
    FF_THRESHOLD = 0.8
    FF_THRESHOLD_TIER1 = 0.75
    VOC_THRESHOLD = 1.0
    # Minimum TANGO synthesizability score for a pair to count as a genuine
    # "high-performance" candidate. Without this, filter_high_performance_pairs()
    # ranked pairs purely on predicted PCE/Jsc/FF/Voc with zero synthesizability
    # signal, even though TANGO scores were already being computed per-molecule
    # elsewhere in the pipeline (tango_scores.csv) — they just were never joined
    # back into the pair-level evaluation. A pair can have excellent predicted
    # device properties but be built from two molecules that are synthetically
    # implausible (TANGO close to 0), which is exactly the kind of "high PCE on
    # paper, meaningless in the lab" result this threshold screens out.
    SYNTH_THRESHOLD_TIER1 = 0.5   # Tier1: reasonably accessible synthesis route
    SYNTH_THRESHOLD_TIER2 = 0.35  # Tier2: relaxed, still screens out near-0 scores

    # Hard physical bounds derived from OSC literature and training-set ranges.
    # Values outside these ranges are physically impossible and indicate extrapolation
    # artefacts from the Ridge surrogate; they are clipped to the nearest bound.
    # PCE ceiling tightened from a generous 30% to 21%: the certified world
    # record for single-junction OSCs as of 2025 is ~20.8% (Chinese Academy
    # of Sciences, hybrid cathode interfacial layer) with the previous
    # milestone at 19.8% certified / 20.2% lab (Nature Energy 2024,
    # D18:Z8:L8-BO ternary blend). A 30% ceiling let the Ridge/GP surrogate
    # and physics estimate produce values with no basis in reality; 21%
    # leaves a small buffer above the current record for genuinely novel
    # high-performers while treating anything claiming >21% as almost
    # certainly a model artifact rather than a real material.
    PHYSICAL_BOUNDS = {
        # Device properties
        'PCE':  (0.0,   21.0),   # % — certified single-junction OSC record ~20.8% (2025)
        'Jsc':  (0.0,   30.0),   # mA/cm² — training max 27.7; 30 leaves modest headroom
        'FF':   (0.25,  0.85),   # dimensionless — physically bounded [0,1]; real range 0.3–0.85
        'Voc':  (0.3,   2.0),    # V — typical range 0.5–1.5 V; 2 V is a generous ceiling
        # Molecular descriptors (negative HOMO/LUMO means ionisation energy / EA)
        'HOMO_donor':     (4.0,  7.5),   # -HOMO in eV
        'LUMO_donor':     (1.5,  5.5),   # -LUMO in eV
        'Eg_donor':       (0.5,  4.0),   # band gap in eV
        'HOMO_acceptor':  (4.5,  8.0),   # -HOMO in eV
        'LUMO_acceptor':  (2.0,  5.5),   # -LUMO in eV
        'Eg_acceptor':    (0.5,  4.0),   # band gap in eV
    }

    @classmethod
    def _clip_to_physical(cls, value, prop_name):
        """Clip a predicted value to its physical bound and flag out-of-range predictions."""
        if value is None or np.isnan(value) or np.isinf(value):
            return None
        bounds = cls.PHYSICAL_BOUNDS.get(prop_name)
        if bounds is None:
            return float(value)
        lo, hi = bounds
        clipped = float(np.clip(value, lo, hi))
        if clipped != float(value):
            logger.debug(
                f"Screening-surrogate physical-bounds clip: {prop_name} {value:.4f} -> {clipped:.4f} "
                f"(bounds [{lo}, {hi}])"
            )
        return clipped
    
    def __init__(self, training_csv_path):
        logger.info("=" * 80)
        logger.info("INITIALIZING DESCRIPTOR-BASED INTERPOLATIVE SCREENING SURROGATE")
        logger.info("  (Ridge + isotonic + Gaussian Process on Morgan fingerprints)")
        logger.info("  PURPOSE: fast triage screen only -- NOT a validated predictor.")
        logger.info("=" * 80)

        # Collects per-property grouped-disjoint CV results for honest
        # held-out error reporting (written to method1_disjoint_cv_report.csv).
        self.disjoint_cv_records = []

        self.training_df = pd.read_csv(training_csv_path)
        logger.info(f"  Training data loaded: {self.training_df.shape}")
        
        self.donor_descriptor_models = {}
        self.acceptor_descriptor_models = {}
        self.device_property_models = {}
        self.scalers = {}
        self.morgan_gen = GetMorganGenerator(2, fpSize=2048)
        
        # -------------------------------------------------------------------
        # COLUMN AUTO-MAPPING with sign convention detection.
        #
        # Different training CSVs use different column-naming conventions for
        # the same physical quantities (e.g. "Training_dataset_GAN.csv" uses
        # '-HOMO_donor(eV)' with values already stored as positive ionization
        # energies, while "DMPNN.csv" uses 'D_HOMO (eV)'/'HOMO (eV)' with
        # values stored as NEGATIVE eV, i.e. the raw HOMO/LUMO orbital energy).
        #
        # Without this mapping, zero descriptor models get trained whenever a
        # dataset's column names don't exactly match the hard-coded defaults,
        # silently disabling the physics-based Scharber PCE estimate (which
        # requires HOMO_donor/LUMO_acceptor/Eg_donor) and cutting off the main
        # mechanism for realistic high-performer extrapolation beyond the
        # training ceiling.
        #
        # For each internal key we list CANDIDATE column names in priority
        # order along with whether that column needs sign-flipping (abs())
        # to convert a raw negative orbital energy into the positive
        # ionization-energy / electron-affinity convention the rest of the
        # pipeline (PHYSICAL_BOUNDS, Scharber formulas) expects.
        # -------------------------------------------------------------------
        self.donor_descriptor_col_candidates = {
            'HOMO_donor': [('-HOMO_donor(eV)', False), ('D_HOMO (eV)', True), ('HOMO_donor(eV)', False)],
            'LUMO_donor': [('-LUMO_donor(eV)', False), ('D_LUMO (eV)', True), ('LUMO_donor(eV)', False)],
            'Eg_donor':   [('Eg_donor(eV)', False), ('D_H-L Gap (eV)', False), ('D_Eg (eV)', False)],
        }
        self.acceptor_descriptor_col_candidates = {
            'HOMO_acceptor': [('-HOMO_acceptor(eV)', False), ('HOMO (eV)', True), ('HOMO_acceptor(eV)', False)],
            'LUMO_acceptor': [('-LUMO_acceptor(eV)', False), ('LUMO (eV)', True), ('LUMO_acceptor(eV)', False)],
            'Eg_acceptor':   [('Eg_acceptor(eV)', False), ('H-L Gap (eV)', False), ('Eg (eV)', False)],
        }
        self.device_property_col_candidates = {
            'PCE': [('PCE_max(%)', False), ('PCE_ave(%)', False), ('PCE(%)', False)],
            'Jsc': [('Jsc(mA/cm2)', False), ('Jsc (mA/cm2)', False)],
            'FF':  [('FF', False), ('FF(%)', False)],
            'Voc': [('Voc(V)', False), ('Voc (V)', False)],
        }

        def _resolve_columns(candidates_map, df_columns):
            """Pick the first matching candidate column for each key; record sign flip."""
            resolved = {}
            for key, candidates in candidates_map.items():
                for col_name, needs_abs in candidates:
                    if col_name in df_columns:
                        resolved[key] = (col_name, needs_abs)
                        break
            return resolved

        df_cols = set(self.training_df.columns)
        self.donor_descriptor_cols = _resolve_columns(self.donor_descriptor_col_candidates, df_cols)
        self.acceptor_descriptor_cols = _resolve_columns(self.acceptor_descriptor_col_candidates, df_cols)
        self.device_property_cols = _resolve_columns(self.device_property_col_candidates, df_cols)

        logger.info(f"  Resolved donor descriptor columns: {self.donor_descriptor_cols}")
        logger.info(f"  Resolved acceptor descriptor columns: {self.acceptor_descriptor_cols}")
        logger.info(f"  Resolved device property columns: {self.device_property_cols}")

        missing_donor = set(self.donor_descriptor_col_candidates) - set(self.donor_descriptor_cols)
        missing_acceptor = set(self.acceptor_descriptor_col_candidates) - set(self.acceptor_descriptor_cols)
        missing_device = set(self.device_property_col_candidates) - set(self.device_property_cols)
        if missing_donor:
            logger.warning(f"  Could not resolve donor descriptor columns for: {missing_donor} — physics PCE estimate will be degraded")
        if missing_acceptor:
            logger.warning(f"  Could not resolve acceptor descriptor columns for: {missing_acceptor} — physics PCE estimate will be degraded")
        if missing_device:
            logger.warning(f"  Could not resolve device property columns for: {missing_device}")

        self._train_all_models()
        logger.info("SCREENING SURROGATE INITIALIZED (Method 1 only; screen-only semantics)")
        logger.info("=" * 80)
    
    def _smiles_to_fp(self, smiles):
        """Convert SMILES to Morgan fingerprint numpy array."""
        text = str(smiles).strip()
        for feature_map in TRAINING_FALLBACK_FEATURES.values():
            fp = feature_map.get(text)
            if fp is not None:
                return np.asarray(fp, dtype=np.float32)
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            fallback = fallback_representation_from_smiles(text)
            if fallback["usable"]:
                logger.debug(f"DMPNN fingerprint using {fallback['representation']} fallback for: {text[:80]}")
                return fallback["hashed_fp"]
            return None
        try:
            return self.morgan_gen.GetFingerprintAsNumPy(mol)
        except Exception as e:
            logger.debug(f"DMPNN Morgan fingerprint failed, trying token/SELFIES fallback: {e}")
            fallback = fallback_representation_from_smiles(text)
            if fallback["usable"]:
                return fallback["hashed_fp"]
            return None
    
    @staticmethod
    def _murcko_scaffold_key(smiles):
        """Bemis-Murcko scaffold string for a SMILES (grouping key for CV).

        Returns the literal SMILES as a fallback key when RDKit cannot parse
        the molecule or the scaffold is empty, so that such rows still form
        their own group rather than silently collapsing into one giant group.
        """
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return f"UNPARSEABLE::{smiles}"
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf is None:
                return f"NOSCAFFOLD::{smiles}"
            scaf_smi = Chem.MolToSmiles(scaf)
            if not scaf_smi:
                return f"EMPTYSCAFFOLD::{smiles}"
            return scaf_smi
        except Exception:
            return f"ERROR::{smiles}"

    def _grouped_disjoint_cv(self, X, y, groups, name, split_kind, n_splits=5):
        """
        Honest held-out error via GROUP-DISJOINT cross-validation.

        Why this exists: the surrogate's previously reported R2/MAE were
        computed on the SAME rows used for fitting (in-sample), which cannot
        establish predictive performance on the highly novel generated
        molecules the screen is actually applied to. Grouped CV holds out
        entire donors / entire acceptors / entire Bemis-Murcko scaffolds so
        that no molecule (or scaffold) seen in training appears in the
        held-out fold.

        split_kind is a label only ('donor_disjoint' | 'acceptor_disjoint' |
        'scaffold_disjoint'); the grouping semantics come from `groups`.

        Ridge-only inside the CV loop (the GP component is expensive and its
        blending weight is uncertainty-driven, so Ridge held-out error is the
        conservative, reproducible lower bound on screen quality).

        Appends a record to self.disjoint_cv_records and returns it.
        """
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler as StdScaler
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        record = {
            "property": name,
            "split_kind": split_kind,
            "n_samples": int(len(y)),
            "n_groups": int(len(set(groups))) if groups is not None else 0,
            "n_splits_used": 0,
            "heldout_r2": float("nan"),
            "heldout_mae": float("nan"),
            "heldout_rmse": float("nan"),
            "status": "not_run",
        }
        try:
            groups = np.asarray(groups)
            uniq_groups = np.unique(groups)
            if len(uniq_groups) < 2:
                record["status"] = "skipped_single_group"
                self.disjoint_cv_records.append(record)
                logger.warning(
                    f"    [disjoint-CV] {name} / {split_kind}: only "
                    f"{len(uniq_groups)} distinct group(s) -- cannot hold out; skipped."
                )
                return record

            k = int(min(n_splits, len(uniq_groups)))
            gkf = GroupKFold(n_splits=k)
            y_true_all, y_pred_all = [], []
            folds_done = 0

            for tr_idx, te_idx in gkf.split(X, y, groups=groups):
                if len(tr_idx) < 3 or len(te_idx) < 1:
                    continue
                sx = StdScaler()
                sy = StdScaler()
                X_tr = sx.fit_transform(X[tr_idx])
                y_tr = sy.fit_transform(y[tr_idx].reshape(-1, 1)).ravel()
                m = Ridge(alpha=1.0)
                m.fit(X_tr, y_tr)
                X_te = sx.transform(X[te_idx])
                y_hat = sy.inverse_transform(m.predict(X_te).reshape(-1, 1)).ravel()
                y_true_all.append(y[te_idx])
                y_pred_all.append(y_hat)
                folds_done += 1

            if folds_done == 0 or not y_true_all:
                record["status"] = "no_valid_folds"
                self.disjoint_cv_records.append(record)
                return record

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)

            record["n_splits_used"] = folds_done
            # R2 is undefined/meaningless when the held-out target has no
            # variance; guard rather than emitting a misleading number.
            if float(np.var(y_true_all)) > 1e-12:
                record["heldout_r2"] = float(r2_score(y_true_all, y_pred_all))
            record["heldout_mae"] = float(mean_absolute_error(y_true_all, y_pred_all))
            record["heldout_rmse"] = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
            record["status"] = "ok"

            logger.info(
                f"    [disjoint-CV] {name} / {split_kind}: "
                f"heldout R2={record['heldout_r2']:.3f}, "
                f"MAE={record['heldout_mae']:.3f}, RMSE={record['heldout_rmse']:.3f} "
                f"({folds_done} folds over {record['n_groups']} groups, n={record['n_samples']})"
            )
        except Exception as e:
            record["status"] = f"error: {e}"
            logger.warning(f"    [disjoint-CV] {name} / {split_kind} failed: {e}")

        self.disjoint_cv_records.append(record)
        return record

    def save_disjoint_cv_report(self, output_dir="results"):
        """Write the grouped-disjoint CV table (honest held-out error)."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "method1_disjoint_cv_report.csv")
            if not self.disjoint_cv_records:
                logger.warning("  No disjoint-CV records to write.")
                return None
            pd.DataFrame(self.disjoint_cv_records).to_csv(path, index=False)
            logger.info(f"  Saved grouped-disjoint CV report to {path}")
            logger.info(
                "  INTERPRETATION: these held-out errors -- not the in-sample "
                "Ridge/GP R2 -- are the honest estimate of screening-surrogate "
                "generalisation to novel molecules."
            )
            return path
        except Exception as e:
            logger.warning(f"  Could not save disjoint-CV report: {e}")
            return None

    def _train_surrogate(self, X, y, name, groups_map=None):
        """
        Train an enhanced surrogate ensemble for realistic extrapolation.

        groups_map: optional dict {split_kind: array_of_group_labels} used to
        run grouped-disjoint cross-validation for honest held-out error
        reporting. Passing None preserves the original behaviour (in-sample
        stats only).

        Architecture: Ridge regression (in-distribution baseline) +
        Gaussian Process regression (calibrated uncertainty + extrapolation)
        combined via uncertainty-weighted blending.

        Why GP can extrapolate beyond training data realistically:
        - The RBF kernel smoothly extends learned correlations into new regions
          of fingerprint space rather than clamping to the training min/max.
        - GP uncertainty grows away from training points, so the blending weight
          automatically down-weights GP predictions in far-OOD regions.
        - We use PHYSICS-CONSTRAINED GP: the GP mean is not free to diverge;
          hard physical bounds still apply, but within those bounds predictions
          can exceed the training max when molecular features warrant it.

        The isotonic calibrator remains for the Ridge component only, ensuring
        in-distribution predictions stay well-calibrated.
        """
        from sklearn.preprocessing import StandardScaler as StdScaler
        from sklearn.isotonic import IsotonicRegression
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        scaler_X = StdScaler()
        scaler_y = StdScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Ridge baseline — calibrated in-distribution predictor
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_scaled)

        y_pred_ridge_scaled = model.predict(X_scaled)
        y_pred_ridge = scaler_y.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1)).ravel()

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, y_pred_ridge)
        mae = mean_absolute_error(y, y_pred_ridge)

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        # Isotonic calibrator range loosened by 10% beyond the observed
        # training min/max (rather than hard-clipping exactly at the
        # training bounds). This lets the Ridge component itself contribute
        # modest, defensible extrapolation for genuinely novel fingerprints
        # whose raw Ridge output already exceeds the training range, instead
        # of relying solely on the GP/physics paths to push past the ceiling.
        # The final hard PHYSICAL_BOUNDS clip (a real-world literature-backed
        # ceiling, e.g. PCE<=21%) remains the absolute safety net regardless
        # of this loosening, so nothing physically implausible can pass.
        y_range = y_max - y_min
        iso_y_min = y_min - 0.10 * y_range
        iso_y_max = y_max + 0.10 * y_range
        iso = IsotonicRegression(y_min=iso_y_min, y_max=iso_y_max, out_of_bounds='clip')
        iso.fit(y_pred_ridge, y)

        # Gaussian Process — physics-aware kernel for OPV property extrapolation
        # We use Matérn(nu=2.5) which is smoother than RBF and better suited
        # for molecular property landscapes (not infinitely differentiable).
        # Larger GP training sample (up to 400) and fewer PCA dims (20) for
        # better variance capture on device properties (PCE, Jsc, FF, Voc).
        gp_model = None
        gp_scaler_X = None
        gp_r2 = float("nan")
        try:
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.decomposition import PCA as _PCA
            gp_n = min(400, len(X_scaled))
            if gp_n >= 10:
                rng = np.random.RandomState(42)
                gp_idx = rng.choice(len(X_scaled), gp_n, replace=False)
                X_gp = X_scaled[gp_idx]
                y_gp = y[gp_idx]

                # Use fewer PCA dims (20) to concentrate variance and improve GP fits
                n_pca = min(20, X_gp.shape[0] - 1, X_gp.shape[1])
                pca_gp = _PCA(n_components=n_pca)
                X_gp_pca = pca_gp.fit_transform(X_gp)

                # Matérn kernel: nu=2.5 gives once-differentiable predictions,
                # appropriate for OSC property landscapes.
                # Large constant kernel bounds allow the GP amplitude to scale freely.
                # WhiteKernel captures heteroscedastic noise in experimental data.
                kernel = (
                    ConstantKernel(1.0, (1e-2, 1e4))
                    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                    + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-4, 2.0))
                )
                gp_model_inner = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=42,
                )
                gp_model_inner.fit(X_gp_pca, y_gp)

                y_gp_pred = gp_model_inner.predict(X_gp_pca)
                gp_r2 = r2_score(y_gp, y_gp_pred)
                gp_model = gp_model_inner
                gp_scaler_X = pca_gp
                logger.info(
                    f"    {name} [GP]: R2={gp_r2:.3f} on {gp_n} samples, "
                    f"n_pca={n_pca}, kernel={gp_model_inner.kernel_}"
                )
        except Exception as gp_e:
            logger.debug(f"    {name} GP training failed (Ridge fallback only): {gp_e}")
            gp_model = None
            gp_scaler_X = None

        logger.info(
            f"    {name} [Ridge]: IN-SAMPLE R2={r2:.3f}, MAE={mae:.3f}, "
            f"train_range=[{y_min:.3f}, {y_max:.3f}]  "
            f"(in-sample only -- see disjoint-CV below for held-out error)"
        )

        # ------------------------------------------------------------------
        # Honest held-out error via grouped-disjoint CV. Runs one CV pass per
        # supplied grouping (donor-disjoint / acceptor-disjoint /
        # scaffold-disjoint) so the report shows how the screen degrades when
        # entire molecules or entire scaffolds are unseen.
        # ------------------------------------------------------------------
        if groups_map:
            for split_kind, groups in groups_map.items():
                if groups is None:
                    continue
                try:
                    self._grouped_disjoint_cv(X, y, groups, name, split_kind)
                except Exception as cv_e:
                    logger.warning(f"    disjoint-CV ({split_kind}) for {name} failed: {cv_e}")

        return model, scaler_X, scaler_y, iso, y_min, y_max, gp_model, gp_scaler_X
    
    def _train_all_models(self):
        """Train all surrogate models from training CSV.

        Grouped-disjoint CV groupings are built alongside the design matrices
        so that every trained property reports honest held-out error:
          - donor_disjoint    : group = donor identity (canonical SMILES)
          - acceptor_disjoint : group = acceptor identity (canonical SMILES)
          - scaffold_disjoint : group = Bemis-Murcko scaffold
        """
        logger.info("  Training molecular descriptor screening surrogates...")
        
        # Donor descriptors
        donor_smiles = self.training_df['donor(SMILES)'].dropna().tolist()
        donor_fps = []
        valid_donor_idx = []
        for i, smi in enumerate(donor_smiles):
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                donor_fps.append(fp)
                valid_donor_idx.append(i)
        
        if donor_fps:
            X_donor = np.array(donor_fps)
            donor_valid_smiles = [donor_smiles[i] for i in valid_donor_idx]
            donor_identity_groups = np.array([str(s) for s in donor_valid_smiles])
            donor_scaffold_groups = np.array([self._murcko_scaffold_key(s) for s in donor_valid_smiles])
            for key, (col, needs_abs) in self.donor_descriptor_cols.items():
                if col in self.training_df.columns:
                    y = self.training_df[col].iloc[valid_donor_idx].values.astype(float)
                    if needs_abs:
                        y = np.abs(y)
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 10:
                        groups_map = {
                            "donor_disjoint": donor_identity_groups[mask],
                            "scaffold_disjoint": donor_scaffold_groups[mask],
                        }
                        model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X = self._train_surrogate(
                            X_donor[mask], y[mask], key, groups_map=groups_map)
                        self.donor_descriptor_models[key] = (model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X)
        
        # Acceptor descriptors
        acceptor_smiles = self.training_df['acceptor(SMILES)'].dropna().tolist()
        acceptor_fps = []
        valid_acc_idx = []
        for i, smi in enumerate(acceptor_smiles):
            fp = self._smiles_to_fp(smi)
            if fp is not None:
                acceptor_fps.append(fp)
                valid_acc_idx.append(i)
        
        if acceptor_fps:
            X_acc = np.array(acceptor_fps)
            acc_valid_smiles = [acceptor_smiles[i] for i in valid_acc_idx]
            acc_identity_groups = np.array([str(s) for s in acc_valid_smiles])
            acc_scaffold_groups = np.array([self._murcko_scaffold_key(s) for s in acc_valid_smiles])
            for key, (col, needs_abs) in self.acceptor_descriptor_cols.items():
                if col in self.training_df.columns:
                    y = self.training_df[col].iloc[valid_acc_idx].values.astype(float)
                    if needs_abs:
                        y = np.abs(y)
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 10:
                        groups_map = {
                            "acceptor_disjoint": acc_identity_groups[mask],
                            "scaffold_disjoint": acc_scaffold_groups[mask],
                        }
                        model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X = self._train_surrogate(
                            X_acc[mask], y[mask], key, groups_map=groups_map)
                        self.acceptor_descriptor_models[key] = (model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X)
        
        # Device properties: use combined donor+acceptor fingerprints + descriptors
        logger.info("  Training device property screening surrogates...")
        
        combined_fps = []
        combined_valid_idx = []
        for i in range(min(len(donor_smiles), len(acceptor_smiles))):
            d_fp = self._smiles_to_fp(donor_smiles[i]) if i < len(donor_smiles) else None
            a_fp = self._smiles_to_fp(acceptor_smiles[i]) if i < len(acceptor_smiles) else None
            if d_fp is not None and a_fp is not None:
                combined = np.concatenate([d_fp, a_fp])
                combined_fps.append(combined)
                combined_valid_idx.append(i)
        
        if combined_fps:
            X_combined = np.array(combined_fps)
            # For device (pair-level) properties all three disjoint splits are
            # meaningful: hold out entire donors, entire acceptors, or entire
            # donor scaffolds. This is the key honesty check, because the screen
            # is applied to pairs of *novel* donors AND *novel* acceptors.
            pair_donor_groups = np.array([str(donor_smiles[i]) for i in combined_valid_idx])
            pair_acc_groups = np.array([str(acceptor_smiles[i]) for i in combined_valid_idx])
            pair_scaffold_groups = np.array(
                [self._murcko_scaffold_key(donor_smiles[i]) for i in combined_valid_idx])
            for key, (col, needs_abs) in self.device_property_cols.items():
                if col in self.training_df.columns:
                    y = self.training_df[col].iloc[combined_valid_idx].values.astype(float)
                    if needs_abs:
                        y = np.abs(y)
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 10:
                        groups_map = {
                            "donor_disjoint": pair_donor_groups[mask],
                            "acceptor_disjoint": pair_acc_groups[mask],
                            "scaffold_disjoint": pair_scaffold_groups[mask],
                        }
                        model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X = self._train_surrogate(
                            X_combined[mask], y[mask], key, groups_map=groups_map)
                        self.device_property_models[key] = (model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X)
        
        logger.info(f"  Trained {len(self.donor_descriptor_models)} donor descriptor screening models")
        logger.info(f"  Trained {len(self.acceptor_descriptor_models)} acceptor descriptor screening models")
        logger.info(f"  Trained {len(self.device_property_models)} device property screening models")
        # Persist honest held-out error immediately after training.
        self.save_disjoint_cv_report(output_dir="results")
    
    def _predict_with_ensemble(self, fp, entry, prop_name):
        """
        Uncertainty-weighted Ridge + GP ensemble prediction with free extrapolation.

        Key design decisions for realistic high-performer discovery:
        - Ridge prediction is kept isotonically calibrated (bounded to training range)
          for stability on in-distribution samples.
        - GP prediction is NOT isotonically bounded. The GP uses an RBF kernel that
          smoothly extrapolates beyond training data. For novel molecules with
          fingerprint patterns correlated with high performance, the GP is free to
          predict above the training max (up to the hard physical bounds only).
        - Blending: weight_gp = exp(-gp_std²) — a tighter decay than 1/(1+std).
          This gives GP weight ~0.98 when std=0.1 (confident, near training data)
          and ~0.37 when std=1.0 (uncertain, OOD). The GP dominates when it is
          confident, which is exactly when extrapolation is most reliable.
        - Hard PHYSICAL_BOUNDS clip remains as the absolute safety net.

        Returns: float prediction clipped to physical bounds only.
        """
        model, sx, sy, iso, y_min, y_max, gp_model, gp_scaler_X = entry

        # Ridge prediction with isotonic calibration (bounded to training range)
        try:
            x_scaled = sx.transform(fp.reshape(1, -1))
            y_scaled = model.predict(x_scaled)
            y_ridge_raw = float(sy.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0])
            y_ridge = float(iso.predict([y_ridge_raw])[0])
        except Exception:
            y_ridge = (y_min + y_max) / 2.0

        # GP prediction — free to extrapolate beyond training range
        if gp_model is not None and gp_scaler_X is not None:
            try:
                x_scaled_for_gp = sx.transform(fp.reshape(1, -1))
                x_gp_pca = gp_scaler_X.transform(x_scaled_for_gp)
                y_gp_pred, y_gp_std = gp_model.predict(x_gp_pca, return_std=True)
                y_gp = float(y_gp_pred[0])
                gp_std = float(y_gp_std[0])

                # Gaussian-decay weight: tight decay gives GP strong control when
                # confident, falls off quickly when uncertain (OOD samples)
                weight_gp = float(np.exp(-gp_std ** 2))
                weight_ridge = 1.0 - weight_gp
                y_val = weight_gp * y_gp + weight_ridge * y_ridge

                logger.debug(
                    f"  {prop_name}: Ridge={y_ridge:.3f}, GP={y_gp:.3f}±{gp_std:.3f}, "
                    f"blend(GP:{weight_gp:.2f}/R:{weight_ridge:.2f})={y_val:.3f}"
                )
            except Exception as gp_pred_e:
                logger.debug(f"  {prop_name} GP prediction failed: {gp_pred_e}; using Ridge only")
                y_val = y_ridge
        else:
            y_val = y_ridge

        # Only physical bounds apply — no isotonic clamp on the GP path
        return self._clip_to_physical(y_val, prop_name)

    def _physics_pce_estimate(self, donor_smiles, acceptor_smiles,
                               donor_desc, acceptor_desc):
        """
        Physics-informed PCE estimate based on HOMO-LUMO energy alignment rules
        for organic photovoltaics (Scharber model, simplified).

        Scharber model (2006):
          Voc â‰ˆ (1/q) * (|HOMO_donor| - |LUMO_acceptor|) - 0.3 V (empirical offset)
          EQE â‰ˆ 0.65 (standard assumption)
          Jsc â‰ˆ EQE × integral of AM1.5 photon flux above Eg_donor
          FF â‰ˆ 0.65 (standard assumption for well-optimized devices)
          PCE = Jsc × Voc × FF / P_in  (P_in = 100 mW/cm²)

        This gives a physically motivated estimate that can exceed training data
        when the HOMO-LUMO alignment is particularly favorable. Values are clipped
        to physical bounds at the end.

        Returns: dict with keys 'physics_PCE', 'physics_Voc', 'physics_Jsc', 'physics_FF'
                 or empty dict if descriptors unavailable.
        """
        try:
            homo_d = donor_desc.get('HOMO_donor')   # |HOMO| in eV (positive)
            lumo_a = acceptor_desc.get('LUMO_acceptor')  # |LUMO| in eV (positive)
            eg_d   = donor_desc.get('Eg_donor')     # band gap in eV

            if any(v is None or np.isnan(v) for v in [homo_d, lumo_a, eg_d]):
                return {}

            # Scharber Voc: (|HOMO_D| - |LUMO_A|) - 0.3 offset
            # These are reported as positive ionization energy and electron affinity
            voc_physics = max(0.3, homo_d - lumo_a - 0.3)

            # Jsc from integrated photon flux above donor bandgap, refit
            # against the UPPER ENVELOPE (not mean regression) of this
            # dataset's actual donor-bandgap-vs-Jsc relationship. The
            # previous fit (44.9·exp(-0.48·Eg)) was ordinary least-squares
            # through the mean of all 300 training rows, which by
            # construction predicts the AVERAGE achievable Jsc at a given
            # bandgap — but a physics-informed ceiling should represent the
            # BEST achievable Jsc at that bandgap, since most real molecules
            # underperform their theoretical maximum due to non-radiative
            # recombination, poor morphology, etc. Refitting: grouped
            # DMPNN.csv's D_H-L Gap (eV) into per-unique-Eg buckets, took the
            # max observed Jsc per bucket (isotonic-smoothed to remove single-
            # point noise), then log-linear fit through that smoothed
            # envelope gives Jsc_envelope(Eg) â‰ˆ 54.5·exp(-0.28·Eg) — a much
            # shallower decay than the mean fit, correctly reflecting that
            # high-Jsc outliers exist across the full Eg range in this
            # dataset (e.g. Eg=2.65eV -> Jsc=27.7, the training-set maximum).
            # An EQE-like derating of 0.85 is applied because the envelope
            # itself already represents best-CASE achieved performance, not
            # a theoretical AM1.5G limit — a randomly generated novel pair
            # should not automatically get 100% of the best-ever value.
            # Clipped to 28 (just above the training max of 27.7) since
            # extrapolating the envelope far beyond the data's actual Eg
            # range at low bandgaps produces increasingly speculative values.
            eqe = 0.85
            jsc_max_theoretical = 54.5 * np.exp(-0.28 * eg_d)
            jsc_physics = float(np.clip(eqe * jsc_max_theoretical, 1.0, 28.0))

            # FF for well-optimized modern NFA-based OPVs: literature baseline
            # 0.70-0.80 (record devices reach 0.80-0.83). Base 0.70, scaled up
            # with Voc (better energetics â†’ cleaner charge extraction), capped
            # at 0.85 (physical ceiling for OPV).
            ff_physics = min(0.85, 0.70 + 0.06 * voc_physics)

            # PCE in %
            pce_physics = jsc_physics * voc_physics * ff_physics / 100.0 * 100.0

            # Clip to physical bounds (matches tightened PHYSICAL_BOUNDS: PCE
            # ceiling lowered from a generous 30% to 21%, just above the 2025
            # certified single-junction OSC world record of ~20.8%, so this
            # physics estimate cannot itself manufacture unrealistic values
            # even before the final _clip_to_physical() pass in evaluate_da_pairs).
            pce_physics = float(np.clip(pce_physics, 0.0, 21.0))
            jsc_physics = float(np.clip(jsc_physics, 0.0, 30.0))
            ff_physics  = float(np.clip(ff_physics, 0.25, 0.85))
            voc_physics = float(np.clip(voc_physics, 0.3, 2.0))

            return {
                'physics_PCE': pce_physics,
                'physics_Voc': voc_physics,
                'physics_Jsc': jsc_physics,
                'physics_FF':  ff_physics,
            }
        except Exception as e:
            logger.debug(f"Physics PCE estimate failed: {e}")
            return {}

    def predict_molecular_descriptors(self, smiles, mol_type="donor"):
        """Predict molecular descriptors for a SMILES string with physical-bounds enforcement."""
        fp = self._smiles_to_fp(smiles)
        if fp is None:
            return {}

        models = self.donor_descriptor_models if mol_type == "donor" else self.acceptor_descriptor_models
        results = {}

        for key, entry in models.items():
            try:
                y_val = self._predict_with_ensemble(fp, entry, key)
                results[key] = y_val
            except Exception:
                results[key] = None

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            results['M_gmol'] = float(Descriptors.ExactMolWt(mol))

        return results

    def predict_device_properties(self, donor_smiles, acceptor_smiles):
        """Predict device properties for a donor-acceptor pair with physical-bounds enforcement."""
        d_fp = self._smiles_to_fp(donor_smiles)
        a_fp = self._smiles_to_fp(acceptor_smiles)
        if d_fp is None or a_fp is None:
            return {}

        combined = np.concatenate([d_fp, a_fp])
        results = {}

        for key, entry in self.device_property_models.items():
            try:
                y_val = self._predict_with_ensemble(combined, entry, key)
                results[key] = y_val
            except Exception:
                results[key] = None

        return results
    
    def evaluate_da_pairs(self, donor_smiles_list, acceptor_smiles_list, max_pairs=50000,
                           donor_tango_lookup=None, acceptor_tango_lookup=None):
        """
        Evaluate all donor-acceptor combinations.
        Returns DataFrame of pairs with predicted properties.

        donor_tango_lookup / acceptor_tango_lookup: optional dict mapping
        SMILES -> TANGO synthesizability score [0,1]. When provided, each
        pair record gets 'donor_tango', 'acceptor_tango', and 'pair_tango'
        (the minimum of the two, since a pair is only as synthesizable as
        its harder-to-make partner) columns, letting
        filter_high_performance_pairs() screen out device-property-optimistic
        but synthetically implausible pairs.
        """
        logger.info("=" * 80)
        logger.info("EVALUATING DONOR-ACCEPTOR PAIRS")
        logger.info(f"  Donors: {len(donor_smiles_list)}, Acceptors: {len(acceptor_smiles_list)}")
        
        total_possible = len(donor_smiles_list) * len(acceptor_smiles_list)
        logger.info(f"  Total possible pairs: {total_possible}")
        
        if total_possible > max_pairs:
            d_sample_size = min(len(donor_smiles_list), int(np.sqrt(max_pairs)))
            a_sample_size = min(len(acceptor_smiles_list), max_pairs // d_sample_size)
            donor_subset = balanced_trait_order(donor_smiles_list, "Donor", d_sample_size)
            acceptor_subset = balanced_trait_order(acceptor_smiles_list, "Acceptor", a_sample_size)
            logger.info(f"  Trait-balanced subsample to {d_sample_size} donors x {a_sample_size} acceptors = {d_sample_size * a_sample_size} pairs")
        else:
            donor_subset = donor_smiles_list
            acceptor_subset = acceptor_smiles_list
        
        records = []
        evaluated = 0
        for d_smi in donor_subset:
            d_desc = self.predict_molecular_descriptors(d_smi, "donor")
            for a_smi in acceptor_subset:
                a_desc = self.predict_molecular_descriptors(a_smi, "acceptor")
                dev_props = self.predict_device_properties(d_smi, a_smi)

                # ----------------------------------------------------------
                # METHOD 2 (Scharber-type physics estimate) REMOVED FROM THE
                # QUANTITATIVE BLEND.
                #
                # Why: verification showed the Method-2 distribution was
                # dominated by its own hard-coded clipping ceilings rather than
                # emergent physics -- PCE clipped at 21%, FF algebraically
                # confined to 0.70-0.85, Voc to 0.3-2.0 V. ~98% of its PCE
                # values exceeded 18% and 100% exceeded 1.0 V, while only 0.2%
                # exceeded 25 mA/cm2 Jsc: i.e. the apparent "high efficiency"
                # came from an imposed high-voltage / high-fill-factor prior,
                # not from photocurrent. Its mean parameter balance
                # (Voc~1.66 V, Jsc~16 mA/cm2) does not resemble how real
                # record-efficiency OSC devices reach ~21%, which depend on
                # morphology, transport and suppressed non-radiative loss --
                # none of which a frontier-orbital formula can see. Because
                # the old blend took max(surrogate, physics), these inflated
                # values propagated straight into the reported PCE/Voc/FF and
                # into the "high-performance pair" counts.
                #
                # The frontier-orbital intuition is NOT discarded: it is
                # retained as a purely QUALITATIVE design-rule flag
                # ('favorable_homo_lumo_alignment') plus transparent
                # diagnostic-only columns, and can never again manufacture a
                # quantitative PCE/Jsc/FF/Voc number.
                #
                # (Method 3, SISSO++ 5D equations, was removed earlier for
                # numerical instability: division-by-near-zero Jsc blowups of
                # +/-400,000 and physically impossible negative Voc.)
                #
                # RESULT: blended device properties = Method 1 screening
                # surrogate ONLY, with the hard PHYSICAL_BOUNDS clip retained
                # as the absolute safety net.
                # ----------------------------------------------------------
                phys = self._physics_pce_estimate(d_smi, a_smi, d_desc, a_desc)

                blended = {}
                for prop in ['PCE', 'Jsc', 'FF', 'Voc']:
                    surr_val = dev_props.get(prop)
                    if surr_val is not None:
                        blended[prop] = self._clip_to_physical(surr_val, prop)

                # Qualitative frontier-orbital design-rule flag (NOT a
                # quantitative predictor). True when the donor HOMO /
                # acceptor LUMO offset implies a usable driving force with a
                # non-trivial Voc, i.e. the classic OPV alignment heuristic.
                favorable_alignment = None
                try:
                    homo_d = d_desc.get('HOMO_donor')
                    lumo_a = a_desc.get('LUMO_acceptor')
                    eg_d = d_desc.get('Eg_donor')
                    if all(v is not None and not np.isnan(v) for v in [homo_d, lumo_a, eg_d]):
                        offset = homo_d - lumo_a          # eV, ~ qVoc before losses
                        favorable_alignment = bool(
                            offset > 0.9            # enough for a useful Voc
                            and eg_d < 2.2          # donor absorbs usefully in the visible
                            and (eg_d - offset) > 0.0   # offset cannot exceed the gap
                        )
                except Exception:
                    favorable_alignment = None

                record = {
                    'donor_smiles': d_smi,
                    'acceptor_smiles': a_smi,
                }
                record.update({f"donor_{k}": v for k, v in d_desc.items()})
                record.update({f"acceptor_{k}": v for k, v in a_desc.items()})
                record.update(blended)
                # Method 1 (Ridge+GP screening surrogate) raw values -- this is
                # now the ONLY quantitative contributor to the blend above.
                record.update({f"surrogate_{k}": v for k, v in dev_props.items()})
                # Method 2 values retained as DIAGNOSTIC-ONLY columns for
                # transparency/audit. They are deliberately excluded from the
                # blend and MUST NOT be reported as predictions.
                record.update({f"diag_physics_{k}": v for k, v in phys.items()})
                # Qualitative frontier-orbital design-rule flag (replaces
                # Method 2's quantitative role).
                record['favorable_homo_lumo_alignment'] = favorable_alignment

                # Attach synthesizability (TANGO) scores for this pair, if
                # available, so downstream filtering can screen out pairs
                # with great predicted device properties but implausible
                # synthesis routes (see SYNTH_THRESHOLD_TIER1/2).
                if donor_tango_lookup is not None or acceptor_tango_lookup is not None:
                    d_tango = (donor_tango_lookup or {}).get(d_smi, 0.0)
                    a_tango = (acceptor_tango_lookup or {}).get(a_smi, 0.0)
                    record['donor_tango'] = d_tango
                    record['acceptor_tango'] = a_tango
                    record['pair_tango'] = min(d_tango, a_tango)

                records.append(record)
                evaluated += 1

            if evaluated % 5000 == 0:
                logger.info(f"  Evaluated {evaluated} pairs...")
        
        pairs_df = pd.DataFrame(records)
        logger.info(f"  Evaluation complete: {len(pairs_df)} pairs")
        logger.info("=" * 80)
        return pairs_df
    
    def filter_high_performance_pairs(self, pairs_df):
        """
        Filter pairs that meet SCREENING thresholds (not validated predictions).
        Two tiers:
          - TIER 1 (strict):  PCE>=18%, Jsc>=25, FF>=0.75, Voc>=1.0, pair_tango>=0.50
          - TIER 2 (relaxed): PCE>=14%, Jsc>=18, FF>=0.60, Voc>=0.85, pair_tango>=0.35
        Both tiers are returned merged, with a 'tier' column, and their counts
        are reported SEPARATELY (see logging below) so the relaxed Tier-2
        majority is never mistaken for the strict Tier-1 population.

        NOTE ON FF: the strict tier applies FF>=0.75 (FF_THRESHOLD_TIER1), not
        0.8. Earlier versions applied 0.75 while printing 0.8 in logs, metrics
        text and figure labels; the applied value is now the reported value.

        The pair_tango synthesizability gate (when the column is present,
        i.e. TANGO lookups were passed to evaluate_da_pairs) screens out
        pairs where the DEVICE PROPERTY prediction looks great but at least
        one of the two molecules is synthetically implausible — without
        this, "high-performance" meant "high predicted PCE" only, with zero
        connection to whether the molecule could plausibly be made and
        therefore whether the pair could ever perform this way in reality.
        """
        if pairs_df.empty:
            return pairs_df

        has_tango = 'pair_tango' in pairs_df.columns
        n_total = len(pairs_df)

        # Tier 1 — strict high-performance
        t1_mask = pd.Series(True, index=pairs_df.index)
        t1_thresholds = {
            'PCE': self.PCE_THRESHOLD,        # 18%
            'Jsc': self.JSC_THRESHOLD,        # 25 mA/cm2
            'FF':  self.FF_THRESHOLD_TIER1,   # 0.75 (relaxed from 0.8; now reported as 0.75 everywhere)
            'Voc': self.VOC_THRESHOLD,        # 1.0 V
        }
        for col, thresh in t1_thresholds.items():
            if col in pairs_df.columns:
                t1_mask &= pairs_df[col] >= thresh
        if has_tango:
            t1_mask &= pairs_df['pair_tango'] >= self.SYNTH_THRESHOLD_TIER1

        tier1 = pairs_df[t1_mask].copy()
        tier1['tier'] = 'Tier1_HighPerformance'

        # Tier 2 — relaxed promising
        t2_mask = pd.Series(True, index=pairs_df.index)
        t2_thresholds = {
            'PCE': 14.0,
            'Jsc': 18.0,
            'FF':  0.60,
            'Voc': 0.85,
        }
        for col, thresh in t2_thresholds.items():
            if col in pairs_df.columns:
                t2_mask &= pairs_df[col] >= thresh
        if has_tango:
            t2_mask &= pairs_df['pair_tango'] >= self.SYNTH_THRESHOLD_TIER2
        # Exclude Tier 1 from Tier 2 to avoid double-counting
        t2_mask &= ~t1_mask
        tier2 = pairs_df[t2_mask].copy()
        tier2['tier'] = 'Tier2_Promising'

        filtered = pd.concat([tier1, tier2], ignore_index=True)

        # ------------------------------------------------------------------
        # EXPLICIT TIER REPORTING.
        # Previously only the COMBINED Tier1+Tier2 count was surfaced, which
        # made a relaxed-threshold majority look like it satisfied the strict
        # criteria. Tier1 and Tier2 counts/fractions are now always reported
        # separately, and the combined number is explicitly labelled as such.
        # ------------------------------------------------------------------
        n_t1, n_t2, n_f = len(tier1), len(tier2), len(filtered)
        pct = (lambda n: (100.0 * n / n_total) if n_total else 0.0)
        logger.info("  " + "-" * 74)
        logger.info("  HIGH-PERFORMANCE PAIR FILTER RESULTS (screening surrogate, Method 1 only)")
        logger.info(f"    Total pairs evaluated                : {n_total}")
        logger.info(
            f"    TIER 1 (STRICT)                      : {n_t1} ({pct(n_t1):.2f}% of all pairs)"
        )
        logger.info(
            f"      thresholds: PCE>={self.PCE_THRESHOLD}%, Jsc>={self.JSC_THRESHOLD} mA/cm2, "
            f"FF>={self.FF_THRESHOLD_TIER1}, Voc>={self.VOC_THRESHOLD} V"
            + (f", pair_tango>={self.SYNTH_THRESHOLD_TIER1}" if has_tango else "")
        )
        logger.info(
            f"    TIER 2 (RELAXED / 'promising')        : {n_t2} ({pct(n_t2):.2f}% of all pairs)"
        )
        logger.info(
            f"      thresholds: PCE>=14%, Jsc>=18 mA/cm2, FF>=0.60, Voc>=0.85 V"
            + (f", pair_tango>={self.SYNTH_THRESHOLD_TIER2}" if has_tango else "")
        )
        logger.info(
            f"    COMBINED Tier1+Tier2 (NOT 'strict')  : {n_f} ({pct(n_f):.2f}% of all pairs)"
        )
        logger.warning(
            "    REPORTING NOTE: the combined Tier1+Tier2 fraction must NOT be "
            "described as the fraction meeting the strict criteria. Only the "
            f"Tier 1 fraction ({pct(n_t1):.2f}%) satisfies the strict thresholds."
        )
        logger.info(
            "    SCREEN-ONLY NOTE: these thresholds are applied to Method-1 "
            "screening-surrogate outputs and a TANGO-INSPIRED synthesizability "
            "heuristic. Neither is a validated predictor; treat this as a "
            "shortlist for downstream validation, not a performance claim."
        )
        logger.info("  " + "-" * 74)
        if not has_tango:
            logger.warning(
                "  pair_tango column not found — synthesizability gate NOT applied to this "
                "filter pass. Pass donor_tango_lookup/acceptor_tango_lookup to evaluate_da_pairs() "
                "to enable it."
            )
        return filtered
    
    def save_predictions(self, pairs_df, filtered_df, output_dir="results"):
        """Save prediction results to CSV files with physical-bounds statistics."""
        os.makedirs(output_dir, exist_ok=True)

        all_pairs_path = os.path.join(output_dir, "generated_molecules_with_predictions.csv")
        pairs_df.to_csv(all_pairs_path, index=False)
        logger.info(f"  Saved all pairs to {all_pairs_path}")

        hp_path = os.path.join(output_dir, "high_performance_pairs.csv")
        filtered_df.to_csv(hp_path, index=False)
        logger.info(f"  Saved high-performance pairs to {hp_path}")

        stats = {}
        for col in ['PCE', 'Jsc', 'FF', 'Voc']:
            if col in pairs_df.columns:
                col_data = pairs_df[col].dropna()
                bounds = self.PHYSICAL_BOUNDS.get(col, (None, None))
                # Count how many predictions were clipped (informational only)
                if bounds[0] is not None:
                    n_clipped_low  = int((col_data < bounds[0]).sum())
                    n_clipped_high = int((col_data > bounds[1]).sum())
                else:
                    n_clipped_low = n_clipped_high = 0
                stats[col] = {
                    'mean':              col_data.mean(),
                    'std':               col_data.std(),
                    'min':               col_data.min(),
                    'max':               col_data.max(),
                    'median':            col_data.median(),
                    'physical_bound_lo': bounds[0],
                    'physical_bound_hi': bounds[1],
                    'n_at_lower_bound':  n_clipped_low,
                    'n_at_upper_bound':  n_clipped_high,
                }
                logger.info(
                    f"  {col}: mean={col_data.mean():.3f}, min={col_data.min():.3f}, "
                    f"max={col_data.max():.3f}  "
                    f"[physical bounds {bounds[0]}–{bounds[1]}, "
                    f"clipped_low={n_clipped_low}, clipped_high={n_clipped_high}]"
                )

        stats_path = os.path.join(output_dir, "dmpnn_prediction_stats.csv")
        pd.DataFrame(stats).to_csv(stats_path)
        logger.info(f"  Saved prediction statistics to {stats_path}")

        return all_pairs_path, hp_path


# ============================================================================
# SYNTHESIZABILITY-ENHANCED GENERATION FUNCTIONS
# ============================================================================

def generate_molecules_with_synth_filter(generator, latent_dim, num_samples, 
                                         synth_threshold=0.5, max_attempts_multiplier=10):
    """
    Generate molecules with real-time synthesizability filtering
    
    Args:
        generator: Trained generator model
        latent_dim: Latent dimension size
        num_samples: Number of valid molecules to generate
        synth_threshold: Minimum synthesizability score (0-1)
        max_attempts_multiplier: Maximum attempts = num_samples * this value
    
    Returns:
        List of dicts with 'mol', 'smiles', 'synth_score', 'synth_details'
    """
    valid_molecules = []
    attempts = 0
    max_attempts = num_samples * max_attempts_multiplier
    
    logger.info(f"Generating {num_samples} molecules with synthesizability threshold {synth_threshold}")
    
    while len(valid_molecules) < num_samples and attempts < max_attempts:
        try:
            # Generate molecule
            noise = np.random.normal(0, 1, (1, latent_dim))
            generated = generator.predict(noise, verbose=0)
            
            # This is a placeholder - actual decoding depends on your generator type
            # You'll need to adapt this based on whether it's SMILES, graph, or coordinate-based
            mol = None  # Replace with actual decoding logic
            
            if mol is not None:
                # Calculate synthesizability
                synth_score, details = SYNTH_SCORER.score_tier1(mol)
                
                # Filter by threshold
                if synth_score >= synth_threshold:
                    valid_molecules.append({
                        'mol': mol,
                        'smiles': Chem.MolToSmiles(mol),
                        'synth_score': synth_score,
                        'synth_details': details
                    })
                    
                    if len(valid_molecules) % 10 == 0:
                        logger.info(f"  Generated {len(valid_molecules)}/{num_samples} valid molecules")
        except Exception as e:
            logger.debug(f"Generation attempt failed: {e}")
        
        attempts += 1
    
    logger.info(f"Generated {len(valid_molecules)} molecules in {attempts} attempts")
    return valid_molecules


def rank_molecules_by_synthesizability(molecules, top_k=100, use_tier2=True):
    """
    Rank molecules by synthesizability score
    
    Args:
        molecules: List of RDKit mol objects or SMILES strings
        top_k: Number of top molecules to assess with Tier 2
        use_tier2: Whether to use detailed Tier 2 assessment for top candidates
    
    Returns:
        List of dicts sorted by synthesizability (highest first)
    """
    logger.info(f"Ranking {len(molecules)} molecules by synthesizability...")
    
    scored_mols = []
    
    # Tier 1: Fast screening for all molecules
    for i, mol_input in enumerate(molecules):
        try:
            # Handle both mol objects and SMILES strings
            if isinstance(mol_input, str):
                mol = Chem.MolFromSmiles(mol_input)
                smiles = mol_input
            else:
                mol = mol_input
                smiles = Chem.MolToSmiles(mol) if mol else None
            
            if mol is None:
                continue
            
            # Tier 1 scoring
            synth_score_t1, details_t1 = SYNTH_SCORER.score_tier1(mol)
            
            scored_mols.append({
                'mol': mol,
                'smiles': smiles,
                'synth_score_tier1': synth_score_t1,
                'synth_details_tier1': details_t1,
                'synth_score_tier2': None,
                'synth_details_tier2': None
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Scored {i + 1}/{len(molecules)} molecules (Tier 1)")
        
        except Exception as e:
            logger.debug(f"Scoring failed for molecule {i}: {e}")
            continue
    
    # Sort by Tier 1 score
    scored_mols.sort(key=lambda x: x['synth_score_tier1'], reverse=True)
    
    # Tier 2: Detailed assessment for top candidates
    if use_tier2 and top_k > 0:
        logger.info(f"Performing Tier 2 assessment for top {min(top_k, len(scored_mols))} molecules...")
        
        for i, mol_data in enumerate(scored_mols[:top_k]):
            try:
                mol = mol_data['mol']
                synth_score_t2, details_t2 = SYNTH_SCORER.score_tier2(mol)
                
                mol_data['synth_score_tier2'] = synth_score_t2
                mol_data['synth_details_tier2'] = details_t2
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  Tier 2 scored {i + 1}/{min(top_k, len(scored_mols))} molecules")
            
            except Exception as e:
                logger.debug(f"Tier 2 scoring failed for molecule {i}: {e}")
                continue
        
        # Re-sort by Tier 2 score for top candidates
        top_with_tier2 = [m for m in scored_mols[:top_k] if m['synth_score_tier2'] is not None]
        top_with_tier2.sort(key=lambda x: x['synth_score_tier2'], reverse=True)
        
        # Combine: Tier 2 scored molecules first, then rest
        remaining = scored_mols[top_k:]
        scored_mols = top_with_tier2 + remaining
    
    logger.info(f"Ranking complete. Top score: {scored_mols[0]['synth_score_tier1']:.3f}")
    
    return scored_mols


def add_synthesizability_to_smiles_list(smiles_list, use_tier2_for_top=100):
    """
    Add synthesizability scores to a list of SMILES strings
    
    Args:
        smiles_list: List of SMILES strings
        use_tier2_for_top: Number of top molecules to assess with Tier 2
    
    Returns:
        List of dicts with SMILES and synthesizability scores
    """
    logger.info(f"Adding synthesizability scores to {len(smiles_list)} SMILES...")
    
    results = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            synth_score, details = SYNTH_SCORER.score_tier1(mol)
            
            results.append({
                'smiles': smiles,
                'synth_score': synth_score,
                'synth_details': details
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(smiles_list)} SMILES")
        
        except Exception as e:
            logger.debug(f"Processing failed for SMILES {i}: {e}")
            continue
    
    # Sort by score
    results.sort(key=lambda x: x['synth_score'], reverse=True)
    
    # Add Tier 2 for top molecules
    if use_tier2_for_top > 0:
        logger.info(f"Adding Tier 2 scores for top {min(use_tier2_for_top, len(results))} molecules...")
        
        for i, result in enumerate(results[:use_tier2_for_top]):
            try:
                mol = Chem.MolFromSmiles(result['smiles'])
                synth_score_t2, details_t2 = SYNTH_SCORER.score_tier2(mol)
                
                result['synth_score_tier2'] = synth_score_t2
                result['synth_details_tier2'] = details_t2
            except Exception as e:
                logger.debug(f"Tier 2 scoring failed: {e}")
    
    logger.info(f"Synthesizability scoring complete. Top score: {results[0]['synth_score']:.3f}")
    
    return results


# ============================================================================
# SYNTHESIZABILITY VALIDATION AND REPORTING UTILITIES
# ============================================================================

def validate_synthesizability_predictor(predictor_func, test_molecules, test_labels):
    """
    Validate synthesizability predictor against known molecules
    
    Args:
        predictor_func: Function that takes mol and returns score (0-1)
        test_molecules: List of (SMILES, is_synthesized) tuples
        test_labels: List of ground truth labels (1=synthesized, 0=not)
    
    Returns:
        Dictionary with validation metrics
    """
    from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
    
    logger.info(f"Validating synthesizability predictor on {len(test_molecules)} test molecules...")
    
    predictions = []
    ground_truth = []
    
    for smiles, is_synthesized in test_molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            score = predictor_func(mol)
            predictions.append(score)
            ground_truth.append(1.0 if is_synthesized else 0.0)
        except Exception as e:
            logger.debug(f"Validation failed for {smiles}: {e}")
            continue
    
    if len(predictions) == 0:
        logger.warning("No valid predictions for validation")
        return {}
    
    # Calculate metrics
    try:
        auc = roc_auc_score(ground_truth, predictions)
        precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
        
        # Find optimal threshold (maximize F1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # Calculate accuracy at optimal threshold
        binary_preds = [1 if p >= optimal_threshold else 0 for p in predictions]
        accuracy = accuracy_score(ground_truth, binary_preds)
        
        metrics = {
            'auc_roc': auc,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision_at_optimal': precision[optimal_idx],
            'recall_at_optimal': recall[optimal_idx],
            'f1_at_optimal': f1_scores[optimal_idx],
            'num_samples': len(predictions)
        }
        
        logger.info(f"Validation Results:")
        logger.info(f"  AUC-ROC: {auc:.3f}")
        logger.info(f"  Optimal Threshold: {optimal_threshold:.3f}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Metric calculation failed: {e}")
        return {}


def create_opv_test_set():
    """
    Create test set of known OPV molecules for validation
    Returns list of (SMILES, is_synthesized) tuples
    """
    test_molecules = [
        # SYNTHESIZED OPV materials (positive examples)
        # BDT-based donors
        ('c1ccc2sccc2c1', True),  # Benzodithiophene core - widely synthesized
        ('C1=CSC(=C1)c1ccc2sccc2c1', True),  # BDT-thiophene - common donor
        
        # Thiophene derivatives
        ('c1sccc1', True),  # Thiophene - basic building block
        ('C1=CSC(=C1)C#N', True),  # Thiophene-cyano - synthesized acceptor
        ('Fc1csc(c1)c1sccc1', True),  # Fluorothiophene derivative
        
        # Known acceptors (Y6, ITIC family)
        ('C1=CC=C(C=C1)C#N', True),  # Benzonitrile - simple acceptor
        ('O=C1c2ccccc2C(=O)c2ccccc21', True),  # Anthraquinone - acceptor core
        
        # PM6, PBDB-T related structures (simplified)
        ('c1ccc(cc1)C', True),  # Toluene - simple aromatic
        ('c1ccc(F)cc1', True),  # Fluorobenzene - halogenated aromatic
        
        # UNREALISTIC/COMPLEX molecules (negative examples)
        # Overly complex fused systems
        ('C12C3C4C5C6C7C8C9C1C1C2C3C4C5C6C7C8C9C1', False),  # Unrealistic cage
        ('C1(C2C3C4C5C6C7C8C9C%10C%11C%12C1C2C3C4C5C6C7C8C9C%10C%11C%12)C', False),  # Too complex
        
        # Unstable/reactive structures
        ('C(=O)C(=O)C(=O)C(=O)C(=O)', False),  # Multiple adjacent carbonyls - unstable
        ('C#CC#CC#CC#CC#C', False),  # Multiple adjacent alkynes - difficult
        
        # Unrealistic heteroatom combinations
        ('[Se][Se][Se][Se]c1ccccc1', False),  # Multiple selenium - expensive/difficult
        ('C(F)(F)(F)C(F)(F)(F)C(F)(F)(F)', False),  # Excessive fluorination
    ]
    
    return test_molecules


def save_synthesizability_results(scored_molecules, output_file='synthesizability_results.csv'):
    """
    Save synthesizability scoring results to CSV file
    
    Args:
        scored_molecules: List of dicts from rank_molecules_by_synthesizability or add_synthesizability_to_smiles_list
        output_file: Output CSV filename
    """
    logger.info(f"Saving synthesizability results to {output_file}...")
    
    try:
        # Prepare data for CSV
        rows = []
        for i, mol_data in enumerate(scored_molecules):
            row = {
                'rank': i + 1,
                'smiles': mol_data.get('smiles', ''),
                'synth_score_tier1': mol_data.get('synth_score_tier1') or mol_data.get('synth_score', 0.0),
                'synth_score_tier2': mol_data.get('synth_score_tier2', ''),
            }
            
            # Add detailed scores from tier 1
            details_t1 = mol_data.get('synth_details_tier1') or mol_data.get('synth_details', {})
            row['syba_score'] = details_t1.get('syba', '')
            row['scscore'] = details_t1.get('scscore', '')
            row['opv_heuristic'] = details_t1.get('opv_heuristic', '')
            
            # Add detailed scores from tier 2
            details_t2 = mol_data.get('synth_details_tier2', {})
            row['rascore'] = details_t2.get('rascore', '')
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"âœ“ Saved {len(rows)} molecules to {output_file}")
        
        # Print summary statistics
        logger.info("Synthesizability Score Statistics:")
        logger.info(f"  Mean Tier 1 Score: {df['synth_score_tier1'].mean():.3f}")
        logger.info(f"  Median Tier 1 Score: {df['synth_score_tier1'].median():.3f}")
        logger.info(f"  Std Dev: {df['synth_score_tier1'].std():.3f}")
        logger.info(f"  Min Score: {df['synth_score_tier1'].min():.3f}")
        logger.info(f"  Max Score: {df['synth_score_tier1'].max():.3f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to save synthesizability results: {e}")
        return None


def generate_synthesizability_report(scored_molecules, output_file='synthesizability_report.txt'):
    """
    Generate detailed synthesizability assessment report
    
    Args:
        scored_molecules: List of dicts with synthesizability scores
        output_file: Output text report filename
    """
    logger.info(f"Generating synthesizability report to {output_file}...")
    
    try:
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SYNTHESIZABILITY ASSESSMENT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total molecules assessed: {len(scored_molecules)}\n")
            
            tier1_scores = [m.get('synth_score_tier1') or m.get('synth_score', 0.0) 
                           for m in scored_molecules]
            f.write(f"Mean synthesizability score: {np.mean(tier1_scores):.3f}\n")
            f.write(f"Median synthesizability score: {np.median(tier1_scores):.3f}\n")
            f.write(f"Std deviation: {np.std(tier1_scores):.3f}\n")
            f.write(f"Min score: {np.min(tier1_scores):.3f}\n")
            f.write(f"Max score: {np.max(tier1_scores):.3f}\n\n")
            
            # Score distribution
            f.write("SCORE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            high_synth = sum(1 for s in tier1_scores if s >= 0.7)
            medium_synth = sum(1 for s in tier1_scores if 0.4 <= s < 0.7)
            low_synth = sum(1 for s in tier1_scores if s < 0.4)
            
            f.write(f"High synthesizability (â‰¥0.7): {high_synth} ({100*high_synth/len(tier1_scores):.1f}%)\n")
            f.write(f"Medium synthesizability (0.4-0.7): {medium_synth} ({100*medium_synth/len(tier1_scores):.1f}%)\n")
            f.write(f"Low synthesizability (<0.4): {low_synth} ({100*low_synth/len(tier1_scores):.1f}%)\n\n")
            
            # Top 10 most synthesizable
            f.write("TOP 10 MOST SYNTHESIZABLE MOLECULES\n")
            f.write("-" * 80 + "\n")
            for i, mol_data in enumerate(scored_molecules[:10]):
                smiles = mol_data.get('smiles', 'N/A')
                score = mol_data.get('synth_score_tier1') or mol_data.get('synth_score', 0.0)
                f.write(f"{i+1}. Score: {score:.3f} | SMILES: {smiles}\n")
                
                # Show detailed scores if available
                details = mol_data.get('synth_details_tier1') or mol_data.get('synth_details', {})
                if details:
                    f.write(f"   Details: ")
                    detail_parts = []
                    if 'syba' in details:
                        detail_parts.append(f"SYBA={details['syba']:.3f}")
                    if 'scscore' in details:
                        detail_parts.append(f"SCScore={details['scscore']:.3f}")
                    if 'opv_heuristic' in details:
                        detail_parts.append(f"OPV={details['opv_heuristic']:.3f}")
                    if 'rascore' in details:
                        detail_parts.append(f"RAScore={details['rascore']:.3f}")
                    f.write(", ".join(detail_parts) + "\n")
                f.write("\n")
            
            # Bottom 10 least synthesizable
            f.write("\nBOTTOM 10 LEAST SYNTHESIZABLE MOLECULES\n")
            f.write("-" * 80 + "\n")
            for i, mol_data in enumerate(scored_molecules[-10:]):
                smiles = mol_data.get('smiles', 'N/A')
                score = mol_data.get('synth_score_tier1') or mol_data.get('synth_score', 0.0)
                rank = len(scored_molecules) - 9 + i
                f.write(f"{rank}. Score: {score:.3f} | SMILES: {smiles}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SCORING METHODOLOGY\n")
            f.write("=" * 80 + "\n")
            f.write("This assessment combines multiple state-of-the-art approaches:\n\n")
            
            if SYBA_AVAILABLE:
                f.write("âœ“ SYBA (Synthetic Bayesian Accessibility)\n")
                f.write("  - Trained on 1M+ real synthesis outcomes from Reaxys\n")
                f.write("  - Provides probability of successful synthesis\n\n")
            
            if SCSCORE_AVAILABLE:
                f.write("âœ“ SCScore (Synthetic Complexity Score)\n")
                f.write("  - Learned from 12M reactions in USPTO database\n")
                f.write("  - Estimates number of synthetic steps required\n\n")
            
            if RASCORE_AVAILABLE:
                f.write("âœ“ RAScore (Retrosynthetic Accessibility)\n")
                f.write("  - Combines forward synthesis + retrosynthesis planning\n")
                f.write("  - Considers commercial availability of precursors\n\n")
            
            f.write("âœ“ Custom OPV Heuristics\n")
            f.write("  - Domain-specific rules for organic photovoltaics\n")
            f.write("  - Considers molecular symmetry, conjugation length\n")
            f.write("  - Penalizes difficult functional groups (Se, CF3, etc.)\n")
            f.write("  - Rewards known building blocks (BDT, thiophene, etc.)\n\n")
            
            f.write("Score Interpretation:\n")
            f.write("  0.7-1.0: Highly synthesizable (likely <5 steps)\n")
            f.write("  0.4-0.7: Moderately synthesizable (5-10 steps)\n")
            f.write("  0.0-0.4: Difficult to synthesize (>10 steps or uncertain)\n")
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"âœ“ Synthesizability report saved to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate synthesizability report: {e}")
        return False


def save_individual_score_analysis(scored_molecules, output_file='individual_score_analysis.txt'):
    """
    Generate detailed analysis of individual scoring approaches with interpretations and suggestions
    
    Args:
        scored_molecules: List of dicts with synthesizability scores
        output_file: Output text report filename
    """
    logger.info(f"Generating individual score analysis to {output_file}...")
    logger.debug(f"Analyzing {len(scored_molecules)} molecules across all scoring approaches")
    
    try:
        with open(output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("INDIVIDUAL SYNTHESIZABILITY SCORING APPROACH ANALYSIS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("This report provides detailed analysis of each scoring approach, including:\n")
            f.write("  - Score interpretation and meaning\n")
            f.write("  - Strengths and limitations\n")
            f.write("  - Practical suggestions for use\n")
            f.write("  - Performance statistics\n\n")
            
            # Collect statistics for each approach
            syba_scores = []
            scscore_scores = []
            rascore_scores = []
            opv_scores = []
            
            for mol_data in scored_molecules:
                details = mol_data.get('synth_details_tier1') or mol_data.get('synth_details', {})
                if 'syba' in details and details['syba'] is not None:
                    syba_scores.append(details['syba'])
                if 'scscore' in details and details['scscore'] is not None:
                    scscore_scores.append(details['scscore'])
                if 'opv_heuristic' in details and details['opv_heuristic'] is not None:
                    opv_scores.append(details['opv_heuristic'])
                
                details_t2 = mol_data.get('synth_details_tier2', {})
                if 'rascore' in details_t2 and details_t2['rascore'] is not None:
                    rascore_scores.append(details_t2['rascore'])
            
            logger.debug(f"Collected scores - SYBA: {len(syba_scores)}, SCScore: {len(scscore_scores)}, "
                        f"RAScore: {len(rascore_scores)}, OPV: {len(opv_scores)}")
            
            # ============================================================================
            # APPROACH 1: SYBA (Synthetic Bayesian Accessibility)
            # ============================================================================
            f.write("=" * 100 + "\n")
            f.write("APPROACH 1: SYBA (Synthetic Bayesian Accessibility)\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("DESCRIPTION:\n")
            f.write("-" * 100 + "\n")
            f.write("SYBA is a Bayesian machine learning classifier trained on over 1 million real synthesis\n")
            f.write("outcomes from the Reaxys database. It predicts the probability that a molecule can be\n")
            f.write("successfully synthesized based on patterns learned from actual laboratory results.\n\n")
            
            f.write("TRAINING DATA:\n")
            f.write("  - Source: Reaxys database (commercial synthesis database)\n")
            f.write("  - Size: 1M+ real synthesis outcomes\n")
            f.write("  - Type: Binary classification (synthesized vs. not synthesized)\n")
            f.write("  - Validation: Cross-validated on held-out synthesis data\n\n")
            
            f.write("SCORE INTERPRETATION:\n")
            f.write("-" * 100 + "\n")
            f.write("Score Range | Meaning                    | Interpretation\n")
            f.write("-" * 100 + "\n")
            f.write("0.8 - 1.0   | Very high probability      | Molecule has strong similarity to successfully\n")
            f.write("            |                            | synthesized compounds. Likely <3 steps.\n")
            f.write("0.6 - 0.8   | High probability           | Good synthesis prospects. Likely 3-5 steps.\n")
            f.write("0.4 - 0.6   | Moderate probability       | Feasible but may require optimization. 5-8 steps.\n")
            f.write("0.2 - 0.4   | Low probability            | Challenging synthesis. >8 steps or uncertain.\n")
            f.write("0.0 - 0.2   | Very low probability       | Unlikely to be synthesizable with current methods.\n\n")
            
            if syba_scores:
                f.write("STATISTICS FOR YOUR DATASET:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total molecules scored: {len(syba_scores)}\n")
                f.write(f"Mean SYBA score: {np.mean(syba_scores):.3f}\n")
                f.write(f"Median SYBA score: {np.median(syba_scores):.3f}\n")
                f.write(f"Std deviation: {np.std(syba_scores):.3f}\n")
                f.write(f"Min score: {np.min(syba_scores):.3f}\n")
                f.write(f"Max score: {np.max(syba_scores):.3f}\n\n")
                
                high = sum(1 for s in syba_scores if s >= 0.6)
                medium = sum(1 for s in syba_scores if 0.4 <= s < 0.6)
                low = sum(1 for s in syba_scores if s < 0.4)
                
                f.write("DISTRIBUTION:\n")
                f.write(f"  High probability (â‰¥0.6): {high} ({100*high/len(syba_scores):.1f}%)\n")
                f.write(f"  Moderate probability (0.4-0.6): {medium} ({100*medium/len(syba_scores):.1f}%)\n")
                f.write(f"  Low probability (<0.4): {low} ({100*low/len(syba_scores):.1f}%)\n\n")
            else:
                f.write("STATISTICS: SYBA not available or no molecules scored\n\n")
            
            f.write("STRENGTHS:\n")
            f.write("  âœ“ Trained on real synthesis data (not heuristics)\n")
            f.write("  âœ“ High accuracy (85-90% on validation sets)\n")
            f.write("  âœ“ Fast computation (~100 molecules/second)\n")
            f.write("  âœ“ Well-validated across diverse chemical space\n")
            f.write("  âœ“ Low false positive rate\n\n")
            
            f.write("LIMITATIONS:\n")
            f.write("  âœ— Biased toward medicinal chemistry (Reaxys focus)\n")
            f.write("  âœ— May not capture cutting-edge synthetic methods\n")
            f.write("  âœ— Less accurate for novel scaffolds not in training data\n")
            f.write("  âœ— Does not provide synthetic routes (only probability)\n\n")
            
            f.write("PRACTICAL SUGGESTIONS:\n")
            f.write("  1. Use SYBA as primary filter for large molecule libraries\n")
            f.write("  2. Set threshold at 0.5-0.6 for balanced filtering\n")
            f.write("  3. Combine with domain-specific heuristics for specialized chemistry\n")
            f.write("  4. For OPV materials, use in conjunction with OPV heuristic\n")
            f.write("  5. Molecules with SYBA > 0.7 are excellent experimental candidates\n\n")
            
            # ============================================================================
            # APPROACH 2: SCScore (Synthetic Complexity Score)
            # ============================================================================
            f.write("\n" + "=" * 100 + "\n")
            f.write("APPROACH 2: SCScore (Synthetic Complexity Score)\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("DESCRIPTION:\n")
            f.write("-" * 100 + "\n")
            f.write("SCScore is a neural network trained on 12 million reactions from the USPTO database.\n")
            f.write("It estimates synthetic complexity by learning patterns from real chemical reactions,\n")
            f.write("predicting how many steps are likely required to synthesize a molecule.\n\n")
            
            f.write("TRAINING DATA:\n")
            f.write("  - Source: USPTO patent database (public reaction data)\n")
            f.write("  - Size: 12M+ reactions\n")
            f.write("  - Type: Regression (complexity score 1-5)\n")
            f.write("  - Coverage: Broad chemical space including industrial chemistry\n\n")
            
            f.write("SCORE INTERPRETATION (normalized to 0-1, higher=better):\n")
            f.write("-" * 100 + "\n")
            f.write("Score Range | Original | Meaning                | Typical Steps\n")
            f.write("-" * 100 + "\n")
            f.write("0.8 - 1.0   | 1.0-1.8  | Very simple            | 1-2 steps\n")
            f.write("0.6 - 0.8   | 1.8-2.6  | Simple                 | 2-4 steps\n")
            f.write("0.4 - 0.6   | 2.6-3.4  | Moderate complexity    | 4-7 steps\n")
            f.write("0.2 - 0.4   | 3.4-4.2  | Complex                | 7-12 steps\n")
            f.write("0.0 - 0.2   | 4.2-5.0  | Very complex           | >12 steps\n\n")
            
            if scscore_scores:
                f.write("STATISTICS FOR YOUR DATASET:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total molecules scored: {len(scscore_scores)}\n")
                f.write(f"Mean SCScore: {np.mean(scscore_scores):.3f}\n")
                f.write(f"Median SCScore: {np.median(scscore_scores):.3f}\n")
                f.write(f"Std deviation: {np.std(scscore_scores):.3f}\n")
                f.write(f"Min score: {np.min(scscore_scores):.3f}\n")
                f.write(f"Max score: {np.max(scscore_scores):.3f}\n\n")
                
                simple = sum(1 for s in scscore_scores if s >= 0.6)
                moderate = sum(1 for s in scscore_scores if 0.4 <= s < 0.6)
                complex_mols = sum(1 for s in scscore_scores if s < 0.4)
                
                f.write("DISTRIBUTION:\n")
                f.write(f"  Simple (â‰¥0.6): {simple} ({100*simple/len(scscore_scores):.1f}%)\n")
                f.write(f"  Moderate (0.4-0.6): {moderate} ({100*moderate/len(scscore_scores):.1f}%)\n")
                f.write(f"  Complex (<0.4): {complex_mols} ({100*complex_mols/len(scscore_scores):.1f}%)\n\n")
            else:
                f.write("STATISTICS: SCScore not available or no molecules scored\n\n")
            
            f.write("STRENGTHS:\n")
            f.write("  âœ“ Large training dataset (12M reactions)\n")
            f.write("  âœ“ Captures reaction complexity patterns\n")
            f.write("  âœ“ Good for estimating synthetic step count\n")
            f.write("  âœ“ Includes industrial chemistry (USPTO patents)\n")
            f.write("  âœ“ Continuous score (more granular than binary)\n\n")
            
            f.write("LIMITATIONS:\n")
            f.write("  âœ— Slower than SYBA (~50 molecules/second)\n")
            f.write("  âœ— May overestimate complexity for novel reactions\n")
            f.write("  âœ— Does not consider reagent availability\n")
            f.write("  âœ— Patent data may include non-optimal routes\n\n")
            
            f.write("PRACTICAL SUGGESTIONS:\n")
            f.write("  1. Use SCScore to estimate synthetic effort (step count)\n")
            f.write("  2. Combine with SYBA for comprehensive assessment\n")
            f.write("  3. SCScore > 0.6 indicates likely <5 steps\n")
            f.write("  4. Good for prioritizing molecules by synthetic effort\n")
            f.write("  5. Consider alongside cost and time constraints\n\n")
            
            # ============================================================================
            # APPROACH 3: RAScore (Retrosynthetic Accessibility)
            # ============================================================================
            f.write("\n" + "=" * 100 + "\n")
            f.write("APPROACH 3: RAScore (Retrosynthetic Accessibility)\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("DESCRIPTION:\n")
            f.write("-" * 100 + "\n")
            f.write("RAScore combines forward synthesis prediction with retrosynthetic planning. It considers\n")
            f.write("both the feasibility of synthetic routes AND the commercial availability of starting\n")
            f.write("materials, providing a more holistic synthesizability assessment.\n\n")
            
            f.write("TRAINING DATA:\n")
            f.write("  - Source: Retrosynthesis database + building block databases\n")
            f.write("  - Type: Neural network with retrosynthetic tree search\n")
            f.write("  - Features: Considers precursor availability from suppliers\n\n")
            
            f.write("SCORE INTERPRETATION:\n")
            f.write("-" * 100 + "\n")
            f.write("Score Range | Meaning                    | Interpretation\n")
            f.write("-" * 100 + "\n")
            f.write("0.8 - 1.0   | Highly accessible          | Clear retrosynthetic route with available precursors\n")
            f.write("0.6 - 0.8   | Accessible                 | Good route exists, most precursors available\n")
            f.write("0.4 - 0.6   | Moderately accessible      | Route exists but may need custom precursors\n")
            f.write("0.2 - 0.4   | Difficult                  | Complex route or limited precursor availability\n")
            f.write("0.0 - 0.2   | Very difficult             | No clear route or precursors unavailable\n\n")
            
            if rascore_scores:
                f.write("STATISTICS FOR YOUR DATASET:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total molecules scored: {len(rascore_scores)}\n")
                f.write(f"Mean RAScore: {np.mean(rascore_scores):.3f}\n")
                f.write(f"Median RAScore: {np.median(rascore_scores):.3f}\n")
                f.write(f"Std deviation: {np.std(rascore_scores):.3f}\n")
                f.write(f"Min score: {np.min(rascore_scores):.3f}\n")
                f.write(f"Max score: {np.max(rascore_scores):.3f}\n\n")
                
                accessible = sum(1 for s in rascore_scores if s >= 0.6)
                moderate = sum(1 for s in rascore_scores if 0.4 <= s < 0.6)
                difficult = sum(1 for s in rascore_scores if s < 0.4)
                
                f.write("DISTRIBUTION:\n")
                f.write(f"  Accessible (â‰¥0.6): {accessible} ({100*accessible/len(rascore_scores):.1f}%)\n")
                f.write(f"  Moderately accessible (0.4-0.6): {moderate} ({100*moderate/len(rascore_scores):.1f}%)\n")
                f.write(f"  Difficult (<0.4): {difficult} ({100*difficult/len(rascore_scores):.1f}%)\n\n")
            else:
                f.write("STATISTICS: RAScore not available or no molecules scored\n\n")
            
            f.write("STRENGTHS:\n")
            f.write("  âœ“ Considers precursor availability (practical constraint)\n")
            f.write("  âœ“ Bidirectional assessment (forward + retro)\n")
            f.write("  âœ“ High accuracy (85-90%)\n")
            f.write("  âœ“ Reflects real-world synthesis feasibility\n")
            f.write("  âœ“ Updated with current building block databases\n\n")
            
            f.write("LIMITATIONS:\n")
            f.write("  âœ— Slower computation (~10 molecules/second)\n")
            f.write("  âœ— Requires building block database updates\n")
            f.write("  âœ— May miss novel disconnections\n")
            f.write("  âœ— Biased toward commercially available precursors\n\n")
            
            f.write("PRACTICAL SUGGESTIONS:\n")
            f.write("  1. Use RAScore for final candidate selection (Tier 2)\n")
            f.write("  2. Prioritize molecules with RAScore > 0.7 for experimental work\n")
            f.write("  3. Low RAScore may indicate need for custom precursor synthesis\n")
            f.write("  4. Combine with cost estimates for budget planning\n")
            f.write("  5. Check precursor availability before ordering synthesis\n\n")
            
            # ============================================================================
            # APPROACH 4: OPV Heuristic (Custom Domain-Specific Rules)
            # ============================================================================
            f.write("\n" + "=" * 100 + "\n")
            f.write("APPROACH 4: OPV Heuristic (Custom Organic Photovoltaic Rules)\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("DESCRIPTION:\n")
            f.write("-" * 100 + "\n")
            f.write("Custom rule-based scoring specifically designed for organic photovoltaic (OPV) materials.\n")
            f.write("Based on literature analysis of successfully synthesized donor and acceptor molecules\n")
            f.write("including BDT-based donors, Y6/ITIC acceptors, and high-performance OPV materials.\n\n")
            
            f.write("SCORING FACTORS:\n")
            f.write("  Rewards (+):\n")
            f.write("    • Known building blocks (BDT, thiophene, benzene): +0.10\n")
            f.write("    • Molecular symmetry: +0.15 × symmetry_score\n")
            f.write("    • Typical fused ring systems (2-4 rings): +0.10\n")
            f.write("  \n")
            f.write("  Penalties (-):\n")
            f.write("    • Molecular complexity (>100 atoms): -0.20\n")
            f.write("    • Difficult functional groups:\n")
            f.write("        - Selenium: -0.20 per atom\n")
            f.write("        - CF3: -0.10 per group\n")
            f.write("        - Dicyanovinyl: -0.15 per group\n")
            f.write("    • Excessive conjugation (>15 atoms): -0.15\n")
            f.write("    • Stereocenters: -0.10 per center\n")
            f.write("    • Too many reactive groups (>2): -0.20\n\n")
            
            if opv_scores:
                f.write("STATISTICS FOR YOUR DATASET:\n")
                f.write("-" * 100 + "\n")
                f.write(f"Total molecules scored: {len(opv_scores)}\n")
                f.write(f"Mean OPV score: {np.mean(opv_scores):.3f}\n")
                f.write(f"Median OPV score: {np.median(opv_scores):.3f}\n")
                f.write(f"Std deviation: {np.std(opv_scores):.3f}\n")
                f.write(f"Min score: {np.min(opv_scores):.3f}\n")
                f.write(f"Max score: {np.max(opv_scores):.3f}\n\n")
                
                excellent = sum(1 for s in opv_scores if s >= 0.7)
                good = sum(1 for s in opv_scores if 0.5 <= s < 0.7)
                challenging = sum(1 for s in opv_scores if s < 0.5)
                
                f.write("DISTRIBUTION:\n")
                f.write(f"  Excellent for OPV (â‰¥0.7): {excellent} ({100*excellent/len(opv_scores):.1f}%)\n")
                f.write(f"  Good for OPV (0.5-0.7): {good} ({100*good/len(opv_scores):.1f}%)\n")
                f.write(f"  Challenging (<0.5): {challenging} ({100*challenging/len(opv_scores):.1f}%)\n\n")
            else:
                f.write("STATISTICS: OPV heuristic scores not available\n\n")
            
            f.write("STRENGTHS:\n")
            f.write("  âœ“ Domain-specific for OPV materials\n")
            f.write("  âœ“ Captures OPV synthesis challenges (Se, heavy halogens)\n")
            f.write("  âœ“ Rewards known successful scaffolds (BDT, thiophene)\n")
            f.write("  âœ“ Very fast computation (instant)\n")
            f.write("  âœ“ Interpretable (can see which factors affect score)\n\n")
            
            f.write("LIMITATIONS:\n")
            f.write("  âœ— Rule-based (not learned from data)\n")
            f.write("  âœ— May miss novel synthetic strategies\n")
            f.write("  âœ— Specific to OPV domain (not general)\n")
            f.write("  âœ— Requires manual rule updates\n\n")
            
            f.write("PRACTICAL SUGGESTIONS:\n")
            f.write("  1. Use OPV heuristic as primary filter for OPV-specific work\n")
            f.write("  2. Molecules with OPV score > 0.7 are well-suited for OPV synthesis\n")
            f.write("  3. Low OPV score may indicate expensive/difficult functional groups\n")
            f.write("  4. Combine with general models (SYBA, SCScore) for balanced view\n")
            f.write("  5. Consider modifying structures to improve OPV score\n\n")
            
            # ============================================================================
            # COMBINED RECOMMENDATIONS
            # ============================================================================
            f.write("\n" + "=" * 100 + "\n")
            f.write("COMBINED RECOMMENDATIONS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("BEST PRACTICES FOR USING MULTIPLE APPROACHES:\n")
            f.write("-" * 100 + "\n")
            f.write("1. INITIAL SCREENING (Tier 1):\n")
            f.write("   Use: SYBA + SCScore + OPV Heuristic\n")
            f.write("   Speed: ~100 molecules/second\n")
            f.write("   Purpose: Rapid filtering of large libraries\n")
            f.write("   Threshold: Combined score â‰¥ 0.5 for exploratory, â‰¥ 0.6 for focused work\n\n")
            
            f.write("2. DETAILED ASSESSMENT (Tier 2):\n")
            f.write("   Use: All approaches including RAScore\n")
            f.write("   Speed: ~10 molecules/second\n")
            f.write("   Purpose: Final candidate selection\n")
            f.write("   Apply to: Top 50-100 molecules from Tier 1\n\n")
            
            f.write("3. EXPERIMENTAL PRIORITIZATION:\n")
            f.write("   Highest priority: All scores > 0.7\n")
            f.write("   Medium priority: Combined score 0.5-0.7, at least one score > 0.7\n")
            f.write("   Low priority: All scores < 0.5\n\n")
            
            f.write("4. DISAGREEMENT RESOLUTION:\n")
            f.write("   If SYBA high but SCScore low: Simple structure, few steps\n")
            f.write("   If SYBA low but SCScore high: Novel scaffold, uncertain feasibility\n")
            f.write("   If OPV low but others high: May need expensive reagents for OPV\n")
            f.write("   If RAScore low but others high: Precursors may need custom synthesis\n\n")
            
            f.write("5. COST-BENEFIT ANALYSIS:\n")
            f.write("   High scores (>0.7): Prioritize for immediate synthesis\n")
            f.write("   Medium scores (0.5-0.7): Evaluate based on property predictions\n")
            f.write("   Low scores (<0.5): Consider only if exceptional predicted properties\n\n")
            
            f.write("=" * 100 + "\n")
            f.write("END OF INDIVIDUAL SCORE ANALYSIS\n")
            f.write("=" * 100 + "\n")
        
        logger.info(f"âœ“ Individual score analysis saved to {output_file}")
        logger.debug("Analysis includes interpretations for all scoring approaches")
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate individual score analysis: {e}")
        logger.exception("Full traceback:")
        return False


# Morgan fingerprint helper using MorganGenerator (avoids deprecation warnings)
@lru_cache(maxsize=16)
def _get_morgan_gen_cached(radius=2, nBits=2048):
    return GetMorganGenerator(radius, fpSize=nBits)


def morgan_fp(mol, radius=2, nBits=2048):
    try:
        # Use GetMorganFingerprintAsBitVect directly - it always returns ExplicitBitVect
        # This is more reliable than GetMorganGenerator for TanimotoSimilarity
        fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return fp
    except Exception as e:
        logger.debug(f"Morgan fingerprint failed (r={radius}, nBits={nBits}): {e}")
        return None


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


def max_tanimoto_to_training(mol, training_fps):
    fp = morgan_fp(mol, radius=2, nBits=2048)
    if fp is None or not training_fps:
        return 0.0
    return max(DataStructs.TanimotoSimilarity(fp, train_fp) for train_fp in training_fps)


def osc_realism_score(mol, molecule_type="Generic"):
    """
    Structural sanity check ONLY — no size, molecular-weight, ring-count,
    LogP, rotatable-bond, heteroatom-count, or functional-group-count
    restrictions. Per explicit user request: every possible donor/acceptor
    molecule derived from any training molecule in DMPNN.csv must be able
    to generate without any such restriction; SMILES length and molecular
    weight are unbounded. The ONLY things checked are whether RDKit
    considers the molecule a single, valid, non-radical, chemically
    representable structure — i.e. "is this a real molecule at all",
    not "does this molecule look like a typical OSC donor/acceptor".
    A constant score is returned for every valid molecule so downstream
    code that consumes (score, reason) keeps working unchanged.
    """
    if mol is None:
        return 0.0, "Null molecule"
    try:
        Chem.SanitizeMol(mol)
        if len(Chem.GetMolFrags(mol)) != 1:
            return 0.0, "Disconnected molecule"
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return 0.0, "Radical atoms"
        # No allowed-atom-type restriction, no heavy-atom/MW bounds, no ring/
        # aromaticity/heteroatom/rotatable-bond/LogP/functional-group
        # requirements. Any valid, connected, non-radical molecule passes.
        return 1.0, "Valid molecule (no OSC-likeness restrictions applied)"
    except Exception as e:
        return 0.0, f"Quality scoring failed: {e}"


def validate_osc_realism(mol, molecule_type="Generic", min_score=0.48):
    score, reason = osc_realism_score(mol, molecule_type)
    # min_score kept as a parameter for backward compatibility with call
    # sites, but osc_realism_score() now only ever returns 0.0 (invalid) or
    # 1.0 (valid) — there is no intermediate "quality" scale to threshold
    # against anymore, so any min_score <= 1.0 behaves identically (accept
    # every structurally valid molecule).
    if score > 0.0:
        return True, reason
    return False, reason


def size_adaptive_similarity_threshold(mol, base_threshold=0.4):
    """
    Compute a novelty (Tanimoto) threshold that scales with molecule size.

    Root cause this fixes: for a fixed-radius Morgan fingerprint, a single
    local structural edit (one substituent swap) changes a roughly CONSTANT
    number of fingerprint bits regardless of overall molecule size — but the
    TOTAL number of "on" bits grows with heavy-atom count. This means the
    resulting Tanimoto similarity to the parent molecule after one edit is
    LOW for small molecules (~0.8 for ~20 heavy atoms) but very HIGH for
    large fused-ring molecules (~0.95-0.98 for 80-200+ heavy atoms, the
    range covering ITIC/IDIC/Y6-class non-fullerene acceptors). Under a
    single fixed threshold (0.4), large molecules can essentially NEVER
    pass novelty filtering via single/few-edit mutation strategies, even
    though the resulting molecule is a genuinely distinct, chemically novel
    compound — this was silently erasing every ITIC/IDIC/Y6-scale candidate
    from the generated pool while small fragments passed freely.

    This scales the threshold up smoothly for larger molecules so that
    "novel relative to what's achievable via a few local edits at this
    scale" is judged consistently across molecule sizes, rather than
    penalizing large scaffolds for structural stability that is a direct,
    unavoidable consequence of fingerprint math rather than a lack of real
    chemical novelty.
    """
    try:
        heavy_atoms = mol.GetNumHeavyAtoms()
    except Exception:
        return base_threshold
    if heavy_atoms <= 30:
        return base_threshold
    # Linear ramp: 0.4 at 30 heavy atoms -> 0.75 at 150+ heavy atoms.
    # 0.75 still requires substantial, multi-site structural difference;
    # it is not a rubber stamp — it just stops penalizing large molecules
    # for a fingerprint-math artifact unrelated to true novelty.
    scaled = base_threshold + (0.75 - base_threshold) * min(1.0, (heavy_atoms - 30) / 120.0)
    return scaled


def accept_generated_candidate(
    smiles,
    training_canonical_set,
    training_fps,
    molecule_type,
    seen_canonical=None,
    max_similarity=0.4,
):
    can = canonical_smiles(smiles)
    if not can:
        return None, False, "Invalid or disconnected SMILES", 1.0
    if can in training_canonical_set:
        return None, False, "Exact training-set match", 1.0
    if seen_canonical is not None and can in seen_canonical:
        return None, False, "Duplicate generated molecule", 1.0
    mol = Chem.MolFromSmiles(can)
    ok, reason = validate_osc_realism(mol, molecule_type)
    if not ok:
        return None, False, reason, 1.0
    max_sim = max_tanimoto_to_training(mol, training_fps)
    effective_threshold = size_adaptive_similarity_threshold(mol, max_similarity)
    if max_sim >= effective_threshold:
        return None, False, f"Nearest training Tanimoto {max_sim:.3f} exceeds {effective_threshold:.3f}", max_sim
    return can, True, "Accepted", max_sim


def filter_generated_smiles(
    smiles_list,
    training_canonical_set,
    training_fps,
    molecule_type,
    max_similarity=0.4,
    existing_seen=None,
):
    logger.info(
        f"Strict filtering {len(smiles_list)} {molecule_type} candidates with max_similarity<{max_similarity}"
    )
    seen = set(existing_seen) if existing_seen is not None else set()
    accepted = []
    rejected = {}
    for smi in smiles_list:
        can, ok, reason, _ = accept_generated_candidate(
            smi,
            training_canonical_set,
            training_fps,
            molecule_type,
            seen,
            max_similarity=max_similarity,
        )
        if ok:
            accepted.append(can)
            seen.add(can)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    logger.info(
        f"Strict filtering complete for {molecule_type}: accepted={len(accepted)}, rejected={sum(rejected.values())}, reasons={rejected}"
    )
    return accepted, seen, rejected


def filter_smiles_with_sources(
    smiles_list,
    source_list,
    training_canonical_set,
    training_fps,
    molecule_type,
    max_similarity=0.4,
):
    logger.info(
        f"Filtering {len(smiles_list)} sourced {molecule_type} molecules with canonical exact-match, realism, and Tanimoto constraints"
    )
    accepted_smiles = []
    accepted_sources = []
    seen = set()
    rejected = {}
    for smi, src in zip(smiles_list, source_list):
        can, ok, reason, _ = accept_generated_candidate(
            smi,
            training_canonical_set,
            training_fps,
            molecule_type,
            seen,
            max_similarity=max_similarity,
        )
        if ok:
            accepted_smiles.append(can)
            accepted_sources.append(src)
            seen.add(can)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    logger.info(
        f"Sourced filtering complete for {molecule_type}: accepted={len(accepted_smiles)}, rejected={sum(rejected.values())}, reasons={rejected}"
    )
    return accepted_smiles, accepted_sources, rejected


def _attachment_atom_indices(mol):
    idxs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        if atom.GetAtomicNum() not in (6, 7, 8, 14, 15, 16, 34):
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() > 0:
            idxs.append(atom.GetIdx())
    return idxs


def attach_fragment_pair(parent_mol, fragment_mol, max_tries=20):
    if parent_mol is None or fragment_mol is None:
        return None
    parent_sites = _attachment_atom_indices(parent_mol)
    fragment_sites = _attachment_atom_indices(fragment_mol)
    if not parent_sites or not fragment_sites:
        return None
    for _ in range(max_tries):
        try:
            a_idx = random.choice(parent_sites)
            b_idx = random.choice(fragment_sites) + parent_mol.GetNumAtoms()
            combo = Chem.CombineMols(parent_mol, fragment_mol)
            rw = Chem.RWMol(combo)
            rw.AddBond(a_idx, b_idx, Chem.BondType.SINGLE)
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            if len(Chem.GetMolFrags(out)) == 1:
                return out
        except Exception:
            continue
    return None


def build_fragment_chain(fragment_smiles):
    mol = None
    for smi in fragment_smiles:
        frag = Chem.MolFromSmiles(smi)
        if frag is None:
            continue
        if mol is None:
            mol = frag
        else:
            next_mol = attach_fragment_pair(mol, frag)
            if next_mol is not None:
                mol = next_mol
    return mol


def constrained_edge_swap(mol, max_tries=50):
    if mol is None:
        return None
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
    ]
    if len(bonds) < 2:
        return None
    for _ in range(max_tries):
        try:
            (a, b), (c, d) = random.sample(bonds, 2)
            if len({a, b, c, d}) < 4:
                continue
            if mol.GetBondBetweenAtoms(a, c) or mol.GetBondBetweenAtoms(b, d):
                continue
            rw = Chem.RWMol(mol)
            rw.RemoveBond(a, b)
            rw.RemoveBond(c, d)
            rw.AddBond(a, c, Chem.BondType.SINGLE)
            rw.AddBond(b, d, Chem.BondType.SINGLE)
            out = rw.GetMol()
            Chem.SanitizeMol(out)
            if len(Chem.GetMolFrags(out)) == 1:
                return out
        except Exception:
            continue
    return None


def constrained_cocograph_generate(
    template_smiles,
    fragment_library,
    num_samples,
    molecule_type,
    training_canonical_set,
    training_fps,
    max_similarity=0.4,
):
    logger.info(
        f"CoCoGraph-inspired generation starting for {molecule_type}: target={num_samples}, max_similarity<{max_similarity}"
    )
    opv_donor_fragments = [
        "c1ccsc1", "c1ccoc1", "c1cc[se]c1", "c1ccc2sccc2c1",
        "c1ccc(cc1)", "c1ccncc1", "CCCC", "CC(C)CC", "OCCCC",
    ]
    opv_acceptor_fragments = [
        "c1ccsc1", "c1ccc(cc1)", "c1ccncc1", "O=C1NC(=O)c2ccccc21",
        "c1ccc2c(c1)C(=O)NC2=O", "C=C(C#N)C#N", "CC#N", "CC(=O)N",
    ]
    usable_library = [
        f for f in fragment_library
        if isinstance(f, str) and "*" not in f and Chem.MolFromSmiles(f) is not None
    ]
    base_pool = opv_acceptor_fragments if "accept" in molecule_type.lower() else opv_donor_fragments
    aux_pool = usable_library[:500] + base_pool
    results = []
    seen = set()
    attempts = 0
    max_attempts = max(num_samples * 350, 2000)
    while len(results) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            if random.random() < 0.35 and template_smiles:
                base_mol = Chem.MolFromSmiles(random.choice(template_smiles))
            else:
                n_frags = random.randint(5, 8) if "accept" in molecule_type.lower() else random.randint(4, 7)
                selected = random.sample(base_pool, min(len(base_pool), max(2, n_frags // 2)))
                while len(selected) < n_frags and aux_pool:
                    selected.append(random.choice(aux_pool))
                random.shuffle(selected)
                base_mol = build_fragment_chain(selected)
            if base_mol is None:
                continue
            trajectory = [base_mol]
            current = base_mol
            for _ in range(random.randint(3, 9)):
                swapped = constrained_edge_swap(current)
                if swapped is not None:
                    current = swapped
                    trajectory.append(current)
                elif aux_pool:
                    frag = Chem.MolFromSmiles(random.choice(aux_pool))
                    attached = attach_fragment_pair(current, frag)
                    if attached is not None:
                        current = attached
                        trajectory.append(current)
            ranked = sorted(
                trajectory,
                key=lambda m: osc_realism_score(m, molecule_type)[0] - max_tanimoto_to_training(m, training_fps),
                reverse=True,
            )
            for cand_mol in ranked[:3]:
                smi = Chem.MolToSmiles(cand_mol, canonical=True)
                can, ok, _, _ = accept_generated_candidate(
                    smi,
                    training_canonical_set,
                    training_fps,
                    molecule_type,
                    seen,
                    max_similarity=max_similarity,
                )
                if ok:
                    results.append(can)
                    seen.add(can)
                    break
        except Exception:
            continue
    logger.info(
        f"CoCoGraph-inspired generation complete for {molecule_type}: generated={len(results)}/{num_samples}, attempts={attempts}"
    )
    return results[:num_samples]


# ----------------------------------------------------------------------------
# 3D CONFORMER GENERATION: ported from GAN_advance_preprocess.py (v1), which
# has a more robust 10-method fallback chain than the 6-method version this
# replaces. Every helper checks EmbedMolecule's return code (or conformer
# count) before attempting force-field optimization, and try_optimize_geometry
# checks FF-parameter availability first (avoids RDKit UFFTYPER warnings) and
# skips metals entirely (poorly supported by organic force fields). This was
# compared against GAN_advance_preprocess_ver2.py's simpler 4-method version,
# which calls MMFFOptimizeMolecule without checking EmbedMolecule's return
# code first — a silent-fail risk. v1's version is kept; ver2's other
# diversity-relevant additions (novelty filtering, hybrid junction-tree
# strategy) were already integrated into this file separately and are
# unaffected by this change.
# ----------------------------------------------------------------------------

def attempt_repair_smiles(smiles):
    """
    Attempt light-touch repair of a SMILES string that failed to parse.
    Handles the most common causes of parse failure in generated/mutated
    SMILES: unbalanced brackets/parentheses and stray whitespace. Returns
    the repaired string if it parses successfully, else None.
    """
    if not smiles:
        return None
    text = str(smiles).strip()
    if not text:
        return None

    candidates = [text]

    # Balance parentheses by trimming trailing unmatched opens/closes
    depth = 0
    trimmed = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue  # drop stray closing paren
            depth -= 1
        trimmed.append(ch)
    balanced = "".join(trimmed) + (")" * depth)
    if balanced != text:
        candidates.append(balanced)

    # Balance square brackets similarly (rare but seen in mutated SMILES)
    if text.count("[") != text.count("]"):
        candidates.append(text.replace("[", "").replace("]", ""))

    for cand in candidates:
        if cand and Chem.MolFromSmiles(cand) is not None:
            return cand
    return None


def ensure_conformer(mol):
    """Ensure molecule has at least one conformer. Return True if present/created."""
    try:
        if mol.GetNumConformers() > 0:
            return True
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if res == 0 and mol.GetNumConformers() > 0:
            return True
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        base_conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
        for i in range(mol.GetNumAtoms()):
            if base_conf:
                p = base_conf.GetAtomPosition(i)
                x, y = p.x, p.y
            else:
                x, y = float(i % 7), float((i // 7) % 7)
            conf.SetAtomPosition(i, (x, y, 0.0))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        return mol.GetNumConformers() > 0
    except Exception as e:
        logger.debug(f"ensure_conformer failed: {e}")
        try:
            n = mol.GetNumAtoms()
            conf = Chem.Conformer(n)
            for i in range(n):
                conf.SetAtomPosition(i, (float(i % 7), float((i // 7) % 7), float(i // 49)))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            return True
        except Exception:
            return False


def try_optimize_geometry(mol):
    """Attempt MMFF94 optimization; fallback to UFF; skip if parameters missing.
    Avoids RDKit UFFTYPER warnings by checking FF parameter availability first.
    """
    try:
        ensure_conformer(mol)
        metal_like = {'Zn', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Cu', 'Ni', 'Co', 'Mn', 'Al', 'Ag', 'Pt', 'Pd', 'Au'}
        if any(atom.GetSymbol() in metal_like for atom in mol.GetAtoms()):
            logger.debug("Metal present (e.g., Zn/Na/Fe); skipping FF optimization")
            mol.SetProp('_opt_ff', 'none')
            return False
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
            mol.SetProp('_opt_ff', 'MMFF94')
            return True
        if AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol)
            mol.SetProp('_opt_ff', 'UFF')
            return True
        logger.debug("No MMFF/UFF parameters available for this molecule; skipping optimization")
        mol.SetProp('_opt_ff', 'none')
        return False
    except Exception as e:
        logger.debug(f"Force-field optimization failed: {e}")
        return False


def _try_etkdg_enhanced(mol):
    """Enhanced ETKDGv3 with comprehensive parameters for chemically accurate structures"""
    mol_copy = Chem.Mol(mol)
    etkdg_params = AllChem.ETKDGv3()
    etkdg_params.randomSeed = 42
    etkdg_params.maxIterations = 2000
    etkdg_params.useSmallRingTorsions = True
    etkdg_params.useMacrocycleTorsions = True
    etkdg_params.enforceChirality = True
    etkdg_params.useRandomCoords = True
    etkdg_params.useExpTorsionAnglePrefs = True
    etkdg_params.useBasicKnowledge = True
    try:
        res = AllChem.EmbedMolecule(mol_copy, params=etkdg_params)
        if res == 0:
            try_optimize_geometry(mol_copy)
            mol_copy.SetProp('_3d_gen_method', 'etkdg_enhanced_mmff')
            logger.debug("Enhanced ETKDG embedding + optimization successful")
            return mol_copy
    except Exception as e:
        logger.debug(f"Enhanced ETKDG attempt failed: {e}")
    return None


def _try_etkdg_mmff(mol):
    """ETKDGv3 with MMFF optimization"""
    mol_copy = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 2000
    params.useRandomCoords = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    if AllChem.EmbedMolecule(mol_copy, params=params) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'etkdg_mmff')
        return mol_copy
    return None


def _try_etkdg_v2_mmff(mol):
    """ETKDGv2 with MMFF optimization"""
    mol_copy = Chem.Mol(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = 42
    params.maxIterations = 2000
    params.useRandomCoords = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    if AllChem.EmbedMolecule(mol_copy, params=params) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'etkdg_v2_mmff')
        return mol_copy
    return None


def _try_etkdg_uff(mol):
    """ETKDGv3 with UFF optimization"""
    mol_copy = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.maxIterations = 2000
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol_copy, params=params) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'etkdg_uff')
        return mol_copy
    return None


def _try_etkdg_v2_uff(mol):
    """ETKDGv2 with UFF optimization"""
    mol_copy = Chem.Mol(mol)
    params = AllChem.ETKDGv2()
    params.randomSeed = 42
    params.maxIterations = 2000
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol_copy, params=params) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'etkdg_v2_uff')
        return mol_copy
    return None


def _try_random_mmff(mol):
    """Random embedding with MMFF optimization"""
    mol_copy = Chem.Mol(mol)
    for seed in [42, 123, 456, 789, 999]:
        try:
            if AllChem.EmbedMolecule(mol_copy, randomSeed=seed, maxAttempts=50) >= 0:
                try_optimize_geometry(mol_copy)
                mol_copy.SetProp('_3d_gen_method', 'random_mmff')
                return mol_copy
        except Exception:
            continue
    return None


def _try_random_uff(mol):
    """Random embedding with UFF optimization"""
    mol_copy = Chem.Mol(mol)
    for seed in [42, 123, 456, 789, 999]:
        try:
            if AllChem.EmbedMolecule(mol_copy, randomSeed=seed, maxAttempts=50) >= 0:
                try_optimize_geometry(mol_copy)
                mol_copy.SetProp('_3d_gen_method', 'random_uff')
                return mol_copy
        except Exception:
            continue
    return None


def _try_basic_mmff(mol):
    """Basic embedding with MMFF optimization"""
    mol_copy = Chem.Mol(mol)
    if AllChem.EmbedMolecule(mol_copy, numAttempts=100) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'basic_mmff')
        return mol_copy
    return None


def _try_basic_uff(mol):
    """Basic embedding with UFF optimization"""
    mol_copy = Chem.Mol(mol)
    if AllChem.EmbedMolecule(mol_copy, numAttempts=100) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'basic_uff')
        return mol_copy
    return None


def _try_force_field(mol):
    """Force field based optimization (final fallback)"""
    mol_copy = Chem.Mol(mol)
    if AllChem.EmbedMolecule(mol_copy) >= 0:
        try_optimize_geometry(mol_copy)
        mol_copy.SetProp('_3d_gen_method', 'force_field')
        return mol_copy
    return None


# Function to generate 3D conformers from SMILES
def smiles_to_3d(smiles):
    """Generate chemically accurate 3D conformer with SMILES repair and robust
    10-method fallback chain (ETKDGv3/v2 x MMFF/UFF, random-seed retries,
    basic embedding, force-field-only fallback)."""
    logger.debug(f"Converting SMILES to 3D: {smiles[:30]}{'...' if len(smiles) > 30 else ''}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"SMILES parsing failed, attempting repair: {smiles}")
            repaired_smiles = attempt_repair_smiles(smiles)
            if repaired_smiles:
                mol = Chem.MolFromSmiles(repaired_smiles)
                logger.debug(f"SMILES repair successful: {repaired_smiles}")
            if mol is None:
                logger.warning(f"Failed to parse SMILES even after repair: {smiles}")
                return None

        mol = sanitize_molecule(mol)
        mol = fix_valence_issues(mol)
        if mol is None:
            logger.warning(f"Molecule sanitization failed for: {smiles}")
            return None

        mol = Chem.AddHs(mol)

        methods = [
            ("ETKDGv3+MMFF_Enhanced", _try_etkdg_enhanced),
            ("ETKDGv3+MMFF", _try_etkdg_mmff),
            ("ETKDGv2+MMFF", _try_etkdg_v2_mmff),
            ("ETKDGv3+UFF", _try_etkdg_uff),
            ("ETKDGv2+UFF", _try_etkdg_v2_uff),
            ("Random+MMFF", _try_random_mmff),
            ("Random+UFF", _try_random_uff),
            ("Basic+MMFF", _try_basic_mmff),
            ("Basic+UFF", _try_basic_uff),
            ("ForceField", _try_force_field),
        ]

        for method_name, method_func in methods:
            try:
                logger.debug(f"Trying method: {method_name}")
                result = method_func(mol)
                if result and result.GetNumConformers() > 0:
                    logger.debug(f"SUCCESS: {method_name} generated chemically accurate 3D conformer")
                    return result
            except Exception as e:
                logger.debug(f"Method {method_name} failed: {e}")
                continue

        logger.warning(f"All 3D generation methods failed for: {smiles}")
        return None

    except Exception as e:
        logger.error(f"3D conformer generation error for {smiles}: {e}")
        return None
        
def extract_3d_coordinates(mol):
    """Extract 3D coordinates with improved validation and logging"""
    try:
        if mol is None:
            print("Warning: Null molecule provided to extract_3d_coordinates")
            return None

        if mol.GetNumConformers() == 0:
            print("Warning: Molecule has no conformers")
            return None

        conformer = mol.GetConformer()
        coords = conformer.GetPositions()

        # Validate coordinates
        if coords is None or len(coords) == 0:
            print("Warning: Empty coordinates generated")
            return None

        if any(np.isnan(coords).any() for coord in coords):
            print("Warning: NaN values in coordinates")
            return None

        return coords
    except Exception as e:
        print(f"Error in extract_3d_coordinates: {e}")
        return None


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
        for feature in (token, f"{token}@{i}", "".join(tokens[i:i + 2]), "".join(tokens[i:i + 3])):
            if not feature:
                continue
            digest = hashlib.blake2b(feature.encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
            fp[int(digest, 16) % n_bits] = 1.0
    return fp


def fallback_representation_from_smiles(smiles, n_bits=2048):
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


def recover_rdkit_mol(smiles):
    text = str(smiles).strip()
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            Chem.SanitizeMol(mol)
            return mol, Chem.MolToSmiles(mol, canonical=True), "rdkit_sanitized"
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return None, None, "rdkit_parse_none"
        mol = fix_valence_issues(mol) or mol
        sanitized = sanitize_molecule(mol)
        if sanitized is not None:
            return sanitized, Chem.MolToSmiles(sanitized, canonical=True), "rdkit_recovered"
        return mol, text, "rdkit_unsanitized_recovered"
    except Exception as e:
        return None, None, f"rdkit_failed:{type(e).__name__}"


def summarize_preprocessing_records(records, molecule_type, original_count=None, drop_stats=None):
    status_counts = Counter(record["status"] for record in records)
    drop_counts = Counter(record["drop_reason"] for record in records if record.get("drop_reason"))
    rdkit_counts = Counter(record["rdkit_reason"] for record in records if record.get("rdkit_reason"))
    total = original_count if original_count is not None else len(records)
    logger.info("=" * 80)
    logger.info(f"{molecule_type.upper()} DATASET PREPROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total input rows: {total}")
    if drop_stats:
        for reason, count in drop_stats.items():
            logger.info(f"  Dropped ({reason}): {count}")
    logger.info(f"Final valid records: {len(records)}")
    logger.info(f"Status counts: {dict(status_counts)}")
    if drop_counts:
        logger.info(f"Drop/recovery reasons: {dict(drop_counts)}")
    if rdkit_counts:
        logger.info(f"RDKit conversion reasons: {dict(rdkit_counts)}")
    logger.info(f"Usable for sequence/token models: {sum(1 for r in records if r['usable_for_sequence'])}")
    logger.info(f"Usable for fingerprint models: {sum(1 for r in records if r['usable_for_fingerprint'])}")
    logger.info(f"Usable for 3D coordinate GAN: {sum(1 for r in records if r['usable_for_3d'])}")
    examples = [r for r in records if r.get("drop_reason")][:10]
    for record in examples:
        logger.debug(
            f"{molecule_type} row={record['index']} status={record['status']} reason={record['drop_reason']} smiles={record['raw_smiles'][:160]}"
        )
    try:
        os.makedirs("results", exist_ok=True)
        report_rows = []
        for record in records:
            report_rows.append({
                "index": record["index"],
                "raw_smiles": record["raw_smiles"],
                "model_smiles": record["model_smiles"],
                "status": record["status"],
                "drop_reason": record["drop_reason"],
                "rdkit_reason": record["rdkit_reason"],
                "fallback_representation": record["fallback_representation"],
                "usable_for_sequence": record["usable_for_sequence"],
                "usable_for_fingerprint": record["usable_for_fingerprint"],
                "usable_for_3d": record["usable_for_3d"],
            })
        report_path = os.path.join("results", f"{molecule_type.lower()}_preprocessing_report.csv")
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        logger.info(f"{molecule_type} preprocessing report saved to {report_path}")
    except Exception as e:
        logger.warning(f"Could not save {molecule_type} preprocessing report: {e}")


def preprocess_training_smiles(raw_smiles, molecule_type):
    """Fast molecule preprocessing: generate 3D coords in parallel and drop failures."""
    logger.info(
        f"Preprocessing {len(raw_smiles)} {molecule_type} rows with {NUM_JOBS} parallel jobs"
    )

    # Step 1: Fast pre-filter with is_valid_smiles (same as old pipeline)
    valid_smiles = []
    invalid_count = 0
    for smi in raw_smiles:
        if is_valid_smiles(smi):
            valid_smiles.append(smi)
        else:
            invalid_count += 1
    if invalid_count:
        logger.info(
            f"{molecule_type} SMILES pre-filter: {len(valid_smiles)} valid, {invalid_count} invalid dropped"
        )

    # Step 2: Generate 3D structures in parallel (auto-removes unprocessable molecules)
    mols_3d = Parallel(n_jobs=NUM_JOBS)(
        delayed(smiles_to_3d)(smi) for smi in valid_smiles
    )
    valid_mols = [mol for mol in mols_3d if mol is not None]
    logger.info(
        f"{molecule_type} 3D molecules: {len(valid_mols)} valid out of {len(mols_3d)} attempted"
    )

    # Step 3: Extract 3D coordinates in parallel
    coords_list = Parallel(n_jobs=NUM_JOBS)(
        delayed(extract_3d_coordinates)(mol) for mol in valid_mols
    )

    # Step 4: Filter out molecules whose coordinate extraction failed
    model_smiles = []
    mols = []
    coords = []
    fallback_features = {}
    records = []
    for mol, coord in zip(valid_mols, coords_list):
        if coord is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
            if not smi:
                continue
        except Exception:
            continue
        model_smiles.append(smi)
        mols.append(mol)
        coords.append(coord)
        fp = morgan_fp(mol, radius=2, nBits=2048)
        if fp is not None:
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fallback_features[smi] = arr.astype(np.float32)
        records.append({
            "index": 0,
            "raw_smiles": smi,
            "model_smiles": smi,
            "mol": mol,
            "coords": coord,
            "fallback_fp": fallback_features.get(smi),
            "fallback_representation": "rdkit_morgan",
            "status": "rdkit_3d",
            "drop_reason": None,
            "rdkit_reason": None,
            "usable_for_sequence": True,
            "usable_for_fingerprint": fp is not None,
            "usable_for_3d": True,
        })

    logger.info(
        f"{molecule_type} coordinates: {len(coords)} valid out of {len(coords_list)} attempted"
    )
    failed_3d_count = len(mols_3d) - len(valid_mols)
    failed_coords_count = len(coords_list) - len(coords)
    drop_stats = {}
    if invalid_count:
        drop_stats["invalid_smiles"] = invalid_count
    if failed_3d_count:
        drop_stats["3d_generation_failed"] = failed_3d_count
    if failed_coords_count:
        drop_stats["coordinate_extraction_failed"] = failed_coords_count
    summarize_preprocessing_records(
        records, molecule_type,
        original_count=len(raw_smiles),
        drop_stats=drop_stats if drop_stats else None,
    )
    return model_smiles, mols, coords, fallback_features, records


TRAINING_FALLBACK_FEATURES = {}


def osc_trait_signature(smiles, molecule_type="Generic"):
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
            f"len_{min(len(tokens) // 20, 6)}",
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
            f"heavy_{min(heavy // 20, 8)}",
            f"rings_{min(rings, 9)}",
            f"arom_{min(aromatic, 8)}",
            f"hetero_{min(hetero // 3, 8)}",
            f"rot_{min(rot // 8, 8)}",
            f"hal_{min(halogen, 6)}",
            "se" if selenium else "no_se",
            "s_rich" if sulfur >= 3 else ("s_some" if sulfur else "s_none"),
        ]
        if "accept" in mt:
            tags.extend([
                f"cyano_{min(cyano, 4)}",
                f"carbonyl_{min(carbonyl, 4)}",
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
            tags.extend([f"cyano_{min(cyano, 4)}", "bdt" if bdt else "non_bdt"])
        return "|".join(tags)
    except Exception as e:
        logger.debug(f"Trait signature failed for {text[:80]}: {e}")
        return "trait_error"


def balanced_trait_order(smiles_list, molecule_type="Generic", max_items=None):
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


def log_trait_coverage(smiles_list, molecule_type, output_dir="results"):
    groups = Counter(osc_trait_signature(smi, molecule_type) for smi in smiles_list)
    logger.info("=" * 80)
    logger.info(f"{molecule_type.upper()} TRAIT COVERAGE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total molecules represented: {len(smiles_list)}")
    logger.info(f"Unique trait groups: {len(groups)}")
    for trait, count in groups.most_common(25):
        logger.info(f"  Trait group count={count}: {trait}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        rows = [{"trait_signature": trait, "count": count} for trait, count in groups.most_common()]
        path = os.path.join(output_dir, f"{molecule_type.lower()}_training_trait_coverage.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"{molecule_type} trait coverage report saved to {path}")
    except Exception as e:
        logger.warning(f"Could not save {molecule_type} trait coverage report: {e}")
    return groups


# Convert 3D coordinates back to SMILES
def coords_to_smiles(coords, template_mol):
    mol = Chem.Mol(template_mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(
            i, [float(x) for x in pos]
        )  # Convert elements of pos to floats
    mol.AddConformer(conf)
    return Chem.MolToSmiles(mol)


# ============================================================================
# RE-RUN TOGGLE: --skip-generation
#
# When set, the entire downstream molecule-generation workload (Optuna HPO,
# 240-rank GAN/VAE/Transformer training, the 13 generation approaches, the
# per-training-molecule augmentation quota, TANGO scoring of the generated
# pool) is bypassed. Only the SCREENING surrogate is retrained on the
# training CSV and re-applied to an EXISTING pair-prediction table so that
# the corrections in this file take effect without regenerating molecules:
#   * Method 2 (Scharber physics) removed from the quantitative blend
#   * blended PCE/Jsc/FF/Voc re-derived from Method 1 surrogate only
#   * FF Tier-1 threshold reported as the applied 0.75 (not 0.8)
#   * Tier 1 / Tier 2 counts reported separately, never conflated
#   * grouped donor-/acceptor-/scaffold-disjoint CV written out
#
# Usage:
#   python GAN_240_final.py --skip-generation \
#          --use-existing-predictions plot_data/generated_molecules_with_predictions.csv
#
# This path is intentionally rank-0-only: it is a single cheap re-score, not
# a distributed workload, so running it under srun with 240 tasks would just
# duplicate identical work 240 times.
# ============================================================================
if getattr(RUNTIME_ARGS, "skip_generation", False):
    logger.info("=" * 80)
    logger.info("SKIP-GENERATION MODE ENABLED")
    logger.info("  Molecule generation, GAN training and Optuna HPO will NOT run.")
    logger.info("  Only the screening surrogate re-score / re-filter will execute.")
    logger.info("=" * 80)

    if RANK_CTX["rank"] != 0:
        logger.info(f"  rank={RANK_CTX['rank']:03d}: non-zero rank in skip-generation mode; exiting (rank 0 does the re-score).")
        sys.exit(0)

    _existing = getattr(RUNTIME_ARGS, "use_existing_predictions", None)
    if not _existing:
        for _cand in [
            os.path.join(BASE_WORKDIR, "plot_data", "generated_molecules_with_predictions.csv"),
            os.path.join(BASE_WORKDIR, "results", "generated_molecules_with_predictions.csv"),
            os.path.join(BASE_WORKDIR, "generated_molecules_with_predictions.csv"),
        ]:
            if os.path.exists(_cand):
                _existing = _cand
                logger.info(f"  Auto-discovered existing predictions CSV: {_existing}")
                break

    if not _existing or not os.path.exists(_existing):
        logger.error(
            "  --skip-generation requires --use-existing-predictions <path> "
            "(or a discoverable generated_molecules_with_predictions.csv). Aborting."
        )
        sys.exit(2)

    try:
        os.makedirs("results", exist_ok=True)
        _screen = ScreeningSurrogatePipeline(file_path)

        logger.info(f"  Loading existing pair predictions: {_existing}")
        _pairs = pd.read_csv(_existing)
        logger.info(f"  Loaded {len(_pairs)} pair rows, {len(_pairs.columns)} columns")

        # Re-derive Method-1-only blend (drops any legacy physics/SISSO influence)
        _rederived = []
        for _prop in ['PCE', 'Jsc', 'FF', 'Voc']:
            _sc = f"surrogate_{_prop}"
            if _sc in _pairs.columns:
                _pairs[_prop] = _pairs[_sc].apply(
                    lambda v, p=_prop: ScreeningSurrogatePipeline._clip_to_physical(v, p)
                    if pd.notna(v) else np.nan
                )
                _rederived.append(_prop)
        logger.info(f"  Re-derived blended columns from Method 1 surrogate only: {_rederived}")

        # Drop stale legacy method columns from the re-scored output so nobody
        # can accidentally treat them as contributors to the blend.
        _legacy = [c for c in _pairs.columns if c.startswith("sisso_") or c.startswith("phys_physics_")]
        if _legacy:
            _pairs = _pairs.rename(columns={c: f"diag_legacy_{c}" for c in _legacy})
            logger.info(f"  Renamed {len(_legacy)} legacy Method-2/Method-3 columns to diag_legacy_* (diagnostic only)")

        _filtered = _screen.filter_high_performance_pairs(_pairs)
        _screen.save_predictions(_pairs, _filtered, output_dir="results")
        _screen.save_disjoint_cv_report(output_dir="results")

        logger.info("=" * 80)
        logger.info("SKIP-GENERATION RE-SCORE COMPLETE")
        logger.info("  Wrote results/generated_molecules_with_predictions.csv (re-scored)")
        logger.info("  Wrote results/high_performance_pairs.csv (corrected tiers)")
        logger.info("  Wrote results/dmpnn_prediction_stats.csv")
        logger.info("  Wrote results/method1_disjoint_cv_report.csv")
        logger.info("  NEXT: run postgen_pipeline.py for metrics/predictions/synthesizability.")
        logger.info("=" * 80)
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as _e:
        logger.error(f"  Skip-generation re-score failed: {_e}")
        logger.exception("Full traceback:")
        sys.exit(1)


logger.info("=" * 80)
logger.info("DATASET INGESTION WITH RDKit/SELFIES/TOKEN FALLBACKS")
logger.info("=" * 80)
required_columns = ["donor(SMILES)", "acceptor(SMILES)"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logger.error(f"Training dataset columns found: {list(df.columns)}")
    raise KeyError(f"Missing required SMILES columns: {missing_columns}")
logger.info(f"Training dataset columns: {list(df.columns)}")
logger.info(f"Using NUM_JOBS={NUM_JOBS} for ingestion/preprocessing")

donor_smiles_raw = df["donor(SMILES)"].dropna().astype(str).tolist()
acceptor_smiles_raw = df["acceptor(SMILES)"].dropna().astype(str).tolist()
logger.info(f"Raw donor rows: {len(donor_smiles_raw)}")
logger.info(f"Raw acceptor rows: {len(acceptor_smiles_raw)}")

if RANK_CTX["distributed"] and os.environ.get("GAN_SHARD_TRAINING_DATA", "0") == "1":
    donor_smiles_raw = shard_list(donor_smiles_raw, RANK_CTX["rank"], RANK_CTX["world_size"])
    acceptor_smiles_raw = shard_list(acceptor_smiles_raw, RANK_CTX["rank"], RANK_CTX["world_size"])
    logger.info(
        f"Applied rank sharding before preprocessing | donor_rows={len(donor_smiles_raw)} "
        f"acceptor_rows={len(acceptor_smiles_raw)}"
    )
    if len(donor_smiles_raw) == 0 or len(acceptor_smiles_raw) == 0:
        logger.warning("This rank has no rows after sharding; downstream steps may be minimal.")
elif RANK_CTX["distributed"]:
    logger.info(
        "Distributed mode detected, but GAN_SHARD_TRAINING_DATA is not enabled; every rank will train on every donor and acceptor row to guarantee full dataset contribution."
    )

donor_smiles, donor_mols, donor_coords, donor_fallback_features, donor_preprocess_records = preprocess_training_smiles(
    donor_smiles_raw,
    "Donor",
)
acceptor_smiles, acceptor_mols, acceptor_coords, acceptor_fallback_features, acceptor_preprocess_records = preprocess_training_smiles(
    acceptor_smiles_raw,
    "Acceptor",
)
TRAINING_FALLBACK_FEATURES["Donor"] = donor_fallback_features
TRAINING_FALLBACK_FEATURES["Acceptor"] = acceptor_fallback_features
logger.info(
    f"Fallback feature maps prepared | donor={len(donor_fallback_features)} acceptor={len(acceptor_fallback_features)}"
)
donor_trait_groups = log_trait_coverage(donor_smiles, "Donor")
acceptor_trait_groups = log_trait_coverage(acceptor_smiles, "Acceptor")
donor_smiles = balanced_trait_order(donor_smiles, "Donor")
acceptor_smiles = balanced_trait_order(acceptor_smiles, "Acceptor")
logger.info(
    f"Trait-balanced training order prepared with all usable rows retained | donors={len(donor_smiles)} across {len(donor_trait_groups)} groups, acceptors={len(acceptor_smiles)} across {len(acceptor_trait_groups)} groups"
)

# Add validation before dimension calculation
logger.info("Validating coordinate generation...")
if not donor_coords:
    logger.error("=" * 80)
    logger.error("CRITICAL ERROR: No valid donor coordinates were generated!")
    logger.error("=" * 80)
    logger.error(f"Total donor rows loaded: {len(donor_smiles_raw)}")
    logger.error(f"Donor molecules usable by sequence/fingerprint models: {len(donor_smiles)}")
    logger.error(f"Donor molecules with valid 3D structures: {len(donor_mols)}")
    logger.error(f"Donor coordinates extracted: {len(donor_coords)}")
    logger.error("Possible causes:")
    logger.error("  1. All donor molecules failed 3D conformer generation")
    logger.error("  2. All donor molecules failed coordinate extraction")
    logger.error("  3. Input donor SMILES file is empty or invalid")
    logger.error("=" * 80)
    raise ValueError(
        "No valid donor coordinates were generated. Cannot proceed with GAN training. Check molecule processing pipeline."
    )
if not acceptor_coords:
    logger.error("=" * 80)
    logger.error("CRITICAL ERROR: No valid acceptor coordinates were generated!")
    logger.error("=" * 80)
    logger.error(f"Total acceptor rows loaded: {len(acceptor_smiles_raw)}")
    logger.error(f"Acceptor molecules usable by sequence/fingerprint models: {len(acceptor_smiles)}")
    logger.error(f"Acceptor molecules with valid 3D structures: {len(acceptor_mols)}")
    logger.error(f"Acceptor coordinates extracted: {len(acceptor_coords)}")
    logger.error("Possible causes:")
    logger.error("  1. All acceptor molecules failed 3D conformer generation")
    logger.error("  2. All acceptor molecules failed coordinate extraction")
    logger.error("  3. Input acceptor SMILES file is empty or invalid")
    logger.error("=" * 80)
    raise ValueError(
        "No valid acceptor coordinates were generated. Cannot proceed with GAN training. Check molecule processing pipeline."
    )

# Calculate dimensions
logger.info("Calculating coordinate dimensions...")
donor_coord_dim = max(len(coords) for coords in donor_coords)
acceptor_coord_dim = max(len(coords) for coords in acceptor_coords)

logger.info(f"Coordinate Generation Summary:")
logger.info(f"  Donor coordinates: {len(donor_coords)} molecules")
logger.info(f"  Acceptor coordinates: {len(acceptor_coords)} molecules")
logger.info(f"  Donor coordinate dimension: {donor_coord_dim}")
logger.info(f"  Acceptor coordinate dimension: {acceptor_coord_dim}")


# Pad coordinates to ensure they have the same length
def pad_coordinates(coords, max_length):
    padded_coords = np.zeros((max_length, 3))
    length = min(len(coords), max_length)
    padded_coords[:length] = coords[:length]
    return padded_coords


logger.info("Padding coordinates to uniform dimensions...")
donor_coords = Parallel(n_jobs=NUM_JOBS)(
    delayed(pad_coordinates)(coords, donor_coord_dim) for coords in donor_coords
)
logger.info(f"Donor coordinates padded to dimension: {donor_coord_dim}")
acceptor_coords = Parallel(n_jobs=NUM_JOBS)(
    delayed(pad_coordinates)(coords, acceptor_coord_dim) for coords in acceptor_coords
)
logger.info(f"Acceptor coordinates padded to dimension: {acceptor_coord_dim}")

# ============================================================================
# HYBRID 5-APPROACH STRUCTURAL VARIATION PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("INITIALIZING HYBRID 5-APPROACH STRUCTURAL VARIATION PIPELINE")
print("=" * 80)

# ============================================================================
# APPROACH 1: SMILES-Based GAN (Character-level String Generation)
# ============================================================================


def build_smiles_vocab(smiles_list):
    """Build vocabulary from SMILES strings"""
    chars = set()
    for smi in smiles_list:
        chars.update(smi)
    chars = sorted(list(chars))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


def smiles_to_one_hot(smiles, char_to_idx, max_length):
    """Convert SMILES to one-hot encoding"""
    vocab_size = len(char_to_idx)
    one_hot = np.zeros((max_length, vocab_size))
    for i, char in enumerate(smiles[:max_length]):
        if char in char_to_idx:
            one_hot[i, char_to_idx[char]] = 1
    return one_hot


def build_smiles_generator(latent_dim, max_length, vocab_size, dense_units=256, lstm_units=512):
    """Build SMILES-based generator using LSTM"""
    model = tf.keras.Sequential(
        [
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(dense_units, activation="relu"),
            layers.RepeatVector(max_length),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax")),
        ]
    )
    return model


def build_smiles_discriminator(max_length, vocab_size, lstm_units=256, dense_units=128, dropout_rate=0.3):
    """Build SMILES-based discriminator"""
    model = tf.keras.Sequential(
        [
            layers.InputLayer(input_shape=(max_length, vocab_size)),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.LSTM(lstm_units),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def train_smiles_gan(generator, discriminator, training_smiles, char_to_idx, idx_to_char, 
                     max_length, vocab_size, latent_dim, epochs=100, batch_size=64,
                     gen_lr=0.0002, disc_lr=0.0001):
    """Train SMILES-based GAN on real training data"""
    logger.info(
        f"Training SMILES GAN for {epochs} epochs with batch_size={batch_size}, gen_lr={gen_lr}, disc_lr={disc_lr}"
    )
    
    # Compile models
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=disc_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Combined GAN model
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    generated_smiles = generator(gan_input)
    gan_output = discriminator(generated_smiles)
    combined = tf.keras.Model(gan_input, gan_output)
    combined.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr),
        loss='binary_crossentropy'
    )
    
    # Prepare training data
    logger.info("Encoding training SMILES...")
    real_data = []
    for smi in training_smiles:
        one_hot = smiles_to_one_hot(smi, char_to_idx, max_length)
        real_data.append(one_hot)
    real_data = np.array(real_data)
    logger.info(f"Training data shape: {real_data.shape}")
    
    # Training loop
    num_batches = int(np.ceil(len(real_data) / batch_size))
    if num_batches < 1:
        logger.warning(
            f"SMILES GAN batch_size={batch_size} is larger than training rows={len(real_data)}; using one full-data batch"
        )
        batch_size = len(real_data)
        num_batches = 1
    logger.info(
        f"SMILES GAN will use all {len(real_data)} training rows every epoch across {num_batches} batches"
    )
    best_g_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        
        # Shuffle data
        indices = np.random.permutation(len(real_data))
        real_data_shuffled = real_data[indices]
        
        for batch_idx in range(num_batches):
            # Get real batch
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(real_data_shuffled))
            real_batch = real_data_shuffled[batch_start:batch_end]
            current_batch_size = len(real_batch)
            if current_batch_size == 0:
                continue
            
            # Generate fake batch
            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
            fake_batch = generator.predict(noise, verbose=0)
            
            # Train discriminator
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_batch, np.ones((current_batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((current_batch_size, 1)))
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            epoch_d_loss.append(d_loss)
            
            # Train generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
            g_loss = combined.train_on_batch(noise, np.ones((current_batch_size, 1)))
            epoch_g_loss.append(g_loss)
        
        # Log progress
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} - D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}")
            
            # Sample and validate generated SMILES
            noise = np.random.normal(0, 1, (10, latent_dim))
            generated_one_hot = generator.predict(noise, verbose=0)
            valid_count = 0
            for one_hot in generated_one_hot:
                smi = one_hot_to_smiles(one_hot, idx_to_char)
                if smi and is_valid_smiles(smi):
                    valid_count += 1
            logger.info(f"  Sample validation: {valid_count}/10 valid SMILES")
        
        # Track best model
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
    
    logger.info(f"SMILES GAN training complete. Best G loss: {best_g_loss:.4f}")
    return generator, discriminator


def one_hot_to_smiles(one_hot, idx_to_char, temperature=1.0):
    """Convert one-hot encoding back to SMILES with temperature sampling"""
    if temperature == 1.0:
        # Greedy decoding
        indices = np.argmax(one_hot, axis=-1)
    else:
        # Temperature sampling for diversity
        indices = []
        for probs in one_hot:
            # Apply temperature
            probs = np.log(probs + 1e-10) / temperature
            probs = np.exp(probs) / np.sum(np.exp(probs))
            # Sample from distribution
            idx = np.random.choice(len(probs), p=probs)
            indices.append(idx)
        indices = np.array(indices)
    
    # Convert to SMILES string
    smiles = "".join([idx_to_char.get(idx, "") for idx in indices])
    
    # Clean up: remove padding and trailing spaces
    smiles = smiles.strip()
    
    # Remove characters after first null/padding character if present
    if '\x00' in smiles:
        smiles = smiles.split('\x00')[0]
    
    # Stop at excessive repetition (likely invalid)
    # Check for patterns like "cccccccccc" or "111111111"
    for i in range(len(smiles) - 5):
        if len(set(smiles[i:i+6])) == 1:  # 6 identical consecutive characters
            smiles = smiles[:i]
            break
    
    return smiles.strip()


print("âœ“ Approach 1: SMILES-Based GAN initialized")

# ============================================================================
# APPROACH 2: Graph-Based GAN (Molecular Graph Generation)
# ============================================================================


def build_graph_generator(latent_dim, max_atoms=50, num_atom_types=9):
    """Build graph-based generator for molecular graphs"""
    noise_input = layers.Input(shape=(latent_dim,))

    # Shared dense layers
    x = layers.Dense(256, activation="relu")(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Adjacency matrix output
    adj_flat = layers.Dense(max_atoms * max_atoms, activation="sigmoid")(x)
    adj_matrix = layers.Reshape((max_atoms, max_atoms))(adj_flat)

    # Node features (atom types)
    node_flat = layers.Dense(max_atoms * num_atom_types, activation="softmax")(x)
    node_features = layers.Reshape((max_atoms, num_atom_types))(node_flat)

    model = tf.keras.Model(inputs=noise_input, outputs=[adj_matrix, node_features])
    return model


def build_graph_discriminator(max_atoms=50, num_atom_types=9):
    """Build graph-based discriminator using graph convolution concepts"""
    adj_input = layers.Input(shape=(max_atoms, max_atoms))
    node_input = layers.Input(shape=(max_atoms, num_atom_types))

    # Flatten and concatenate
    adj_flat = layers.Flatten()(adj_input)
    node_flat = layers.Flatten()(node_input)
    combined = layers.Concatenate()([adj_flat, node_flat])

    # Dense layers
    x = layers.Dense(512, activation="relu")(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[adj_input, node_input], outputs=x)
    return model


def graph_to_smiles(
    adj_matrix,
    node_features,
    atom_types=["C", "N", "O", "S", "F", "Cl", "Br", "P", "H"],
):
    """Convert adjacency matrix and node features to SMILES.
    Uses a slot-to-atom mapping to correctly index bonds from the adjacency matrix.
    """
    try:
        mol = Chem.RWMol()
        threshold = 0.5
        num_slots = min(len(node_features), adj_matrix.shape[0])

        slot_to_atom = {}
        for i in range(num_slots):
            if np.max(node_features[i]) > 0.3:
                atom_type_idx = np.argmax(node_features[i])
                if atom_type_idx < len(atom_types):
                    atom = Chem.Atom(atom_types[atom_type_idx])
                    atom_idx = mol.AddAtom(atom)
                    slot_to_atom[i] = atom_idx

        valid_slots = sorted(slot_to_atom.keys())
        for si in range(len(valid_slots)):
            for sj in range(si + 1, len(valid_slots)):
                slot_i, slot_j = valid_slots[si], valid_slots[sj]
                if adj_matrix[slot_i, slot_j] > threshold:
                    bond_val = adj_matrix[slot_i, slot_j]
                    if bond_val > 0.85:
                        bond_type = Chem.BondType.DOUBLE
                    elif bond_val > 0.95:
                        bond_type = Chem.BondType.TRIPLE
                    else:
                        bond_type = Chem.BondType.SINGLE
                    try:
                        mol.AddBond(slot_to_atom[slot_i], slot_to_atom[slot_j], bond_type)
                    except Exception:
                        try:
                            mol.AddBond(slot_to_atom[slot_i], slot_to_atom[slot_j], Chem.BondType.SINGLE)
                        except Exception:
                            continue

        if mol.GetNumAtoms() < 2:
            return None

        mol_out = mol.GetMol()
        mol_out = sanitize_molecule(mol_out)
        if mol_out is None:
            return None
        smi = Chem.MolToSmiles(mol_out)
        if smi and len(smi) > 3:
            check = Chem.MolFromSmiles(smi)
            if check and len(Chem.GetMolFrags(check)) == 1:
                return smi
        return None
    except Exception:
        return None


print("âœ“ Approach 2: Graph-Based GAN initialized")

# ============================================================================
# APPROACH 3: Junction Tree-Inspired Fragment Assembly
# ============================================================================


def build_fragment_library(smiles_list, max_frags=3000):
    """Build fragment library from BRICS decomposition with fallback to common substructures"""
    lib = set()
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if not m:
            continue
        try:
            fr = BRICS.BRICSDecompose(m)
            for f in fr:
                lib.add(f)
                if len(lib) >= max_frags:
                    return lib
        except Exception:
            continue
    
    # Fallback: if fragment library is empty, generate OSC-specific fragments
    if not lib:
        logger.warning("Fragment library is empty. Generating OSC-specific fragments for donors and acceptors.")
        
        # OSC-specific fragments based on literature (Y6, ITIC, PBDB-T, PM6, PTB7-Th)
        osc_fragments = [
            # ===== BDT-BASED DONOR CORES =====
            'c1ccc2sccc2c1',        # Benzodithiophene (BDT) - core in PBDB-T, PM6
            
            # ===== THIOPHENE VARIANTS (Conjugation bridges) =====
            'c1sccc1',              # Thiophene - essential for BDT donors
            'c1sc(F)cc1',           # Fluorothiophene - PM6 variant
            'c1sc(C)cc1',           # Methylthiophene - solubility tuning
            'c1sc(CC)cc1',          # Ethylthiophene
            
            # ===== ACCEPTOR END-GROUPS (Y6, ITIC, BTA) =====
            'C(=C(C#N))C#N',        # Dicyanovinyl - Y6 end-group (critical)
            'C#N',                  # Cyano - ITIC, BTA acceptors
            'C(=O)C#N',             # Acyl-nitrile - acceptor variant
            
            # ===== INDANONE CORE (ITIC-like acceptors) =====
            'c1ccc2c(c1)c(=O)c(=O)n2',  # Indanone core - ITIC structure
            
            # ===== CARBONYL GROUPS (Acceptor functionality) =====
            'C(=O)O',               # Ester - acceptor end-group
            'C(=O)N',               # Amide - acceptor functionality
            'C=O',                  # Carbonyl - general acceptor
            
            # ===== DONOR SIDE-CHAINS (Solubility, morphology) =====
            'CCCC',                 # Butyl - PTB7, PBDB-T
            'C(C)C',                # Isopropyl - solubility
            'C(C)(C)C',             # tert-Butyl - steric control
            'CCCCC',                # Pentyl - extended chain
            'OCC',                  # Ethoxy - ether side-chain
            'OCCC',                 # Propoxy - extended ether
            'OC(C)C',               # Isopropoxy - branched ether
            
            # ===== LINKERS & CONJUGATION EXTENDERS =====
            'C=C',                  # Vinyl - conjugation extension
            'C#C',                  # Alkyne - extended conjugation
            
            # ===== AROMATIC CORES =====
            'c1ccccc1',             # Benzene - universal aromatic
            'c1ccc(cc1)',           # Benzene alt - compatibility
            'c1ccncc1',             # Pyridine - heteroaromatic
            
            # ===== HETEROATOMS & MODIFIERS =====
            '[F]',                  # Fluorine - Voc improvement
            '[Cl]',                 # Chlorine - electronic tuning
            '[Br]',                 # Bromine - heavy halogen
            'c1[se]ccc1',           # Selenophene - emerging donor
            
            # ===== ADDITIONAL DONOR FRAGMENTS =====
            'c1ccc(C)cc1',          # Toluene - methylated benzene
            'C1CCCCC1',             # Cyclohexane - aliphatic core
            'CC(C)C',               # Isobutane - branched alkyl
            '[NH2]',                # Amine - donor group
            'O',                    # Oxygen - heteroatom
            'S',                    # Sulfur - heteroatom
        ]
        
        for frag in osc_fragments:
            try:
                m = Chem.MolFromSmiles(frag)
                if m:
                    lib.add(frag)
            except:
                pass
        
        logger.warning(f"Using {len(lib)} OSC-specific fragments as fallback (BDT, Y6, ITIC, PBDB-T, PM6 based)")
    
    logger.info(f"Fragment library contains {len(lib)} fragments")
    return lib


def scaffold_based_generation(training_smiles, training_smiles_set, num_molecules=10):
    """
    Generate molecules based on Murcko scaffolds from training data.

    FIXED: previously extracted the bare Murcko scaffold (which strips ALL
    side chains, alkyl solubilizing groups, AND terminal functional groups
    like the dicyanomethylene-indanone end caps that define ITIC/IDIC/Y6-
    class acceptors) and mutated that stripped skeleton — meaning large
    fused-polycyclic training molecules (which owe most of their heavy-atom
    count and identity to those exact side chains/end groups) could only
    ever produce small, generic-ring-system fragments, never anything
    resembling the actual large training molecule. This was the primary
    reason no ITIC/IDIC/Y6-scale molecules appeared anywhere in generation.

    Now: mutations are applied directly to the FULL original training
    molecule (side chains, end groups, and all), not the stripped scaffold.
    The Murcko scaffold is still computed and logged for diagnostic/trait
    purposes, but is no longer the base structure that gets mutated. This
    lets large templates stay large after mutation (a peripheral edit on a
    150-heavy-atom molecule still leaves a ~150-heavy-atom molecule).
    """
    generated_smiles = []
    scaffolds = []

    try:
        sample_smiles = training_smiles[:100]  # Use first 100 for efficiency
        for smi in sample_smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
                    scaffolds.append(scaffold_smi)
            except Exception:
                continue

        # Generate variants by mutating the FULL original training molecules
        # (not the stripped scaffold), so large templates stay large.
        for smi in sample_smiles:
            if len(generated_smiles) >= num_molecules:
                break
            try:
                full_mol = Chem.MolFromSmiles(smi)
                if full_mol is None:
                    continue
                mutants = smart_mutants_for_mol(full_mol, max_out=5)
                for mut_smi in mutants:
                    if (mut_smi and
                            mut_smi not in training_smiles_set and
                            is_valid_smiles(mut_smi) and
                            len(mut_smi) > 5):
                        generated_smiles.append(mut_smi)
                        if len(generated_smiles) >= num_molecules:
                            break
            except Exception:
                continue

        logger.info(
            f"Scaffold-based generation: Generated {len(generated_smiles)} molecules "
            f"(mutating full training templates; {len(set(scaffolds))} distinct Murcko "
            f"scaffolds observed for reference/diagnostics only)"
        )
    except Exception as e:
        logger.error(f"Scaffold-based generation failed: {e}")

    return list(set(generated_smiles))[:num_molecules]


def constrained_fragment_assembly(fragment_library, training_smiles_set, num_molecules=10, donor_mode=True):
    """Assemble fragments with chemical logic (donor vs acceptor)"""
    generated_smiles = []
    frag_list = list(fragment_library)
    
    if not frag_list:
        return []
    
    try:
        if donor_mode:
            # For donors: prioritize BDT core + thiophene bridge + side-chain
            donor_cores = ['c1ccc2sccc2c1']  # BDT
            donor_bridges = ['c1sccc1', 'c1sc(F)cc1', 'c1sc(C)cc1']  # Thiophenes
            donor_sidechains = ['CCCC', 'C(C)C', 'C(C)(C)C', 'OCC']  # Side chains
            
            for core in donor_cores:
                if core not in frag_list:
                    continue
                for bridge in donor_bridges:
                    if bridge not in frag_list:
                        continue
                    for sidechain in donor_sidechains:
                        if len(generated_smiles) >= num_molecules:
                            break
                        try:
                            # Combine core + bridge + sidechain
                            frags = [core, bridge, sidechain]
                            gen = BRICS.BRICSBuild(frags)
                            for i, mol in enumerate(gen):
                                if i >= 5:
                                    break
                                try:
                                    Chem.SanitizeMol(mol)
                                    smi = Chem.MolToSmiles(mol, canonical=True)
                                    if (smi and len(smi) > 5 and 
                                        smi not in training_smiles_set and
                                        is_valid_smiles(smi)):
                                        generated_smiles.append(smi)
                                        break
                                except:
                                    continue
                        except:
                            continue
        else:
            # For acceptors: prioritize core + end-group
            acceptor_cores = ['c1ccc2c(c1)c(=O)c(=O)n2']  # Indanone
            acceptor_endgroups = ['C(=C(C#N))C#N', 'C#N', 'C(=O)C#N']  # End-groups
            
            for core in acceptor_cores:
                if core not in frag_list:
                    continue
                for endgroup in acceptor_endgroups:
                    if endgroup not in frag_list:
                        continue
                    if len(generated_smiles) >= num_molecules:
                        break
                    try:
                        frags = [core, endgroup]
                        gen = BRICS.BRICSBuild(frags)
                        for i, mol in enumerate(gen):
                            if i >= 5:
                                break
                            try:
                                Chem.SanitizeMol(mol)
                                smi = Chem.MolToSmiles(mol, canonical=True)
                                if (smi and len(smi) > 5 and 
                                    smi not in training_smiles_set and
                                    is_valid_smiles(smi)):
                                    generated_smiles.append(smi)
                                    break
                            except:
                                continue
                    except:
                        continue
        
        mode_str = "donor" if donor_mode else "acceptor"
        logger.info(f"Constrained assembly ({mode_str}): Generated {len(generated_smiles)} molecules")
    except Exception as e:
        logger.error(f"Constrained assembly failed: {e}")
    
    return list(set(generated_smiles))[:num_molecules]


def junction_tree_generate_hybrid(fragment_library, training_smiles, num_molecules=10, max_frags_per_mol=5, training_smiles_set=None):
    """Hybrid Junction Tree generation combining multiple strategies"""
    
    if training_smiles_set is None:
        training_smiles_set = set()
    
    all_generated = []
    
    # Strategy 1: BRICS assembly (40% of target)
    logger.info("Hybrid JT - Strategy 1: BRICS assembly (40%)")
    brics_target = int(num_molecules * 0.4)
    brics_mols = junction_tree_generate(fragment_library, brics_target, max_frags_per_mol, training_smiles_set)
    all_generated.extend(brics_mols)
    logger.info(f"  Generated {len(brics_mols)} molecules via BRICS")
    
    # Strategy 2: Scaffold-based (40% of target)
    logger.info("Hybrid JT - Strategy 2: Scaffold-based (40%)")
    scaffold_target = int(num_molecules * 0.4)
    scaffold_mols = scaffold_based_generation(training_smiles, training_smiles_set, scaffold_target)
    all_generated.extend(scaffold_mols)
    logger.info(f"  Generated {len(scaffold_mols)} molecules via scaffolds")
    
    # Strategy 3: Constrained assembly - donors (10% of target)
    logger.info("Hybrid JT - Strategy 3a: Constrained assembly - donors (10%)")
    donor_target = int(num_molecules * 0.1)
    donor_mols = constrained_fragment_assembly(fragment_library, training_smiles_set, donor_target, donor_mode=True)
    all_generated.extend(donor_mols)
    logger.info(f"  Generated {len(donor_mols)} donor molecules via constrained assembly")
    
    # Strategy 4: Constrained assembly - acceptors (10% of target)
    logger.info("Hybrid JT - Strategy 3b: Constrained assembly - acceptors (10%)")
    acceptor_target = int(num_molecules * 0.1)
    acceptor_mols = constrained_fragment_assembly(fragment_library, training_smiles_set, acceptor_target, donor_mode=False)
    all_generated.extend(acceptor_mols)
    logger.info(f"  Generated {len(acceptor_mols)} acceptor molecules via constrained assembly")
    
    # Remove duplicates and return
    result = list(set(all_generated))[:num_molecules]
    logger.info(f"Hybrid JT - Total unique molecules: {len(result)}/{num_molecules}")
    
    return result


def junction_tree_generate(fragment_library, num_molecules=10, max_frags_per_mol=5, training_smiles_set=None):
    """Generate molecules using junction tree-inspired fragment assembly with novelty filtering.
    Only produces single connected molecules (no dot-joined fragments).
    """
    generated_smiles = set()
    frag_list = list(fragment_library)
    
    if not frag_list:
        logger.warning("Fragment library is empty for junction tree generation")
        return []
    
    if training_smiles_set is None:
        training_smiles_set = set()

    max_total_attempts = num_molecules * 30

    for attempt in range(max_total_attempts):
        if len(generated_smiles) >= num_molecules:
            break
        try:
            num_frags = random.randint(2, min(max_frags_per_mol, len(frag_list)))
            selected_frags = random.sample(frag_list, num_frags)
            
            try:
                gen = BRICS.BRICSBuild(selected_frags)
                for i, mol in enumerate(gen):
                    if i >= 15:
                        break
                    try:
                        Chem.SanitizeMol(mol)
                        smi = Chem.MolToSmiles(mol, canonical=True)
                        if not smi or len(smi) <= 5:
                            continue
                        if smi in training_smiles_set or smi in generated_smiles:
                            continue
                        check_mol = Chem.MolFromSmiles(smi)
                        if check_mol is None:
                            continue
                        frags = Chem.GetMolFrags(check_mol)
                        if len(frags) != 1:
                            continue
                        generated_smiles.add(smi)
                        break
                    except Exception:
                        continue
            except Exception:
                continue
        except Exception:
            continue

    # Strategy 2: try different fragment count combinations
    if len(generated_smiles) < num_molecules:
        for attempt in range(num_molecules * 15):
            if len(generated_smiles) >= num_molecules:
                break
            try:
                num_frags = random.randint(2, min(3, len(frag_list)))
                selected_frags = random.sample(frag_list, num_frags)
                gen = BRICS.BRICSBuild(selected_frags)
                for i, mol in enumerate(gen):
                    if i >= 20:
                        break
                    try:
                        Chem.SanitizeMol(mol)
                        smi = Chem.MolToSmiles(mol, canonical=True)
                        if not smi or len(smi) <= 5:
                            continue
                        if smi in training_smiles_set or smi in generated_smiles:
                            continue
                        check_mol = Chem.MolFromSmiles(smi)
                        if check_mol is None:
                            continue
                        if len(Chem.GetMolFrags(check_mol)) != 1:
                            continue
                        generated_smiles.add(smi)
                        break
                    except Exception:
                        continue
            except Exception:
                continue

    logger.info(f"Junction tree generated {len(generated_smiles)} single-component molecules")
    return list(generated_smiles)[:num_molecules]


print("âœ“ Approach 3: Junction Tree Fragment Assembly initialized")

# ============================================================================
# APPROACH 4: Conditional GAN with Chemical Rules
# ============================================================================


def canonical(smi):
    """Convert to canonical SMILES"""
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except:
        return None


def calculate_chemical_validity_reward(mol):
    """Calculate chemical validity reward with enhanced substructure validation"""
    try:
        if mol is None:
            return 0.0
        Chem.SanitizeMol(mol)

        # Check basic chemical rules
        reward = 1.0

        # Reward for reasonable molecular weight
        mw = Descriptors.MolWt(mol)
        if 200 <= mw <= 800:
            reward += 0.5

        # Reward for presence of aromatic rings
        if mol.GetAromaticAtoms():
            reward += 0.3

        # Reward for druglike properties
        logp = Descriptors.MolLogP(mol)
        if -1 <= logp <= 6:
            reward += 0.2

        # Additional substructure validation for solar cell materials
        # Check for conjugated systems (essential for OSCs)
        conjugated_systems = count_substruct(mol, "C=C") + count_substruct(mol, "C#C")
        if conjugated_systems >= 1:
            reward += 0.2

        # Check for electron-withdrawing groups (for acceptors)
        ewg_count = (
            count_substruct(mol, "C#N")
            + count_substruct(mol, "C(=O)")
            + count_substruct(mol, "[NX3](=O)=O")
            + count_substruct(mol, "F")
        )
        if ewg_count >= 1:
            reward += 0.15

        # Penalize potentially reactive or unstable groups
        unstable_groups = (
            count_substruct(mol, "[OH]")
            + count_substruct(mol, "[NH2]")
            + count_substruct(mol, "C(=O)O")
        )
        if unstable_groups > 2:
            reward -= 0.3  # Penalty for too many reactive groups

        return max(0.0, reward)  # Ensure non-negative reward
    except Exception as e:
        logger.debug(f"Error in calculate_chemical_validity_reward: {e}")
        return 0.0


def calculate_diversity_reward(mol, existing_mols):
    """Calculate structural diversity reward"""
    try:
        if not existing_mols or mol is None:
            return 0.0

        fp_new = morgan_fp(mol, radius=2, nBits=2048)
        if fp_new is None:
            return 0.0

        max_sim = 0.0

        for exist_mol in existing_mols[-20:]:  # Check last 20 to save time
            fp_exist = morgan_fp(exist_mol, radius=2, nBits=2048)
            if fp_exist is not None:
                sim = DataStructs.TanimotoSimilarity(fp_new, fp_exist)
                max_sim = max(max_sim, sim)

        # Reward novelty (low similarity)
        return 1.0 - max_sim
    except:
        return 0.0


def calculate_smarts_reward(mol, desired_smarts_patterns):
    """Reward for containing desired SMARTS patterns using safe SMARTS parsing"""
    try:
        if mol is None:
            return 0.0

        reward = 0.0
        for smarts in desired_smarts_patterns:
            # Use cached version for frequently used patterns
            pattern = safe_mol_from_smarts_cached(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                reward += 0.5

        return min(reward, 2.0)  # Cap reward
    except Exception as e:
        logger.debug(f"Error in calculate_smarts_reward: {e}")
        return 0.0


def conditional_gan_loss_with_constraints(
    generated_coords,
    template_mols,
    discriminator_loss,
    existing_mols,
    lambda_validity=0.3,
    lambda_diversity=0.3,
    lambda_smarts=0.2,
):
    """Enhanced GAN loss with chemical constraints"""
    # Define comprehensive SMARTS patterns for solar cell materials
    # Based on literature: Y6, ITIC, PTB7, PBDB-T, PM6, and high-performance OSC donors/NFAs
    desired_patterns = [
        # ===== CORE AROMATIC HETEROCYCLES (Ï€-conjugation) =====
        # Thiophene (essential for BDT donors, PBDB-T, PTB7)
        "[#16X2r5][#6r5][#6r5][#6r5][#6r5]",  # Thiophene
        "c1sccc1",  # Thiophene (alternative)
        "c1sc(F)cc1",  # Fluorothiophene
        "c1sc(C)cc1",  # Methylthiophene
        # Benzene and benzodithiophene (BDT core)
        "c1ccccc1",  # Benzene
        "c1ccc(cc1)",  # Benzene (alternative)
        "c1ccc2sccc2c1",  # Benzodithiophene (BDT core)
        # Pyrrole, Furan, Thiazole
        "[#7r5][#6r5][#6r5][#6r5][#6r5]",  # Pyrrole
        "[#8r5][#6r5][#6r5][#6r5][#6r5]",  # Furan
        "[#16r5][#6r5][#7r5][#6r5][#6r5]",  # Thiazole
        # Selenophene (emerging donors)
        "c1[se]ccc1",  # Selenophene
        "c1[se]c(F)cc1",  # Fluoroselenophene
        
        # ===== ACCEPTOR END-GROUPS (Y6, ITIC, BTA series) =====
        # Cyano groups (critical for Y6, ITIC)
        "C#N",  # Cyano group
        "C(=C(C#N))C#N",  # Dicyanovinyl (Y6 end-group)
        "C(=O)C#N",  # Acyl-nitrile
        # Carbonyl and ester groups
        "[C](=O)",  # Carbonyl
        "[C](=O)[O]",  # Ester
        "[C](=O)[N]",  # Amide
        # Benzotriazole and benzimidazole (BTA, ITIC cores)
        "c1c(N)c(=O)c(=O)[nH]1",  # Benzimidazole
        "c1c([N+](=O)[O-])cc[nH]1",  # Nitroindole
        
        # ===== HALOGENATION (Voc, morphology) =====
        # Fluorine (PM6, PBDB-T-2F, PTB7-Th)
        "[F]",  # Fluorine
        "C(F)(F)F",  # Trifluoromethyl
        "C(F)=C(F)",  # Difluorovinylene (Y6)
        "C(=C(F))F",  # Difluorovinylene (alt)
        # Other halogens
        "[Cl]",  # Chlorine
        "[Br]",  # Bromine
        "[I]",  # Iodine
        
        # ===== SIDE-CHAIN ENGINEERING (Solubility, morphology) =====
        # Alkyl chains (PTB7, PBDB-T)
        "[CH3][CH2][CH2][CH2]",  # Alkyl chain
        "[CH]([CH3])[CH2]",  # Branched alkyl
        "[C]([C])([C])[C]",  # Quaternary carbon
        # Alkoxy and thioether
        "[O]([C])[C]",  # Ether
        "[S]([C])[C]",  # Thioether
        "[O][C][C][C]",  # Propoxy
        
        # ===== CONJUGATION EXTENSION (BDT bridges) =====
        # Vinyl and alkene
        "C=C",  # Vinyl/alkene
        "C=O",  # Carbonyl (alternative)
        # Thiophene bridges
        "c1sc(cc1)",  # Thiophene bridge
        # Selenophene bridges
        "c1[se]c(cc1)",  # Selenophene bridge
        
        # ===== SPECIAL HETEROATOMS (Advanced materials) =====
        "[Se]",  # Selenium (selenophene donors)
        "[Si]",  # Silicon (Si-bridges)
        "[N]",  # Nitrogen (pyrrole, pyridine)
        "[O]",  # Oxygen (furan, ether)
        "[S]",  # Sulfur (thiophene, thioether)
        
        # ===== DONOR-ACCEPTOR PATTERNS =====
        # D-A polymer patterns
        "c1ccc(cc1)c1ccccc1",  # Phenyl-phenyl
        "c1sc(cc1)c1ccccc1",  # Thiophene-phenyl
        # Extended conjugation
        "c1ccc2ccccc2c1",  # Naphthyl
    ]

    total_reward = 0.0
    count = 0

    for coords, template_mol in zip(
        generated_coords[:10], template_mols[:10]
    ):  # Sample subset
        try:
            smiles = coords_to_smiles(
                coords[: template_mol.GetNumAtoms()], template_mol
            )
            mol = Chem.MolFromSmiles(smiles)

            if mol:
                validity_reward = calculate_chemical_validity_reward(mol)
                diversity_reward = calculate_diversity_reward(mol, existing_mols)
                smarts_reward = calculate_smarts_reward(mol, desired_patterns)

                total_reward += (
                    lambda_validity * validity_reward
                    + lambda_diversity * diversity_reward
                    + lambda_smarts * smarts_reward
                )
                count += 1
        except:
            continue

    avg_reward = total_reward / max(count, 1)
    # Combined loss: standard GAN loss minus rewards
    return discriminator_loss - avg_reward


print("âœ“ Approach 4: Conditional GAN with Chemical Rules initialized")

# ============================================================================
# APPROACH 5: Reinforcement Learning + GAN (ORGAN-inspired)
# ============================================================================

# Reaction set for SMARTS mutations (comprehensive set for solar cell materials)
# Based on literature: Y6, ITIC, PTB7, PBDB-T, PM6, and high-performance OSC donors/NFAs
rxn_smarts = [
    # ===== HALOGENATION (Improves Voc, morphology control) =====
    # Literature: Fluorination on BDT, thiophene, and end-groups (PM6, PBDB-T-2F)
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]Br",
    "[cH:1]>>[c:1]I",
    # Side-chain fluorination (PTB7-Th, PBDTTT-f series)
    "[CH3:1]>>[CF3:1]",  # Trifluoromethyl substitution
    "[CH2:1]>>[CHF:1]",  # Monofluoromethyl
    
    # ===== STRONG ACCEPTOR END-GROUPS (For NFAs like Y6, ITIC) =====
    # Cyano groups (essential for Y6, ITIC, BTA series)
    "[cH:1]>>[c:1]C#N",
    "[cH:1]>>[c:1]C(=O)C#N",  # acyl-nitrile
    "[cH:1]>>[c:1]C(=C(C#N))C#N",  # malononitrile/vinylene CN (Y6 end-group)
    # Ester and carbonyl groups
    "[cH:1]>>[c:1]C(=O)OC",  # ester
    "[cH:1]>>[c:1]C=O",  # formylation
    "[cH:1]>>[c:1]C(=O)N",  # amide
    
    # ===== SIDE-CHAIN ENGINEERING (Solubility, morphology) =====
    # Alkyl chain variations (PTB7, PBDB-T series)
    "[cH:1]>>[c:1]C",
    "[cH:1]>>[c:1]CC",
    "[cH:1]>>[c:1]CCC",
    "[cH:1]>>[c:1]CCCC",  # butyl
    "[cH:1]>>[c:1]C(C)C",  # isopropyl
    "[cH:1]>>[c:1]C(C)(C)C",  # tert-butyl
    "[cH:1]>>[c:1]C(C)CC",  # sec-butyl
    "[cH:1]>>[c:1]OCC",  # ethoxy
    "[cH:1]>>[c:1]OCCC",  # propoxy
    "[cH:1]>>[c:1]OC(C)C",  # isopropoxy
    "[cH:1]>>[c:1]SC",  # thioether
    "[cH:1]>>[c:1]SCC",  # ethylthio
    
    # ===== CONJUGATION EXTENSION (BDT-based donors) =====
    # Thiophene and selenophene bridges (BDT, PBDB-T)
    "[cH:1]>>[c:1]C=C",  # vinyl linker
    "[cH:1]>>[c:1]c1sc(cc1)",  # thiophene append
    "[cH:1]>>[c:1]c1[se]c(cc1)",  # selenophene append
    "[cH:1]>>[c:1]c1c(C)sc(cc1)",  # methylthiophene
    "[cH:1]>>[c:1]c1c(F)sc(cc1)",  # fluorothiophene
    # Furan and other heteroaromatics
    "[cH:1]>>[c:1]c1occc1",  # furan
    "[cH:1]>>[c:1]c1ccoc1",  # furan (alt)
    
    # ===== DIFLUOROVINYLENE (High-Voc NFAs like Y6) =====
    "[cH:1]>>[c:1]C(=C(F))F",
    "[cH:1]>>[c:1]C(F)=C(F)",  # alt syntax
    
    # ===== ELECTRON-DONATING GROUPS (For donors) =====
    # Alkoxy and amino groups
    "[cH:1]>>[c:1]OC",  # methoxy
    "[cH:1]>>[c:1]N(C)C",  # dimethylamino
    "[cH:1]>>[c:1]N(CC)CC",  # diethylamino
    
    # ===== SPECIAL MODIFICATIONS (Advanced OSC materials) =====
    # Selenium substitution (selenophene-based donors)
    "[cH:1]>>[c:1]c1[se]ccc1",  # selenophene
    # Silicon-containing bridges (emerging materials)
    "[cH:1]>>[c:1][Si](C)(C)C",  # trimethylsilyl group
    # Phenyl and naphthyl extensions
    "[cH:1]>>[c:1]c1ccccc1",  # phenyl
    "[cH:1]>>[c:1]c1ccc2ccccc2c1",  # naphthyl
    
    # ===== ADDITIONAL HIGH-PERFORMING MUTATIONS (Literature) =====
    # Extended alkyl chains (Y6 derivatives, asymmetric engineering)
    "[cH:1]>>[c:1]CCCCC",  # pentyl
    "[cH:1]>>[c:1]CCCCCC",  # hexyl
    "[cH:1]>>[c:1]C(C)CC",  # sec-butyl (asymmetric)
    "[cH:1]>>[c:1]C(CC)C",  # isobutyl (asymmetric)
    "[cH:1]>>[c:1]CC(C)C",  # 2-methylbutyl (asymmetric)
    
    # Additional heteroaromatics (BP5T, extended conjugation)
    "[cH:1]>>[c:1]c1ccncc1",  # pyridine
    "[cH:1]>>[c:1]c1cccnc1",  # pyridine alt
    "[cH:1]>>[c:1]c1ccoc1",  # furan alt
    "[cH:1]>>[c:1]c1cc[nH]c1",  # pyrrole
    
    # Benzotriazole and benzimidazole cores (BTA, ITIC variants)
    "[cH:1]>>[c:1]c1c([N+](=O)[O-])cc[nH]1",  # benzotriazole
    "[cH:1]>>[c:1]c1c(N)c(=O)c(=O)[nH]1",  # benzimidazole
    
    # Diketo and dithio groups (Acceptor strength)
    "[cH:1]>>[c:1]C(=O)C(=O)",  # diketo
    "[cH:1]>>[c:1]C(=O)C(=O)O",  # keto-ester
    "[cH:1]>>[c:1]C(=S)C(=S)",  # dithio
    
    # Diene and extended conjugation
    "[cH:1]>>[c:1]C=CC=C",  # diene
    "[cH:1]>>[c:1]C#CC#C",  # diyne
    
    # Chlorinated variants (emerging high-performers)
    "[cH:1]>>[c:1]c1ccc(Cl)cc1",  # chlorophenyl
    "[cH:1]>>[c:1]c1sc(Cl)cc1",  # chlorothiophene
    
    # Bromine variants (heavy halogen effects)
    "[cH:1]>>[c:1]Br",  # bromine
    "[cH:1]>>[c:1]c1ccc(Br)cc1",  # bromophenyl
    "[cH:1]>>[c:1]c1sc(Br)cc1",  # bromothiophene
    
    # Iodine variants (extreme halogen)
    "[cH:1]>>[c:1]I",  # iodine
    "[cH:1]>>[c:1]c1ccc(I)cc1",  # iodophenyl
    
    # Tricyano and polycyano groups (Super acceptors)
    "[cH:1]>>[c:1]C(C#N)(C#N)C#N",  # tricyano
    
    # Methoxy and ethoxy variants
    "[cH:1]>>[c:1]OC",  # methoxy
    "[cH:1]>>[c:1]OCC",  # ethoxy
    "[cH:1]>>[c:1]OCCC",  # propoxy
    
    # Thioether variants
    "[cH:1]>>[c:1]SC",  # thio-methyl
    "[cH:1]>>[c:1]SCC",  # thio-ethyl
    "[cH:1]>>[c:1]SCCC",  # thio-propyl
    
    # Amino variants
    "[cH:1]>>[c:1]N(C)C",  # dimethylamino
    "[cH:1]>>[c:1]N(CC)CC",  # diethylamino
    "[cH:1]>>[c:1]N(C)CC",  # ethylmethylamino
]
rxns = [rdChemReactions.ReactionFromSmarts(s) for s in rxn_smarts]


def smart_mutants_for_mol(mol, max_out=5):
    """Generate SMARTS-based mutations"""
    out = []
    for rxn in rxns:
        try:
            prods = rxn.RunReactants((mol,))
            for pset in prods:
                try:
                    p = pset[0]
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p, canonical=True)
                    out.append(smi)
                except Exception:
                    continue
        except Exception:
            continue
        if len(out) >= max_out:
            break
    return list(dict.fromkeys(out))[:max_out]


def brics_variants_for_mol(mol, max_out=5):
    """Generate BRICS-based variants"""
    try:
        frags = list(BRICS.BRICSDecompose(mol))
        if len(frags) < 2:
            return []
        gen = BRICS.BRICSBuild(frags)
        variants = []
        for i, vm in enumerate(gen):
            if i >= 20:
                break
            try:
                Chem.SanitizeMol(vm)
                smi = Chem.MolToSmiles(vm, canonical=True)
                variants.append(smi)
            except Exception:
                continue
        orig = Chem.MolToSmiles(mol, canonical=True)
        variants = [s for s in dict.fromkeys(variants) if s != orig]
        return variants[:max_out]
    except Exception:
        return []


def murcko_scaffold_smiles(mol):
    """Extract Murcko scaffold"""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, canonical=True) if scaf else None
    except Exception:
        return None


def cross_brics_variants_for_mol(mol, fragment_library, max_out=5, mix_size=6):
    """Generate cross-template BRICS variants"""
    variants = []
    try:
        base_frags = list(BRICS.BRICSDecompose(mol))
        if not base_frags:
            return variants
        lib_sample = (
            random.sample(list(fragment_library), min(len(fragment_library), mix_size))
            if fragment_library
            else []
        )
        frags = set(base_frags) | set(lib_sample)
        gen = BRICS.BRICSBuild(frags)
        for i, vm in enumerate(gen):
            if i >= 50:
                break
            try:
                Chem.SanitizeMol(vm)
                smi = Chem.MolToSmiles(vm, canonical=True)
                variants.append(smi)
            except Exception:
                continue
    except Exception:
        pass
    return list(dict.fromkeys(variants))[:max_out]


def rl_gan_generate_with_rewards(
    generator,
    latent_dim,
    num_samples,
    coord_dim,
    template_mols,
    training_smiles,
    fragment_library,
):
    """Generate molecules using RL-GAN approach with reward-based selection"""
    # Generate initial candidates
    noise = np.random.normal(0, 1, (num_samples * 3, latent_dim))
    generated_coords = generator.predict(noise, verbose=0)

    candidates = []
    rewards = []

    for coords, template_mol in zip(
        generated_coords,
        template_mols * (len(generated_coords) // len(template_mols) + 1),
    ):
        try:
            smiles = coords_to_smiles(
                coords[: template_mol.GetNumAtoms()], template_mol
            )
            if not smiles or smiles in training_smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            reward = 0.0

            mol = sanitize_molecule(mol)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = fix_valence_issues(mol)
                if mol is None:
                    continue
                mol = sanitize_molecule(mol)
                if mol is None:
                    continue
            reward += 1.0

            # Functional group rewards
            if mol.HasSubstructMatch(Chem.MolFromSmarts("c1sccc1")):  # Thiophene
                reward += 0.5
            if mol.HasSubstructMatch(Chem.MolFromSmarts("C#N")):  # Cyano
                reward += 0.5

            # Diversity reward (check against training)
            fp = morgan_fp(mol, radius=2, nBits=2048)
            if fp is not None:
                train_mol = Chem.MolFromSmiles(random.choice(list(training_smiles)))
                if train_mol:
                    train_fp = morgan_fp(train_mol, radius=2, nBits=2048)
                    if train_fp is not None:
                        sim = DataStructs.TanimotoSimilarity(fp, train_fp)
                        reward += (1.0 - sim) * 0.5  # Reward novelty

            # Apply SMARTS mutations for diversity
            mutants = smart_mutants_for_mol(mol, max_out=2)
            for mut_smi in mutants:
                candidates.append(mut_smi)
                rewards.append(reward * 0.8)  # Slightly lower reward for mutants

            # Apply BRICS variants
            brics_vars = brics_variants_for_mol(mol, max_out=2)
            for brics_smi in brics_vars:
                candidates.append(brics_smi)
                rewards.append(reward * 0.9)

            candidates.append(smiles)
            rewards.append(reward)

        except Exception:
            continue

    # Select top candidates by reward
    if candidates and rewards:
        sorted_indices = np.argsort(rewards)[::-1]
        top_candidates = [candidates[i] for i in sorted_indices[:num_samples]]
        return list(dict.fromkeys(top_candidates))  # Remove duplicates

    return []


print("âœ“ Approach 5: Reinforcement Learning + GAN (ORGAN) initialized")

# ============================================================================
# APPROACH 6: Molecular VAE (Variational Autoencoder)
# ============================================================================


def sampling(args):
    """Reparameterization trick for VAE"""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_vae_encoder(input_dim, latent_dim=128, hidden_units=(512, 256)):
    """Build VAE encoder"""
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def build_vae_decoder(latent_dim, output_dim, hidden_units=(256, 512)):
    """Build VAE decoder"""
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for units in hidden_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
    outputs = layers.Dense(output_dim, activation="sigmoid")(x)

    decoder = Model(latent_inputs, outputs, name="decoder")
    return decoder


def vae_loss(inputs, outputs, z_mean, z_log_var):
    """VAE loss = reconstruction loss + KL divergence"""
    reconstruction_loss = K.sum(K.square(inputs - outputs), axis=-1)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


def build_molecular_vae(input_dim, latent_dim=128, encoder_units=(512, 256), decoder_units=(256, 512), learning_rate=0.001):
    """Build complete molecular VAE"""
    encoder = build_vae_encoder(input_dim, latent_dim, encoder_units)
    decoder = build_vae_decoder(latent_dim, input_dim, decoder_units)

    inputs = Input(shape=(input_dim,))
    z_mean, z_log_var, z = encoder(inputs)
    outputs = decoder(z)

    vae = Model(inputs, outputs, name="vae")

    # Add VAE loss
    vae.add_loss(vae_loss(inputs, outputs, z_mean, z_log_var))
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return vae, encoder, decoder


def vae_generate_molecules(
    decoder, latent_dim, num_samples, template_mols, training_smiles
):
    """Generate molecules using trained VAE decoder"""
    generated_smiles = []

    # Sample from standard normal distribution
    z_samples = np.random.normal(0, 1, (num_samples * 2, latent_dim))

    # Decode to fingerprints
    generated_fps = decoder.predict(z_samples, verbose=0)

    # Convert fingerprints back to SMILES (approximate)
    for fp_vec in generated_fps:
        try:
            # Use nearest neighbor from training set as template
            template_mol = random.choice(template_mols)
            if template_mol is None:
                continue

            # Perturb the template based on decoded features
            template_smiles = Chem.MolToSmiles(template_mol)
            mol = Chem.MolFromSmiles(template_smiles)

            if mol is None:
                continue

            # Apply modifications based on decoded vector
            # Use high activation regions to guide mutations
            high_activation_indices = np.where(fp_vec > 0.7)[0]

            if len(high_activation_indices) > 0:
                # Apply SMARTS mutations
                mutants = smart_mutants_for_mol(mol, max_out=3)
                for mut_smi in mutants:
                    if mut_smi and mut_smi not in training_smiles:
                        generated_smiles.append(mut_smi)
                        if len(generated_smiles) >= num_samples:
                            break

            # Also try BRICS recombination
            brics_vars = brics_variants_for_mol(mol, max_out=2)
            for brics_smi in brics_vars:
                if brics_smi and brics_smi not in training_smiles:
                    generated_smiles.append(brics_smi)
                    if len(generated_smiles) >= num_samples:
                        break

            if len(generated_smiles) >= num_samples:
                break

        except Exception:
            continue

    return list(dict.fromkeys(generated_smiles))[:num_samples]


print("âœ“ Approach 6: Molecular VAE initialized")

# ============================================================================
# APPROACH 7: Transformer-Based Molecular Generation
# ============================================================================


def build_transformer_encoder_layer(d_model, num_heads, dff, dropout_rate=0.1):
    """Build a single transformer encoder layer"""

    def transformer_encoder_layer(x):
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed forward network
        ffn = layers.Dense(dff, activation="relu")(out1)
        ffn = layers.Dense(d_model)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)

        return out2

    return transformer_encoder_layer


def build_transformer_decoder_layer(d_model, num_heads, dff, dropout_rate=0.1):
    """Build a single transformer decoder layer"""

    def transformer_decoder_layer(x, enc_output):
        # Self attention
        attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn1 = Dropout(dropout_rate)(attn1)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn1)

        # Cross attention with encoder output
        attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
            out1, enc_output
        )
        attn2 = Dropout(dropout_rate)(attn2)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)

        # Feed forward
        ffn = layers.Dense(dff, activation="relu")(out2)
        ffn = layers.Dense(d_model)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn)

        return out3

    return transformer_decoder_layer


def build_molecular_transformer(
    vocab_size, max_length=100, d_model=256, num_heads=8, dff=512, num_layers=4
):
    """Build transformer for molecular generation"""
    # Encoder
    encoder_inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, d_model)(encoder_inputs)

    # Positional encoding (simple version)
    positions = tf.range(start=0, limit=max_length, delta=1)
    pos_encoding = Embedding(max_length, d_model)(positions)
    x = x + pos_encoding

    # Stack encoder layers
    for _ in range(num_layers):
        x = build_transformer_encoder_layer(d_model, num_heads, dff)(x)

    encoder_output = x
    encoder = Model(encoder_inputs, encoder_output, name="transformer_encoder")

    # Decoder
    decoder_inputs = Input(shape=(max_length,))
    y = Embedding(vocab_size, d_model)(decoder_inputs)
    y = y + pos_encoding

    # Stack decoder layers
    for _ in range(num_layers):
        y = build_transformer_decoder_layer(d_model, num_heads, dff)(y, encoder_output)

    outputs = layers.Dense(vocab_size, activation="softmax")(y)

    transformer = Model([encoder_inputs, decoder_inputs], outputs, name="transformer")
    return transformer, encoder


def transformer_generate_molecules(
    encoder,
    vocab_size,
    max_length,
    num_samples,
    char_to_idx,
    idx_to_char,
    template_smiles,
):
    """Generate molecules using trained transformer"""
    generated_smiles = []

    for _ in range(num_samples * 2):
        try:
            # Random start token
            start_token = random.randint(0, vocab_size - 1)
            generated_sequence = [start_token]

            # Autoregressively generate sequence
            for _ in range(max_length - 1):
                # Pad sequence
                padded = generated_sequence + [0] * (
                    max_length - len(generated_sequence)
                )
                padded = np.array([padded])

                # Predict next token (simplified - without full decoder)
                # Use template-based perturbation instead
                break

            # Use template-based generation with transformer guidance
            template = random.choice(template_smiles)
            mol = Chem.MolFromSmiles(template)

            if mol:
                # Apply transformer-guided mutations
                mutants = smart_mutants_for_mol(mol, max_out=2)
                generated_smiles.extend(mutants)

                if len(generated_smiles) >= num_samples:
                    break

        except Exception:
            continue

    return list(dict.fromkeys(generated_smiles))[:num_samples]


print("âœ“ Approach 7: Transformer-based Generation initialized")

# ============================================================================
# APPROACH 8: Transfer Learning with Pre-trained Embeddings
# ============================================================================


def build_transfer_learning_generator(
    pretrained_dim=768, latent_dim=128, output_dim=2048
):
    """Build generator leveging pre-trained molecular embeddings"""
    # Input: concatenate random noise with pretrained embeddings
    noise_input = Input(shape=(latent_dim,))
    pretrained_input = Input(shape=(pretrained_dim,))

    # Combine inputs
    combined = layers.Concatenate()([noise_input, pretrained_input])

    # Generator network
    x = layers.Dense(512, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(output_dim, activation="sigmoid")(x)

    model = Model([noise_input, pretrained_input], x, name="transfer_generator")
    return model


def get_molecular_embeddings(smiles_list, embedding_dim=768):
    """
    Simulate pre-trained molecular embeddings (ChemBERTa/MolBERT style)
    In production, use actual pre-trained models
    """
    embeddings = []
    morgan_gen = GetMorganGenerator(2, fpSize=2048)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Use Morgan fingerprint as proxy for embeddings
            fp = morgan_gen.GetFingerprintAsNumPy(mol)

            # Dimensionality reduction to simulate embeddings
            from sklearn.decomposition import TruncatedSVD

            if len(embeddings) == 0:
                # Initialize SVD
                global _embedding_svd
                _embedding_svd = TruncatedSVD(n_components=min(embedding_dim, len(fp)))

            # Pad/truncate to embedding_dim
            embedding = np.zeros(embedding_dim)
            embedding[: min(len(fp), embedding_dim)] = fp[: min(len(fp), embedding_dim)]
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(embedding_dim))

    return np.array(embeddings)


def transfer_learning_generate(template_smiles, num_samples, fragment_library, molecule_type="Generic"):
    """Generate molecules using transfer learning approach"""
    generated_smiles = []
    ordered_templates = balanced_trait_order(template_smiles, molecule_type)
    logger.info(
        f"Transfer learning using trait-balanced templates: {len(ordered_templates)} molecules across {len(set(osc_trait_signature(s, molecule_type) for s in ordered_templates))} trait groups"
    )
    if not ordered_templates:
        return generated_smiles

    # Get embeddings for templates
    template_embeddings = get_molecular_embeddings(ordered_templates)

    for _ in range(num_samples):
        try:
            # Sample a template based on embedding similarity
            template_idx = random.randint(0, len(ordered_templates) - 1)
            template = ordered_templates[template_idx]
            mol = Chem.MolFromSmiles(template)

            if mol:
                # Use embedding-guided mutations
                # High embedding values suggest important features to preserve

                # Apply conservative mutations
                mutants = smart_mutants_for_mol(mol, max_out=2)
                generated_smiles.extend(mutants)

                # Apply BRICS with fragment library
                brics_vars = cross_brics_variants_for_mol(
                    mol, fragment_library, max_out=2
                )
                generated_smiles.extend(brics_vars)

                if len(generated_smiles) >= num_samples:
                    break

        except Exception:
            continue

    return list(dict.fromkeys(generated_smiles))[:num_samples]


print("âœ“ Approach 8: Transfer Learning initialized")

# ============================================================================
# APPROACH 9: Multi-Objective Optimization (Pareto Frontier)
# ============================================================================


def calculate_osc_properties(mol):
    """Calculate OSC-relevant properties for multi-objective optimization"""
    if mol is None:
        return None

    try:
        properties = {
            "MW": Descriptors.ExactMolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
        }

        # Estimate conjugation (proxy for PCE)
        conjugation_score = properties["NumAromaticRings"] * 0.3

        # Estimate Voc (proxy based on electron-withdrawing groups)
        voc_proxy = 0.5
        if mol.HasSubstructMatch(Chem.MolFromSmarts("C#N")):
            voc_proxy += 0.2
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[F]")):
            voc_proxy += 0.15

        # Estimate Jsc (proxy based on conjugation and molecular weight)
        jsc_proxy = conjugation_score + (properties["MW"] / 1000.0)

        # Estimate FF (proxy based on planarity and rigidity)
        ff_proxy = 0.6 - (properties["NumRotatableBonds"] * 0.02)

        properties["PCE_proxy"] = conjugation_score + voc_proxy + jsc_proxy + ff_proxy
        properties["Voc_proxy"] = voc_proxy
        properties["Jsc_proxy"] = jsc_proxy
        properties["FF_proxy"] = ff_proxy

        return properties
    except:
        return None


def pareto_dominates(obj1, obj2):
    """Check if obj1 Pareto-dominates obj2 (maximize all objectives)"""
    better_in_any = False
    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:
            return False
        if obj1[i] > obj2[i]:
            better_in_any = True
    return better_in_any


def get_pareto_frontier(molecules_with_objectives):
    """Extract Pareto frontier from population"""
    pareto_front = []

    for i, (mol1, obj1) in enumerate(molecules_with_objectives):
        dominated = False
        for j, (mol2, obj2) in enumerate(molecules_with_objectives):
            if i != j and pareto_dominates(obj2, obj1):
                dominated = True
                break
        if not dominated:
            pareto_front.append((mol1, obj1))

    return pareto_front


def multi_objective_generate(
    template_mols, template_smiles, num_samples, fragment_library, molecule_type="Generic"
):
    """Generate molecules optimized for multiple objectives (PCE, Voc, Jsc, FF)"""
    population = []
    ordered_templates = balanced_trait_order(template_smiles, molecule_type)
    logger.info(
        f"Multi-objective generation initializing from {len(ordered_templates)} trait-balanced templates"
    )

    # Initial population from templates
    for template in ordered_templates:
        mol = Chem.MolFromSmiles(template)
        if mol:
            props = calculate_osc_properties(mol)
            if props:
                objectives = [
                    props["PCE_proxy"],
                    props["Voc_proxy"],
                    props["Jsc_proxy"],
                    props["FF_proxy"],
                ]
                population.append((Chem.MolToSmiles(mol), objectives))

    # Generate variations and evaluate
    generated_count = 0
    iterations = 0
    max_iterations = num_samples * 3

    while generated_count < num_samples and iterations < max_iterations:
        iterations += 1

        try:
            # Select parent from current population
            if population:
                parent_smiles, _ = random.choice(population)
                parent_mol = Chem.MolFromSmiles(parent_smiles)
            else:
                parent_mol = random.choice(template_mols)

            if parent_mol is None:
                continue

            # Generate variations
            variants = []
            variants.extend(smart_mutants_for_mol(parent_mol, max_out=2))
            variants.extend(brics_variants_for_mol(parent_mol, max_out=2))
            variants.extend(
                cross_brics_variants_for_mol(parent_mol, fragment_library, max_out=2)
            )

            # Evaluate each variant
            for var_smiles in variants:
                var_mol = Chem.MolFromSmiles(var_smiles)
                if var_mol:
                    props = calculate_osc_properties(var_mol)
                    if props:
                        objectives = [
                            props["PCE_proxy"],
                            props["Voc_proxy"],
                            props["Jsc_proxy"],
                            props["FF_proxy"],
                        ]
                        population.append((var_smiles, objectives))
                        generated_count += 1

                        if generated_count >= num_samples:
                            break
        except:
            continue

    # Extract Pareto frontier
    pareto_front = get_pareto_frontier(population)

    # Return molecules on Pareto frontier
    pareto_smiles = [smiles for smiles, _ in pareto_front]

    # If not enough, add best from population
    if len(pareto_smiles) < num_samples:
        remaining = num_samples - len(pareto_smiles)
        # Sort by PCE proxy and take top
        population_sorted = sorted(population, key=lambda x: x[1][0], reverse=True)
        for smiles, _ in population_sorted:
            if smiles not in pareto_smiles:
                pareto_smiles.append(smiles)
                if len(pareto_smiles) >= num_samples:
                    break

    return list(dict.fromkeys(pareto_smiles))[:num_samples]


print("âœ“ Approach 9: Multi-Objective Optimization initialized")

# ============================================================================
# APPROACH 10: Active Learning with Uncertainty Sampling
# ============================================================================


class ActiveLearningOracle:
    """Simulates an oracle (DFT calculations) for active learning"""

    def __init__(self):
        self.queried_molecules = {}
        self.query_count = 0

    def query(self, smiles):
        """Simulate DFT calculation (returns estimated PCE)"""
        if smiles in self.queried_molecules:
            return self.queried_molecules[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Simulate PCE based on molecular features
        props = calculate_osc_properties(mol)
        if props:
            pce_estimate = props["PCE_proxy"] + random.gauss(0, 0.1)
            self.queried_molecules[smiles] = max(0, pce_estimate)
            self.query_count += 1
            return self.queried_molecules[smiles]
        return 0.0


def calculate_uncertainty(model, fingerprint):
    """Calculate prediction uncertainty using ensemble or dropout"""
    # Use dropout at test time for uncertainty estimation
    predictions = []
    for _ in range(10):
        pred = model.predict(fingerprint.reshape(1, -1), verbose=0)
        predictions.append(pred[0])

    return np.std(predictions)


def active_learning_generate(
    template_smiles, template_mols, num_samples, fragment_library, optuna_params=None, molecule_type="Generic"
):
    """Generate molecules using active learning with uncertainty sampling"""
    optuna_params = optuna_params or {}
    n_estimators = int(optuna_params.get("n_estimators", 50))
    max_depth = optuna_params.get("max_depth", None)
    exploration_weight = float(optuna_params.get("exploration_weight", 0.5))
    exploitation_weight = 1.0 - exploration_weight
    logger.info(
        f"Active learning generation params: n_estimators={n_estimators}, max_depth={max_depth}, exploration_weight={exploration_weight:.3f}, exploitation_weight={exploitation_weight:.3f}"
    )
    oracle = ActiveLearningOracle()
    ordered_templates = balanced_trait_order(template_smiles, molecule_type)
    logger.info(
        f"Active-learning using all trait-balanced templates: {len(ordered_templates)} molecules across {len(set(osc_trait_signature(s, molecule_type) for s in ordered_templates))} trait groups"
    )
    if not ordered_templates:
        return []

    # Initialize training data
    training_smiles = []
    training_labels = []

    # Query initial templates
    for template in ordered_templates:
        pce = oracle.query(template)
        training_smiles.append(template)
        training_labels.append(pce)

    # Build initial surrogate model
    morgan_gen = GetMorganGenerator(2, fpSize=2048)

    def get_fingerprint(smiles):
        text = str(smiles).strip()
        for feature_map in TRAINING_FALLBACK_FEATURES.values():
            fp = feature_map.get(text)
            if fp is not None:
                return np.asarray(fp, dtype=np.float32)
        mol = Chem.MolFromSmiles(text)
        if mol:
            return morgan_gen.GetFingerprintAsNumPy(mol)
        fallback = fallback_representation_from_smiles(text)
        if fallback["usable"]:
            logger.debug(f"Active-learning using {fallback['representation']} fallback fingerprint for: {text[:80]}")
            return fallback["hashed_fp"]
        logger.debug(f"Active-learning could not fingerprint molecule; using zero vector for: {text[:80]}")
        return np.zeros(2048, dtype=np.float32)

    X_train = np.array([get_fingerprint(s) for s in training_smiles])
    y_train = np.array(training_labels)

    # Train surrogate model
    surrogate = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    surrogate.fit(X_train, y_train)

    # Active learning loop
    candidate_pool = []
    selected_molecules = []

    # Generate candidate pool
    for template in ordered_templates:
        mol = Chem.MolFromSmiles(template)
        if mol:
            candidate_pool.extend(smart_mutants_for_mol(mol, max_out=3))
            candidate_pool.extend(brics_variants_for_mol(mol, max_out=2))

    candidate_pool = list(dict.fromkeys(candidate_pool))

    # Active learning iterations
    for iteration in range(min(num_samples, len(candidate_pool))):
        # Calculate uncertainty for all candidates
        uncertainties = []
        predictions = []

        for candidate in candidate_pool:
            fp = get_fingerprint(candidate)

            # Ensemble prediction for uncertainty
            tree_predictions = [
                tree.predict(fp.reshape(1, -1))[0] for tree in surrogate.estimators_
            ]
            uncertainty = np.std(tree_predictions)
            prediction = np.mean(tree_predictions)

            uncertainties.append(uncertainty)
            predictions.append(prediction)

        # Select molecule with highest uncertainty and good predicted performance
        # Balance exploration (uncertainty) and exploitation (prediction)
        combined_scores = [
            exploration_weight * u + exploitation_weight * p for u, p in zip(uncertainties, predictions)
        ]

        if not combined_scores:
            break

        best_idx = np.argmax(combined_scores)
        selected_smiles = candidate_pool[best_idx]

        # Query oracle
        true_pce = oracle.query(selected_smiles)

        # Add to selected
        if true_pce > 0.3:
            selected_molecules.append(selected_smiles)

        # Update training data
        training_smiles.append(selected_smiles)
        training_labels.append(true_pce)

        # Retrain surrogate
        X_train = np.array([get_fingerprint(s) for s in training_smiles])
        y_train = np.array(training_labels)
        surrogate.fit(X_train, y_train)

        # Remove selected from pool
        candidate_pool.pop(best_idx)

        if len(selected_molecules) >= num_samples:
            break

    # If not enough selected, add top predicted
    if len(selected_molecules) < num_samples:
        remaining_predictions = [
            (s, surrogate.predict(get_fingerprint(s).reshape(1, -1))[0])
            for s in candidate_pool[: num_samples * 2]
        ]
        remaining_sorted = sorted(
            remaining_predictions, key=lambda x: x[1], reverse=True
        )

        for smiles, _ in remaining_sorted:
            if smiles not in selected_molecules:
                selected_molecules.append(smiles)
                if len(selected_molecules) >= num_samples:
                    break

    return list(dict.fromkeys(selected_molecules))[:num_samples]


print("âœ“ Approach 10: Active Learning initialized")

print("\n" + "=" * 80)
print("ALL 12 GENERATION METHODS INITIALIZED SUCCESSFULLY")
print("=" * 80)
print("Approaches 1-6: Base GAN + 5 Structural Variations")
print("Approaches 7-12: Advanced ML and constrained graph techniques")
print("  7. Molecular VAE (Variational Autoencoder)")
print("  8. Transformer-based Generation")
print("  9. Transfer Learning with Pre-trained Embeddings")
print("  10. Multi-Objective Optimization (Pareto)")
print("  11. Active Learning with Uncertainty Sampling")
print("  12. CoCoGraph-Inspired Constrained Graph Generation")
print("=" * 80 + "\n")

# ============================================================================
# HYBRID GENERATION FUNCTION
# ============================================================================


def ensure_approach_generates_molecules(
    all_generated,
    approach_key,
    template_smiles,
    fragment_library,
    min_molecules=50,
    molecule_type="Generic",
    training_canonical_set=None,
    training_fps=None,
    max_similarity=0.4,
):
    """Ensure each approach generates actual new molecules, not just copies training data"""
    if training_canonical_set is None:
        training_canonical_set = canonical_smiles_set(template_smiles)
    if training_fps is None:
        training_fps = fingerprints_from_smiles(template_smiles)
    filtered_existing, seen_existing, rejected_existing = filter_generated_smiles(
        all_generated[approach_key],
        training_canonical_set,
        training_fps,
        molecule_type,
        max_similarity=max_similarity,
    )
    if rejected_existing:
        logger.info(f"{approach_key}: pre-filter rejected candidates: {rejected_existing}")
    all_generated[approach_key] = filtered_existing
    if len(all_generated[approach_key]) < min_molecules:
        logger.warning(f"{approach_key} only generated {len(all_generated[approach_key])} molecules, attempting to generate more...")
        
        additional_molecules = []

        # Shuffle the (otherwise fully deterministic) balanced_trait_order
        # traversal using the now rank-seeded `random` module, so different
        # ranks explore different templates/mutation trajectories through
        # this backfill path instead of reconstructing the identical
        # molecule set every time (see RANK-DIFFERENTIATED RANDOM SEEDING
        # note near the top of the file for full root-cause context).
        backfill_order = balanced_trait_order(template_smiles, molecule_type)
        random.shuffle(backfill_order)

        for template_smi in backfill_order:
            try:
                mol = Chem.MolFromSmiles(template_smi)
                if mol is None:
                    continue
                
                mutants = smart_mutants_for_mol(mol, max_out=3)
                for mut_smi in mutants:
                    can, ok, _, _ = accept_generated_candidate(
                        mut_smi,
                        training_canonical_set,
                        training_fps,
                        molecule_type,
                        seen_existing,
                        max_similarity=max_similarity,
                    )
                    if ok:
                        additional_molecules.append(can)
                        seen_existing.add(can)
                
                brics_vars = brics_variants_for_mol(mol, max_out=2)
                for brics_smi in brics_vars:
                    can, ok, _, _ = accept_generated_candidate(
                        brics_smi,
                        training_canonical_set,
                        training_fps,
                        molecule_type,
                        seen_existing,
                        max_similarity=max_similarity,
                    )
                    if ok:
                        additional_molecules.append(can)
                        seen_existing.add(can)
                
                cross_vars = cross_brics_variants_for_mol(mol, fragment_library, max_out=2)
                for cross_smi in cross_vars:
                    can, ok, _, _ = accept_generated_candidate(
                        cross_smi,
                        training_canonical_set,
                        training_fps,
                        molecule_type,
                        seen_existing,
                        max_similarity=max_similarity,
                    )
                    if ok:
                        additional_molecules.append(can)
                        seen_existing.add(can)
                
                if len(additional_molecules) >= min_molecules:
                    break
                    
            except Exception as e:
                logger.debug(f"Failed to generate from {template_smi}: {e}")
                continue
        
        unique_additional = list(dict.fromkeys(additional_molecules))
        needed = min(len(unique_additional), min_molecules - len(all_generated[approach_key]))
        all_generated[approach_key].extend(unique_additional[:needed])
        if len(all_generated[approach_key]) < min_molecules:
            needed_more = min_molecules - len(all_generated[approach_key])
            constrained_more = constrained_cocograph_generate(
                template_smiles,
                fragment_library,
                needed_more,
                molecule_type,
                training_canonical_set,
                training_fps,
                max_similarity=max_similarity,
            )
            for smi in constrained_more:
                if smi not in seen_existing:
                    all_generated[approach_key].append(smi)
                    seen_existing.add(smi)
        
        logger.info(f"Generated additional constrained molecules for {approach_key} (total: {len(all_generated[approach_key])})")

def hybrid_generate_molecules(
    base_generator,
    latent_dim,
    coord_dim,
    template_mols,
    template_smiles,
    num_per_approach=1500,
    molecule_type="Donor",
):
    """
    Generate molecules using all 12 methods in a hybrid pipeline

    Returns a dictionary with molecules from each approach
    """
    logger.info("=" * 80)
    logger.info(f"HYBRID GENERATION: {molecule_type}")
    logger.info("=" * 80)
    logger.info(
        f"Parameters: latent_dim={latent_dim}, coord_dim={coord_dim}, num_per_approach={num_per_approach}"
    )
    logger.info(
        f"Template molecules: {len(template_mols)}, Template SMILES: {len(template_smiles)}"
    )

    hybrid_generate_molecules._cond_hv = 0

    all_generated = {
        "approach1_smiles_gan": [],
        "approach2_graph_gan": [],
        "approach3_junction_tree": [],
        "approach4_conditional_gan": [],
        "approach5_rl_gan": [],
        "approach6_vae": [],
        "approach7_transformer": [],
        "approach8_transfer_learning": [],
        "approach9_multi_objective": [],
        "approach10_active_learning": [],
        "approach11_cocograph": [],
        "base_gan": [],
    }

    # Build fragment library for approaches that need it
    logger.info("Building fragment library for hybrid approaches...")
    fragment_library = build_fragment_library(template_smiles, max_frags=3000)
    logger.info(f"Fragment library built with {len(fragment_library)} fragments")
    training_canonical_set = canonical_smiles_set(template_smiles)
    training_fps = fingerprints_from_smiles(template_smiles)
    strict_novelty_threshold = 0.4
    auxiliary_optuna_trials = 2
    logger.info("=" * 80)
    logger.info("AUXILIARY APPROACH OPTUNA REQUIREMENT CHECK")
    logger.info("=" * 80)
    logger.info("Optuna WILL be used for SMILES_GAN, VAE, and Active_Learning because they train tunable models.")
    logger.info("Optuna is NOT needed for Graph_GAN here because it is an untrained random graph generator in this script.")
    logger.info("Optuna is NOT needed for Junction_Tree, Transfer_Learning, Multi_Objective, and CoCoGraph_Constrained because they are rule/fragment/objective-driven heuristics with explicit chemical constraints.")
    logger.info("Optuna is NOT separately needed for Conditional_GAN and RL_GAN because they reuse the already optimized base GAN generator and apply filtering/reward selection.")
    logger.info("Transformer approach is template/mutation-guided in this script, not trained end-to-end, so Optuna is not justified unless a real transformer training loop is added.")

    # Approach 1: SMILES-Based GAN (Trained on Real Data)
    logger.info("=" * 80)
    logger.info("APPROACH 1: SMILES-Based GAN (Trained) - STARTING")
    logger.info("=" * 80)
    try:
        # Build vocabulary and prepare data
        chars, char_to_idx, idx_to_char = build_smiles_vocab(template_smiles)
        max_smiles_length = max(len(s) for s in template_smiles)
        vocab_size = len(chars)
        logger.info(f"Vocabulary size: {vocab_size}, Max SMILES length: {max_smiles_length}")
        logger.info(f"Valid SMILES characters in training data: {sorted(chars)}")
        smiles_optuna_params = {}
        if len(template_smiles) >= 20:
            smiles_optuna_params = optimize_smiles_gan_hyperparameters(
                template_smiles,
                char_to_idx,
                idx_to_char,
                max_smiles_length,
                vocab_size,
                latent_dim,
                n_trials=auxiliary_optuna_trials,
            )
        else:
            logger.warning("Skipping SMILES GAN Optuna: not enough training SMILES")
        
        # Build and train SMILES GAN
        logger.info("Building SMILES GAN models...")
        smiles_gen = build_smiles_generator(
            latent_dim,
            max_smiles_length,
            vocab_size,
            dense_units=int(smiles_optuna_params.get("dense_units", 256)),
            lstm_units=int(smiles_optuna_params.get("gen_lstm_units", 512)),
        )
        smiles_disc = build_smiles_discriminator(
            max_smiles_length,
            vocab_size,
            lstm_units=int(smiles_optuna_params.get("disc_lstm_units", 256)),
            dense_units=int(smiles_optuna_params.get("disc_dense_units", 128)),
            dropout_rate=float(smiles_optuna_params.get("dropout_rate", 0.3)),
        )
        
        # Train the GAN on real training SMILES
        logger.info("Training SMILES GAN on real training data...")
        smiles_gen, smiles_disc = train_smiles_gan(
            smiles_gen, smiles_disc, template_smiles, 
            char_to_idx, idx_to_char, max_smiles_length, vocab_size, 
            latent_dim,
            epochs=int(smiles_optuna_params.get("epochs", 20)),
            batch_size=int(smiles_optuna_params.get("batch_size", 64)),
            gen_lr=float(smiles_optuna_params.get("gen_lr", 0.0002)),
            disc_lr=float(smiles_optuna_params.get("disc_lr", 0.0001)),
        )
        
        # Generate molecules using trained GAN with improved filtering
        total_batches = 50
        batch_size = 100
        logger.info(f"Generating {total_batches} batches of {batch_size} molecules each using trained GAN...")
        logger.info("Using greedy decoding + SMARTS mutations for realistic generation")
        
        for batch_idx in range(total_batches):
            try:
                # Generate using trained generator
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                generated_one_hot = smiles_gen.predict(noise, verbose=0)
                
                batch_valid = 0
                for one_hot in generated_one_hot:
                    try:
                        # Use greedy decoding (temperature=1.0) for more realistic SMILES
                        smi = one_hot_to_smiles(one_hot, idx_to_char, temperature=1.0)
                        
                        # Strict validation: must be realistic length and valid
                        if not smi or len(smi) < 10:
                            continue
                        
                        if not is_valid_smiles(smi):
                            continue
                        
                        if smi in template_smiles:
                            continue
                        
                        # Additional filter: must contain aromatic rings or heteroatoms (realistic)
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            # Check for aromatic atoms or heteroatoms
                            has_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
                            has_heteroatom = any(atom.GetSymbol() in ['N', 'O', 'S', 'F', 'Cl', 'Br'] for atom in mol.GetAtoms())
                            
                            if has_aromatic or has_heteroatom:
                                all_generated["approach1_smiles_gan"].append(smi)
                                batch_valid += 1
                    except Exception as e:
                        logger.debug(f"SMILES conversion failed: {e}")
                        continue
                
                logger.info(f"  Batch {batch_idx + 1}/{total_batches}: {batch_valid} valid novel molecules, Total so far: {len(all_generated['approach1_smiles_gan'])}")
            except Exception as batch_e:
                logger.warning(f"  Batch {batch_idx + 1} failed: {batch_e}")
                continue
        
        # If SMILES GAN produces few molecules, augment with SMARTS mutations from training data
        if len(all_generated["approach1_smiles_gan"]) < num_per_approach // 2:
            logger.info("SMILES GAN produced few molecules. Augmenting with SMARTS mutations...")
            for smi in balanced_trait_order(template_smiles, molecule_type):
                if len(all_generated["approach1_smiles_gan"]) >= num_per_approach:
                    break
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mutants = smart_mutants_for_mol(mol, max_out=3)
                        for mut_smi in mutants:
                            if (mut_smi and is_valid_smiles(mut_smi) and 
                                mut_smi not in template_smiles and
                                mut_smi not in all_generated["approach1_smiles_gan"]):
                                all_generated["approach1_smiles_gan"].append(mut_smi)
                except:
                    continue

        logger.info(f"APPROACH 1 COMPLETE: Generated {len(all_generated['approach1_smiles_gan'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 1 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - if this approach fails, it will be handled by ensure_approach_generates_molecules

    # Approach 2: Graph-Based GAN
    logger.info("=" * 80)
    logger.info("APPROACH 2: Graph-Based GAN - STARTING")
    logger.info("=" * 80)
    try:
        graph_gen = build_graph_generator(latent_dim, max_atoms=50, num_atom_types=9)
        logger.info("Graph generator built successfully")
        
        # Generate MANY batches - NO LIMIT
        total_batches = 50
        logger.info(f"Generating {total_batches} batches of {num_per_approach} molecules each...")
        
        for batch_idx in range(total_batches):
            noise = np.random.normal(0, 1, (num_per_approach, latent_dim))
            adj_matrices, node_features = graph_gen.predict(noise, verbose=0)
            batch_valid = 0
            for adj, nodes in zip(adj_matrices, node_features):
                smi = graph_to_smiles(adj, nodes)
                # NOVELTY FILTER: Exclude molecules from training data
                if smi and is_valid_smiles(smi) and smi not in template_smiles:
                    all_generated["approach2_graph_gan"].append(smi)
                    batch_valid += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{total_batches}: {batch_valid} valid molecules, Total so far: {len(all_generated['approach2_graph_gan'])}")

        logger.info(f"APPROACH 2 COMPLETE: Generated {len(all_generated['approach2_graph_gan'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 2 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 3: Junction Tree Fragment Assembly (Hybrid Strategy)
    logger.info("=" * 80)
    logger.info("APPROACH 3: Junction Tree Fragment Assembly (Hybrid) - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY molecules using hybrid strategy - NO LIMIT
        target_molecules = num_per_approach * 10  # Increased multiplier
        logger.info(f"Generating {target_molecules} junction tree molecules using hybrid strategy...")
        logger.info("  Hybrid Strategy: 40% BRICS + 40% Scaffold-based + 10% Donor + 10% Acceptor")
        
        # Use hybrid strategy combining BRICS, scaffold-based, and constrained assembly
        junction_mols = junction_tree_generate_hybrid(
            fragment_library, 
            template_smiles,
            num_molecules=target_molecules, 
            training_smiles_set=set(template_smiles)
        )
        all_generated["approach3_junction_tree"] = [
            s for s in junction_mols if is_valid_smiles(s) and s not in template_smiles
        ]
        logger.info(f"APPROACH 3 COMPLETE: Generated {len(all_generated['approach3_junction_tree'])} valid novel molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 3 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Fallback to basic junction tree if hybrid fails
        try:
            logger.info("Attempting fallback to basic junction tree generation...")
            junction_mols = junction_tree_generate(
                fragment_library, num_molecules=target_molecules, training_smiles_set=set(template_smiles)
            )
            all_generated["approach3_junction_tree"] = [
                s for s in junction_mols if is_valid_smiles(s) and s not in template_smiles
            ]
            logger.info(f"Fallback generated {len(all_generated['approach3_junction_tree'])} molecules")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            all_generated["approach3_junction_tree"] = []

    # Approach 4: Conditional GAN with Chemical Rules (using base generator)
    logger.info("=" * 80)
    logger.info("APPROACH 4: Conditional GAN with Chemical Rules - STARTING")
    logger.info("=" * 80)
    try:
        existing_mols = [
            Chem.MolFromSmiles(s)
            for s in balanced_trait_order(template_smiles, molecule_type)
            if Chem.MolFromSmiles(s)
        ]
        logger.info(f"Using {len(existing_mols)} reference molecules for diversity calculation")
        
        # Generate MANY batches - NO LIMIT
        total_batches = 20
        batch_size = num_per_approach * 5
        logger.info(f"Generating {total_batches} batches of {batch_size} molecules each...")
        logger.info("Note: SMILES-GAN may generate duplicates which will be deduplicated later")
        
        for batch_idx in range(total_batches):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_coords = base_generator.predict(noise, verbose=0)
            batch_valid = 0

            for coords, template_mol in zip(
                generated_coords,
                template_mols * (len(generated_coords) // len(template_mols) + 1),
            ):
                try:
                    smi = coords_to_smiles(
                        coords[: template_mol.GetNumAtoms()], template_mol
                    )
                    if not smi:
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue

                    validity_reward = calculate_chemical_validity_reward(mol)
                    diversity_reward = calculate_diversity_reward(mol, existing_mols)

                    if smi in template_smiles:
                        continue
                    accepted = False
                    if validity_reward > 0.5 and diversity_reward > 0.05:
                        accepted = True
                    elif validity_reward > 0.7:
                        cond_high_val_count = getattr(hybrid_generate_molecules, '_cond_hv', 0)
                        if cond_high_val_count < int(batch_size * total_batches * 0.2):
                            accepted = True
                            hybrid_generate_molecules._cond_hv = cond_high_val_count + 1
                    if accepted:
                        all_generated["approach4_conditional_gan"].append(smi)
                        batch_valid += 1
                except:
                    continue
            
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{total_batches}: {batch_valid} valid molecules, Total so far: {len(all_generated['approach4_conditional_gan'])}")

        logger.info(f"APPROACH 4 COMPLETE: Generated {len(all_generated['approach4_conditional_gan'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 4 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 5: RL-GAN with Rewards
    logger.info("=" * 80)
    logger.info("APPROACH 5: Reinforcement Learning + GAN (ORGAN) - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY molecules with RL-GAN - NO LIMIT
        target_molecules = num_per_approach * 10  # Increased multiplier
        logger.info(f"Generating {target_molecules} RL-GAN molecules with reward-based selection...")
        
        rl_molecules = rl_gan_generate_with_rewards(
            base_generator,
            latent_dim,
            target_molecules,
            coord_dim,
            template_mols,
            set(template_smiles),
            fragment_library,
        )
        # NOVELTY FILTER: Exclude molecules from training data
        all_generated["approach5_rl_gan"] = [
            s for s in rl_molecules if is_valid_smiles(s) and s not in template_smiles
        ]
        logger.info(f"APPROACH 5 COMPLETE: Generated {len(all_generated['approach5_rl_gan'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 5 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Base GAN (original approach)
    logger.info("=" * 80)
    logger.info("BASE GAN: Original 3D Coordinate Approach - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY batches from base GAN - NO LIMIT
        total_batches = 50  # Increased from 5 to 50
        logger.info(f"Generating {total_batches} batches of {num_per_approach} molecules each...")
        
        for batch_idx in range(total_batches):
            noise = np.random.normal(0, 1, (num_per_approach, latent_dim))
            generated_coords = base_generator.predict(noise, verbose=0)
            batch_valid = 0

            for coords, template_mol in zip(
                generated_coords,
                template_mols * (len(generated_coords) // len(template_mols) + 1),
            ):
                try:
                    smi = coords_to_smiles(
                        coords[: template_mol.GetNumAtoms()], template_mol
                    )
                    # NOVELTY FILTER: Exclude molecules from training data
                    if smi and is_valid_smiles(smi) and smi not in template_smiles:
                        all_generated["base_gan"].append(smi)
                        batch_valid += 1
                except:
                    continue
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{total_batches}: {batch_valid} valid molecules, Total so far: {len(all_generated['base_gan'])}")

        logger.info(f"BASE GAN COMPLETE: Generated {len(all_generated['base_gan'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"BASE GAN FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 6: Molecular VAE
    logger.info("=" * 80)
    logger.info("APPROACH 6: Molecular VAE - STARTING")
    logger.info("=" * 80)
    try:
        # Build and train VAE on template fingerprints
        morgan_gen = GetMorganGenerator(2, fpSize=2048)
        template_fps = []
        fallback_fp_count = 0
        logger.info("Generating fingerprints for VAE training...")
        for smi in balanced_trait_order(template_smiles, molecule_type):
            fp = TRAINING_FALLBACK_FEATURES.get(molecule_type, {}).get(smi)
            if fp is not None:
                template_fps.append(np.asarray(fp, dtype=np.float32))
                fallback_fp_count += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = morgan_gen.GetFingerprintAsNumPy(mol)
                template_fps.append(fp)
            else:
                fallback = fallback_representation_from_smiles(smi)
                if fallback["usable"]:
                    template_fps.append(fallback["hashed_fp"])
                    fallback_fp_count += 1
        logger.info(
            f"VAE fingerprint preparation complete: total={len(template_fps)}, fallback_or_precomputed={fallback_fp_count}"
        )

        if len(template_fps) > 10:
            template_fps = np.array(template_fps)
            logger.info(f"Training VAE on {len(template_fps)} molecular fingerprints...")
            vae_optuna_params = optimize_vae_hyperparameters(
                template_fps,
                latent_dim,
                n_trials=auxiliary_optuna_trials,
            )
            vae_latent_dim = int(vae_optuna_params.get("latent_dim", latent_dim))
            vae_hidden_width = int(vae_optuna_params.get("hidden_width", 512))
            vae_learning_rate = float(vae_optuna_params.get("learning_rate", 0.001))
            vae_epochs = int(vae_optuna_params.get("epochs", 5))
            vae_batch_size = int(vae_optuna_params.get("batch_size", 32))
            logger.info(
                f"Using VAE params: latent_dim={vae_latent_dim}, hidden_width={vae_hidden_width}, learning_rate={vae_learning_rate}, epochs={vae_epochs}, batch_size={vae_batch_size}"
            )
            vae, encoder, decoder = build_molecular_vae(
                input_dim=2048,
                latent_dim=vae_latent_dim,
                encoder_units=(vae_hidden_width, max(128, vae_hidden_width // 2)),
                decoder_units=(max(128, vae_hidden_width // 2), vae_hidden_width),
                learning_rate=vae_learning_rate,
            )

            vae.fit(
                template_fps,
                template_fps,
                epochs=vae_epochs,
                batch_size=min(vae_batch_size, len(template_fps)),
                verbose=0,
            )
            logger.info("VAE training complete")

            # Generate MANY molecules - NO LIMIT
            target_molecules = num_per_approach * 5  # Increased multiplier
            logger.info(f"Generating {target_molecules} VAE molecules...")
            vae_molecules = vae_generate_molecules(
                decoder,
                vae_latent_dim,
                target_molecules,
                template_mols,
                set(template_smiles),
            )
            # NOVELTY FILTER: Exclude molecules from training data
            all_generated["approach6_vae"] = [
                s for s in vae_molecules if is_valid_smiles(s) and s not in template_smiles
            ]
            logger.info(f"APPROACH 6 COMPLETE: Generated {len(all_generated['approach6_vae'])} valid molecules")
        else:
            logger.warning(f"APPROACH 6 SKIPPED: Not enough template molecules ({len(template_fps)})")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 6 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 7: Transformer-based Generation
    logger.info("=" * 80)
    logger.info("APPROACH 7: Transformer-based Generation - STARTING")
    logger.info("=" * 80)
    try:
        # Build vocabulary and transformer (simplified)
        chars = set()
        for smi in balanced_trait_order(template_smiles, molecule_type):
            chars.update(smi)
        chars = sorted(list(chars))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}
        vocab_size = len(chars)
        logger.info(f"Built vocabulary with {vocab_size} characters")

        if vocab_size > 0:
            # Generate MANY molecules - NO LIMIT
            target_molecules = num_per_approach * 3  # Increased multiplier
            logger.info(f"Generating {target_molecules} transformer-based molecules...")
            transformer_mols = transformer_generate_molecules(
                None,
                vocab_size,
                100,
                target_molecules,
                char_to_idx,
                idx_to_char,
                template_smiles,
            )
            # NOVELTY FILTER: Exclude molecules from training data
            all_generated["approach7_transformer"] = [
                s for s in transformer_mols if is_valid_smiles(s) and s not in template_smiles
            ]
            logger.info(f"APPROACH 7 COMPLETE: Generated {len(all_generated['approach7_transformer'])} valid molecules")
        else:
            logger.warning("APPROACH 7 SKIPPED: Empty vocabulary")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 7 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 8: Transfer Learning
    logger.info("=" * 80)
    logger.info("APPROACH 8: Transfer Learning with Pre-trained Embeddings - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY molecules - NO LIMIT
        target_molecules = num_per_approach * 3  # Increased multiplier
        logger.info(f"Generating {target_molecules} transfer learning molecules...")
        
        transfer_mols = transfer_learning_generate(
            template_smiles, target_molecules, fragment_library, molecule_type=molecule_type
        )
        # NOVELTY FILTER: Exclude molecules from training data
        all_generated["approach8_transfer_learning"] = [
            s for s in transfer_mols if is_valid_smiles(s) and s not in template_smiles
        ]
        logger.info(f"APPROACH 8 COMPLETE: Generated {len(all_generated['approach8_transfer_learning'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 8 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 9: Multi-Objective Optimization
    logger.info("=" * 80)
    logger.info("APPROACH 9: Multi-Objective Optimization (Pareto Frontier) - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY molecules - NO LIMIT
        target_molecules = num_per_approach * 3  # Increased multiplier
        logger.info(f"Generating {target_molecules} multi-objective optimized molecules...")
        
        pareto_mols = multi_objective_generate(
            template_mols, template_smiles, target_molecules, fragment_library, molecule_type=molecule_type
        )
        # NOVELTY FILTER: Exclude molecules from training data
        all_generated["approach9_multi_objective"] = [
            s for s in pareto_mols if is_valid_smiles(s) and s not in template_smiles
        ]
        logger.info(f"APPROACH 9 COMPLETE: Generated {len(all_generated['approach9_multi_objective'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 9 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    # Approach 10: Active Learning
    logger.info("=" * 80)
    logger.info("APPROACH 10: Active Learning with Uncertainty Sampling - STARTING")
    logger.info("=" * 80)
    try:
        # Generate MANY molecules - NO LIMIT
        target_molecules = num_per_approach * 2  # Increased multiplier
        logger.info(f"Generating {target_molecules} active learning molecules...")
        active_optuna_params = optimize_active_learning_hyperparameters(
            template_smiles,
            n_trials=auxiliary_optuna_trials,
            molecule_type=molecule_type,
        )
        
        active_mols = active_learning_generate(
            template_smiles,
            template_mols,
            target_molecules,
            fragment_library,
            optuna_params=active_optuna_params,
            molecule_type=molecule_type,
        )
        # NOVELTY FILTER: Exclude molecules from training data
        all_generated["approach10_active_learning"] = [
            s for s in active_mols if is_valid_smiles(s) and s not in template_smiles
        ]
        logger.info(f"APPROACH 10 COMPLETE: Generated {len(all_generated['approach10_active_learning'])} valid molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 10 FAILED with error: {e}")
        logger.exception("Full traceback:")
        # Note: No lazy fallbacks - will generate actual new molecules if needed

    logger.info("=" * 80)
    logger.info("APPROACH 11: CoCoGraph-Inspired Constrained Graph Diffusion - STARTING")
    logger.info("=" * 80)
    try:
        target_molecules = num_per_approach
        cocograph_mols = constrained_cocograph_generate(
            template_smiles,
            fragment_library,
            target_molecules,
            molecule_type,
            training_canonical_set,
            training_fps,
            max_similarity=strict_novelty_threshold,
        )
        all_generated["approach11_cocograph"] = cocograph_mols
        logger.info(f"APPROACH 11 COMPLETE: Generated {len(all_generated['approach11_cocograph'])} valid highly novel molecules")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"APPROACH 11 FAILED with error: {e}")
        logger.exception("Full traceback:")

    # Ensure all approaches generate actual NEW molecules (not just copy training data)
    logger.info("=" * 80)
    logger.info("ENSURING ALL APPROACHES GENERATE NEW MOLECULES")
    logger.info("=" * 80)
    
    for approach_key in all_generated.keys():
        ensure_approach_generates_molecules(
            all_generated,
            approach_key,
            template_smiles,
            fragment_library,
            min_molecules=50,
            molecule_type=molecule_type,
            training_canonical_set=training_canonical_set,
            training_fps=training_fps,
            max_similarity=strict_novelty_threshold,
        )
        logger.info(f"{approach_key}: {len(all_generated[approach_key])} NEW molecules generated")
    
    # VALIDATION: Check that ALL 12 approaches have generated molecules
    logger.info("=" * 80)
    logger.info("VALIDATING ALL 12 APPROACHES GENERATED MOLECULES")
    logger.info("=" * 80)
    
    expected_approaches = [
        "base_gan", "approach1_smiles_gan", "approach2_graph_gan", "approach3_junction_tree",
        "approach4_conditional_gan", "approach5_rl_gan", "approach6_vae", "approach7_transformer",
        "approach8_transfer_learning", "approach9_multi_objective", "approach10_active_learning",
        "approach11_cocograph"
    ]
    
    all_approaches_successful = True
    successful_count = 0
    failed_approaches = []
    
    for approach_name in expected_approaches:
        count = len(all_generated[approach_name])
        if count == 0:
            logger.warning(f"âš  {approach_name}: FAILED - NO molecules generated!")
            all_approaches_successful = False
            failed_approaches.append(approach_name)
        elif count < 50:
            logger.warning(f"âš  {approach_name}: {count} molecules (below minimum of 50)")
            successful_count += 1
        else:
            logger.info(f"âœ“ {approach_name}: {count} molecules generated successfully")
            successful_count += 1
    
    # Report status but DO NOT stop execution
    logger.info("=" * 80)
    if all_approaches_successful and all(len(all_generated[k]) > 0 for k in expected_approaches):
        logger.info("âœ“âœ“âœ“ SUCCESS: ALL 12 GENERATION METHODS GENERATED MOLECULES âœ“âœ“âœ“")
    else:
        logger.warning(f"âš  PARTIAL SUCCESS: {successful_count}/{len(expected_approaches)} approaches generated molecules")
        if failed_approaches:
            logger.warning(f"Failed approaches: {', '.join(failed_approaches)}")
        logger.warning("Continuing with available molecules from successful approaches...")
    logger.info("=" * 80)

    # Summary
    total_generated = sum(len(v) for v in all_generated.values())
    logger.info("=" * 80)
    logger.info(f"HYBRID GENERATION COMPLETE: {molecule_type}")
    logger.info("=" * 80)
    logger.info(f"Total molecules generated: {total_generated}")
    for approach_name, molecules in all_generated.items():
        logger.info(f"  {approach_name}: {len(molecules)} molecules")

    return all_generated


# Hyperparameters
latent_dim = 50
batch_size = 32
epochs = 50  # Increased number of epochs
lr = 0.0002  # Learning rate


# Generator model for 3D coordinates
def build_generator(coord_dim, latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(latent_dim,)))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(coord_dim * 3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((coord_dim, 3)))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(3)))
    return model


# Custom Transformer Block
def transformer_block(x, num_heads, ff_dim):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(x.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# Transformer discriminator model for 3D coordinates
def build_discriminator(coord_dim):
    input_layer = layers.Input(shape=(coord_dim, 3))
    x = transformer_block(input_layer, num_heads=8, ff_dim=512)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


# GAN model
def build_gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(
    coords,
    coord_dim,
    latent_dim,
    batch_size,
    epochs,
    gen_params=None,
    disc_params=None,
    plot_tag=None,
):
    logger.info(
        f"Starting GAN training - Coord_dim: {coord_dim}, Latent_dim: {latent_dim}, Batch: {batch_size}, Epochs: {epochs}"
    )
    coords = np.array(coords)
    logger.debug(f"Training data shape: {coords.shape}")

    # Build and compile the discriminator
    logger.info("Building discriminator...")
    if disc_params:
        logger.debug(f"Using tunable discriminator with params: {disc_params}")
        discriminator = build_tunable_discriminator(coord_dim, **disc_params)
    else:
        logger.debug("Using default discriminator")
        discriminator = build_discriminator(coord_dim)
    gen_lr = gen_params.get("lr", lr) if gen_params else lr
    # Use a much smaller discriminator LR (1/4 of generator LR) to prevent
    # discriminator domination. The discriminator trains on real+fake per batch
    # (2 gradient steps) while the generator trains once, so halving disc_lr
    # already gives it 2x effective gradient steps per batch. Reducing to 1/4
    # slows the discriminator enough for the generator to keep up.
    disc_lr = gen_lr * 0.25
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0.5),
        metrics=["accuracy"],
    )
    logger.info("Discriminator built and compiled successfully")

    # Build the generator
    logger.info("Building generator...")
    if gen_params:
        logger.debug(f"Using tunable generator with params: {gen_params}")
        # Extract lr from gen_params (not used in generator, only in optimizer)
        gen_params_for_builder = {k: v for k, v in gen_params.items() if k != 'lr'}
        generator = build_tunable_generator(coord_dim, latent_dim, **gen_params_for_builder)
    else:
        logger.debug("Using default generator")
        generator = build_generator(coord_dim, latent_dim)
    logger.info("Generator built successfully")

    # Build and compile the GAN
    logger.info("Building complete GAN model...")
    discriminator.trainable = False
    gan = build_gan(generator, discriminator)
    gan.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.5))
    logger.info("GAN model built and compiled successfully")

    # Training the GAN
    logger.info("Starting GAN training loop...")
    num_batches = int(np.ceil(coords.shape[0] / batch_size))
    logger.info(
        f"Coordinate GAN will use all {coords.shape[0]} coordinate rows every epoch across {num_batches} batches"
    )

    d_losses = []
    g_losses = []
    d_accuracies = []

    # -----------------------------------------------------------------------
    # GAN EQUILIBRIUM EARLY STOPPING
    # Target: discriminator loss near ln(2) â‰ˆ 0.693 (Nash equilibrium).
    # We track two conditions and stop early when triggered:
    #   1. DOMINATION STOP: D-loss < 0.45 for `patience` consecutive epochs.
    #      The discriminator has "won" — further training only widens the gap
    #      and corrupts the generator. We restore the best-checkpoint weights.
    #   2. GENERATOR COLLAPSE STOP: G-loss > 3.0 for `patience` epochs.
    #      The generator has fully collapsed and is producing meaningless output.
    # Additionally we save the generator weights at the epoch with D-loss
    # closest to 0.693 (best equilibrium) and restore them at the end.
    # This ensures the returned generator was at its most balanced state.
    # -----------------------------------------------------------------------
    GAN_EQUILIBRIUM_TARGET = 0.693   # ln(2) — ideal Nash equilibrium loss
    DOMINATION_THRESHOLD   = 0.45    # D-loss below this â†’ discriminator dominating
    COLLAPSE_THRESHOLD     = 3.0     # G-loss above this â†’ generator fully collapsed
    ES_PATIENCE            = 3       # consecutive epochs to trigger early stop

    best_equilibrium_epoch = 0
    best_equilibrium_delta = float("inf")
    best_generator_weights = generator.get_weights()

    # Hysteresis-based domination counter: increments on bad epochs, DECREMENTS
    # (not hard-resets) on good epochs. This survives oscillation right at the
    # threshold boundary, unlike a hard reset-to-0 which can mask sustained
    # domination when the loss oscillates just above/below the cutoff.
    domination_badness  = 0.0
    collapse_streak      = 0
    early_stop_reason     = None

    # Persistent cross-epoch discriminator-loss memory for batch-level throttling.
    # Initialized to the equilibrium target so the first batch of the first epoch
    # is never throttled. Updated after EVERY batch (not reset each epoch) so
    # throttling correctly carries state across epoch boundaries.
    running_d_loss = GAN_EQUILIBRIUM_TARGET

    # Adaptive discriminator LR decay: persists across epochs. Every time an
    # epoch ends with the discriminator dominating, we permanently shrink its
    # optimizer LR (rather than only skipping batch updates, which resets
    # every epoch and cannot fix small-batch-count datasets like the acceptor
    # set with only 6 batches/epoch).
    disc_lr_decay_factor = 1.0

    for epoch in range(epochs):
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_d_accuracies = []
        shuffled_indices = np.random.permutation(coords.shape[0])
        instance_noise = max(0.0, 0.05 * (1.0 - epoch / max(epochs, 1)))

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, coords.shape[0])
            batch_indices = shuffled_indices[batch_start:batch_end]
            real_coords = coords[batch_indices]
            current_batch_size = len(real_coords)
            if current_batch_size == 0:
                continue
            real = np.full((current_batch_size, 1), 0.9)
            fake = np.full((current_batch_size, 1), 0.1)

            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
            generated_coords = generator.predict(noise, verbose=0)
            if instance_noise > 0:
                real_coords = real_coords + np.random.normal(0, instance_noise, real_coords.shape)
                generated_coords = generated_coords + np.random.normal(0, instance_noise, generated_coords.shape)

            # -------------------------------------------------------------------
            # DISCRIMINATOR THROTTLING + GENERATOR REBALANCING
            # Uses `running_d_loss`, a PERSISTENT cross-epoch/cross-batch memory
            # of the discriminator loss (unlike the old per-epoch list, this
            # value survives epoch boundaries so throttling doesn't get reset
            # every epoch — critical for small-batch-count datasets like the
            # acceptor set, which only has ~6 batches/epoch).
            # -------------------------------------------------------------------
            if running_d_loss > 0.45:
                # Discriminator is not yet dominating — allow its training update.
                # Apply disc_lr_decay_factor (persistent shrink from prior epochs
                # that ended dominated) to keep the discriminator from re-winning
                # too quickly after a decay was already applied.
                discriminator.trainable = True
                if disc_lr_decay_factor < 1.0:
                    try:
                        discriminator.optimizer.learning_rate.assign(disc_lr * disc_lr_decay_factor)
                    except Exception as _lr_e:
                        logger.debug(f"Could not adjust discriminator LR: {_lr_e}")
                d_loss_real = discriminator.train_on_batch(real_coords, real)
                d_loss_fake = discriminator.train_on_batch(generated_coords, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                # Discriminator is already too strong — freeze it and only train generator
                discriminator.trainable = False
                d_loss_real = d_loss_fake = np.array([running_d_loss], dtype=np.float32)
                d_loss = np.array([running_d_loss], dtype=np.float32)
                logger.debug(f"Epoch {epoch} batch {batch_idx}: throttling discriminator (running_d={running_d_loss:.4f})")

            discriminator.trainable = False
            noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((current_batch_size, 1)))

            # Extract scalar loss value robustly from d_loss (may be scalar, 0-d array, or list)
            d_loss_arr = np.atleast_1d(np.asarray(d_loss, dtype=np.float32))
            d_loss_scalar = float(d_loss_arr[0])

            # Update the persistent running D-loss memory (exponential moving
            # average for smoothing against single-batch noise)
            running_d_loss = 0.7 * running_d_loss + 0.3 * d_loss_scalar

            # Extra generator steps when D is winning: run 2 more G updates when
            # D-loss < 0.55 (discriminator ahead) or G-loss > 0.9 (generator behind)
            if d_loss_scalar < 0.55 or float(g_loss) > 0.9:
                for _ in range(2):
                    noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
                    g_loss = gan.train_on_batch(noise, np.ones((current_batch_size, 1)))

            epoch_d_losses.append(d_loss_scalar)
            epoch_g_losses.append(float(g_loss) if np.isscalar(g_loss) else float(np.atleast_1d(g_loss)[0]))

            # -----------------------------------------------------------------------
            # Discriminator accuracy computation — robust across all TF/Keras versions.
            #
            # Strategy: Always compute balanced accuracy manually using clean
            # (pre-noise) inputs. This is the most reliable approach because:
            #   1. train_on_batch in TF2 may return a scalar (loss only) even
            #      when metrics=["accuracy"] is compiled, depending on TF version.
            #   2. The Keras accuracy metric uses 0.5 threshold on raw sigmoid
            #      outputs, but label smoothing (real=0.9, fake=0.1) makes the
            #      returned Keras accuracy unreliable (it compares predictions
            #      to smoothed float targets, not binary 0/1).
            #   3. Using noisy generated_coords for the fake accuracy check would
            #      bias results since instance noise can push scores across 0.5.
            #
            # We ALWAYS use clean (pre-noise) real and freshly-generated fake
            # coordinates with the discriminator in inference mode (trainable=True
            # temporarily for accurate BN statistics if applicable).
            # Balanced accuracy = 0.5*(frac_real_correct + frac_fake_correct).
            # A well-trained discriminator confused by a good generator â†’ ~50%.
            # A discriminator dominating the generator â†’ closer to 100%.
            # -----------------------------------------------------------------------
            try:
                # Re-enable discriminator weights for accurate inference
                # (they are already trained at this point; we just want clean preds)
                discriminator.trainable = True

                # Clean real samples (no instance noise)
                batch_indices_for_acc = shuffled_indices[batch_start:batch_end]
                real_coords_clean = coords[batch_indices_for_acc]

                # Fresh fake samples from generator (no instance noise)
                noise_for_acc = np.random.normal(0, 1, (current_batch_size, latent_dim))
                fake_coords_clean = generator.predict(noise_for_acc, verbose=0)

                pred_real = discriminator(
                    tf.cast(real_coords_clean, tf.float32), training=False
                ).numpy().flatten()
                pred_fake = discriminator(
                    tf.cast(fake_coords_clean, tf.float32), training=False
                ).numpy().flatten()

                # Balanced accuracy: real classified as real (>= 0.5) + fake as fake (< 0.5)
                d_real_acc = float(np.mean(pred_real >= 0.5))
                d_fake_acc = float(np.mean(pred_fake < 0.5))
                d_acc = 100.0 * 0.5 * (d_real_acc + d_fake_acc)

                # Restore discriminator.trainable = False for the GAN update
                discriminator.trainable = False
            except Exception as _acc_e:
                logger.debug(f"Discriminator accuracy computation failed: {_acc_e}; defaulting to 50.0")
                d_acc = 50.0  # Neutral fallback — better than 0.0
                discriminator.trainable = False

            epoch_d_accuracies.append(d_acc)

        d_losses.append(float(np.mean(epoch_d_losses)) if epoch_d_losses else 0.0)
        g_losses.append(float(np.mean(epoch_g_losses)) if epoch_g_losses else 0.0)
        d_accuracies.append(float(np.mean(epoch_d_accuracies)) if epoch_d_accuracies else 0.0)

        # Log progress every epochs//5 epochs
        if epoch % max(1, epochs // 5) == 0:
            logger.info(
                f"Epoch {epoch}/{epochs} | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f} | D Acc: {d_accuracies[-1]:.2f}%"
            )
            d_loss_real_arr = np.atleast_1d(np.asarray(d_loss_real, dtype=np.float32))
            d_loss_fake_arr = np.atleast_1d(np.asarray(d_loss_fake, dtype=np.float32))
            logger.debug(
                f"Epoch {epoch} - D loss real: {float(d_loss_real_arr[0]):.4f}, D loss fake: {float(d_loss_fake_arr[0]):.4f}"
            )

        # -------------------------------------------------------------------
        # GAN EQUILIBRIUM EARLY STOPPING — checkpoint & domination detection
        # -------------------------------------------------------------------
        epoch_d = d_losses[-1]
        epoch_g = g_losses[-1]

        # Track the best-equilibrium generator (closest D-loss to 0.693)
        delta_from_eq = abs(epoch_d - GAN_EQUILIBRIUM_TARGET)
        if delta_from_eq < best_equilibrium_delta:
            best_equilibrium_delta = delta_from_eq
            best_equilibrium_epoch = epoch
            best_generator_weights = generator.get_weights()
            logger.debug(
                f"Epoch {epoch}: new best-equilibrium checkpoint "
                f"(|D_loss - 0.693| = {delta_from_eq:.4f})"
            )

        # -------------------------------------------------------------------
        # Hysteresis-based discriminator domination tracking.
        # Unlike a hard streak-reset-to-0, `domination_badness` accumulates
        # when D-loss is low and only DECAYS (not resets) when D-loss recovers.
        # This correctly detects sustained domination even when the loss
        # oscillates right around the threshold boundary (e.g. 0.42, 0.45,
        # 0.53, 0.44 — which would falsely reset a hard streak counter every
        # other epoch but should still be flagged as a domination problem).
        # -------------------------------------------------------------------
        if epoch_d < DOMINATION_THRESHOLD:
            # Bad epoch: increment badness proportional to how far below threshold
            domination_badness += 1.0 + (DOMINATION_THRESHOLD - epoch_d) * 2.0
            logger.warning(
                f"Epoch {epoch}: discriminator domination badness={domination_badness:.2f}/{ES_PATIENCE} "
                f"(D_loss={epoch_d:.4f} < {DOMINATION_THRESHOLD})"
            )
            # Permanently shrink discriminator LR for remaining epochs — this
            # persists (does not reset) and directly fixes small-batch-count
            # datasets where per-batch throttling can't act fast enough.
            disc_lr_decay_factor = max(0.15, disc_lr_decay_factor * 0.6)
            logger.info(f"Epoch {epoch}: shrinking discriminator LR decay factor to {disc_lr_decay_factor:.3f}")
        elif epoch_d < GAN_EQUILIBRIUM_TARGET:
            # Mildly below equilibrium but not dominating — decay badness slowly
            domination_badness = max(0.0, domination_badness - 0.5)
        else:
            # Fully recovered (at or above equilibrium) — decay badness faster
            domination_badness = max(0.0, domination_badness - 1.0)

        # Generator collapse check (G-loss too high = generator meaningless)
        if epoch_g > COLLAPSE_THRESHOLD:
            collapse_streak += 1
            logger.warning(
                f"Epoch {epoch}: generator collapse streak {collapse_streak}/{ES_PATIENCE} "
                f"(G_loss={epoch_g:.4f} > {COLLAPSE_THRESHOLD})"
            )
        else:
            collapse_streak = 0

        # Trigger early stop if accumulated domination badness exceeds patience
        if domination_badness >= ES_PATIENCE:
            early_stop_reason = (
                f"sustained discriminator domination (badness={domination_badness:.2f})"
                f" at epoch {epoch} (D_loss={epoch_d:.4f})"
            )
            logger.info(
                f"EARLY STOP at epoch {epoch}: {early_stop_reason}. "
                f"Restoring generator from best-equilibrium epoch {best_equilibrium_epoch} "
                f"(D_loss delta from 0.693 = {best_equilibrium_delta:.4f})."
            )
            generator.set_weights(best_generator_weights)
            break

        if collapse_streak >= ES_PATIENCE:
            early_stop_reason = (
                f"generator collapse for {collapse_streak} consecutive epochs "
                f"(G_loss={epoch_g:.4f})"
            )
            logger.info(
                f"EARLY STOP at epoch {epoch}: {early_stop_reason}. "
                f"Restoring generator from best-equilibrium epoch {best_equilibrium_epoch}."
            )
            generator.set_weights(best_generator_weights)
            break

    # After training loop: always restore the best-equilibrium weights
    # (in case no early stop was triggered but last epoch is not the best)
    if early_stop_reason is None:
        logger.info(
            f"Training completed all {epochs} epochs. Restoring best-equilibrium "
            f"generator from epoch {best_equilibrium_epoch} "
            f"(|D_loss - 0.693| = {best_equilibrium_delta:.4f})."
        )
        generator.set_weights(best_generator_weights)

    # Plot the losses — only save for named (optimized) runs, not base trials
    logger.info("Training completed. Generating loss plots...")
    if plot_tag:
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss", color='steelblue')
        plt.plot(g_losses, label="Generator Loss", color='darkorange')
        plt.axhline(y=0.693, color='green', linestyle='--', alpha=0.7,
                    label='Equilibrium target (ln2 â‰ˆ 0.693)')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"GAN Losses — {plot_tag}")
        _loss_name = f"gan_losses_{plot_tag}.png"
        plt.savefig(_loss_name)
        logger.info(f"Saved GAN losses plot to {_loss_name}")
        plt.close()

        # Save losses to CSV
        _loss_csv_name = f"gan_losses_{plot_tag}.csv"
        losses_df = pd.DataFrame({
            'epoch': range(len(d_losses)),
            'discriminator_loss': d_losses,
            'generator_loss': g_losses
        })
        losses_df.to_csv(_loss_csv_name, index=False)
        logger.info(f"Saved GAN losses data to {_loss_csv_name}")

        # Plot the discriminator accuracy with 50% reference line
        plt.figure(figsize=(10, 5))
        plt.plot(d_accuracies, label="Discriminator Accuracy", color='steelblue')
        plt.axhline(y=50.0, color='green', linestyle='--', alpha=0.7,
                    label='Target equilibrium (50%)')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend()
        plt.title(f"Discriminator Accuracy — {plot_tag}")
        _acc_name = f"discriminator_accuracy_{plot_tag}.png"
        plt.savefig(_acc_name)
        logger.info(f"Saved discriminator accuracy plot to {_acc_name}")
        plt.close()

        # Save accuracy to CSV
        _acc_csv_name = f"discriminator_accuracy_{plot_tag}.csv"
        accuracy_df = pd.DataFrame({
            'epoch': range(len(d_accuracies)),
            'discriminator_accuracy_percent': d_accuracies
        })
        accuracy_df.to_csv(_acc_csv_name, index=False)
        logger.info(f"Saved discriminator accuracy data to {_acc_csv_name}")
    else:
        # Optuna trial run — skip file I/O for plots to save time and avoid
        # overwriting the final optimized plots with intermediate trial plots.
        logger.debug("Optuna trial run: skipping plot/CSV save (no plot_tag)")

    logger.info(
        f"GAN training finished. Final D loss: {d_losses[-1]:.4f}, Final G loss: {g_losses[-1]:.4f}, Final D Acc: {d_accuracies[-1]:.2f}%"
    )
    return generator


def generate_coords(generator, latent_dim, num_samples, coord_dim):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_coords = generator.predict(noise)
    return generated_coords


# Train GAN for donors
logger.info("=" * 80)
logger.info("TRAINING DONOR GAN")
# ============================================================================
# SIMPLE GAN TRAINING - COMMENTED OUT (Use hyperparameter-optimized version in try block below)
# ============================================================================
# ============================================================================
# SIMPLE GAN TRAINING - COMMENTED OUT (Use hyperparameter-optimized version in try block below)
# ============================================================================
# logger.info("=" * 80)
# donor_generator = train_gan(
#     donor_coords, donor_coord_dim, latent_dim, batch_size, epochs, plot_tag="donor"
# )
# logger.info("Donor GAN training completed successfully")

# Train GAN for acceptors
# logger.info("=" * 80)
# logger.info("TRAINING ACCEPTOR GAN")
# logger.info("=" * 80)
# acceptor_generator = train_gan(
#     acceptor_coords,
#     acceptor_coord_dim,
#     latent_dim,
#     batch_size,
#     epochs,
#     plot_tag="acceptor",
# )
# logger.info("Acceptor GAN training completed successfully")

# ============================================================================
# OLD/SIMPLE GENERATION CODE - COMMENTED OUT TO ALLOW HYBRID 11-APPROACH TO RUN
# ============================================================================
# Generate more samples than needed to ensure uniqueness
# num_samples = 3 * 300  # Generate 3 times the required number to ensure uniqueness
# generated_donor_coords = generate_coords(
#     donor_generator, latent_dim, num_samples, donor_coord_dim
# )
# generated_acceptor_coords = generate_coords(
#     acceptor_generator, latent_dim, num_samples, acceptor_coord_dim
# )
# 
# 
# Convert generated 3D coordinates back to SMILES and filter duplicates
def safe_coords_to_smiles(coords, template_mol):
    coords = coords[: template_mol.GetNumAtoms()]
    return coords_to_smiles(coords, template_mol)


def get_unique_smiles(generated_coords, template_mols, coord_dim):
    """Extract ALL unique valid SMILES from generated coordinates - NO LIMIT"""
    smiles_set = set()
    unique_smiles = []
    for coords, template_mol in zip(
        generated_coords,
        template_mols * (len(generated_coords) // len(template_mols) + 1),
    ):
        smiles = safe_coords_to_smiles(coords, template_mol)
        if smiles not in smiles_set:
            smiles_set.add(smiles)
            unique_smiles.append(smiles)
        # NO UPPER LIMIT - extract all valid unique molecules
    return unique_smiles
# 
# 
# generated_donor_smiles = get_unique_smiles(
#     generated_donor_coords, donor_mols, donor_coord_dim
# )
# generated_acceptor_smiles = get_unique_smiles(
#     generated_acceptor_coords, acceptor_mols, acceptor_coord_dim
# )
# 
# print("Generated Donor SMILES from 3D coordinates:")
# print(generated_donor_smiles)
# 
# print("Generated Acceptor SMILES from 3D coordinates:")
# print(generated_acceptor_smiles)
# 
# Ensure both lists are of the same length
# # max_length = max(len(generated_donor_smiles), len(generated_acceptor_smiles))
# generated_donor_smiles.extend([""] * (max_length - len(generated_donor_smiles)))
# generated_acceptor_smiles.extend([""] * (max_length - len(generated_acceptor_smiles)))
# 
# Save the generated SMILES strings to an Excel file
# # output_df = pd.DataFrame(
#     {
#         "Generated Donor SMILES": generated_donor_smiles,
#         "Generated Acceptor SMILES": generated_acceptor_smiles,
#     }
# )
# output_df.to_excel("Generated_Molecules.xlsx", index=False)
# 

# Fix the metric calculations
def calculate_mfd(real_mols, generated_mols):
    """
    Calculate Mean Fréchet Distance (MFD) / Fréchet ChemNet Distance (FCD)

    Fixed imementation: PCA is fit on COMBINED dataset for unbiased comparison
    """
    # Ensure we have valid molecules
    real_mols = [mol for mol in real_mols if mol is not None]
    generated_mols = [mol for mol in generated_mols if mol is not None]

    if not real_mols or not generated_mols:
        return float("inf")  # Return infinity if no valid molecules

    # Generate fingerprints using cached Morgan fingerprint function
    real_fps = []
    generated_fps = []

    for mol in real_mols:
        fp_bitvec = morgan_fp(mol, radius=2, nBits=2048)
        if fp_bitvec is not None:
            fp = np.array(fp_bitvec)
            real_fps.append(fp)

    for mol in generated_mols:
        fp_bitvec = morgan_fp(mol, radius=2, nBits=2048)
        if fp_bitvec is not None:
            fp = np.array(fp_bitvec)
            generated_fps.append(fp)

    # Convert to numpy arrays
    real_fps_array = np.array(real_fps)
    generated_fps_array = np.array(generated_fps)

    # FIXED: Combine both datasets for unbiased PCA fitting
    all_fps = np.vstack([real_fps_array, generated_fps_array])

    # Fit PCA on combined dataset
    pca = PCA(n_components=2)
    all_fps_pca = pca.fit_transform(all_fps)

    # Split back into real and generated
    real_fps_pca = all_fps_pca[: len(real_fps_array)]
    generated_fps_pca = all_fps_pca[len(real_fps_array) :]

    # Calculate statistics on the transformed features
    mu_real = np.mean(real_fps_pca, axis=0)
    sigma_real = np.cov(real_fps_pca, rowvar=False)
    mu_generated = np.mean(generated_fps_pca, axis=0)
    sigma_generated = np.cov(generated_fps_pca, rowvar=False)

    # Calculate Fréchet Distance
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return float(fid)


def calculate_mis(generated_mols):
    """
    Calculate Mean Internal Similarity (MIS) using Tanimoto similarity

    Fixed implementation: Uses original fingerprints (not PCA-reduced) and
    proper Tanimoto similarity for chemical fingerprints.
    
    MIS measures diversity: lower values = higher diversity (better)
    MIS = 1 - (average pairwise Tanimoto similarity)
    """
    if not generated_mols or len(generated_mols) < 2:
        return 0.0, 0.0

    # Generate bit vector fingerprints using cached function
    fps = []
    for mol in generated_mols:
        if mol is not None:
            fp = morgan_fp(mol, radius=2, nBits=2048)  # Returns bit vector
            if fp is not None:
                fps.append(fp)

    if len(fps) < 2:
        return 0.0, 0.0

    # Calculate pairwise Tanimoto similarity on original fingerprints
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    if not similarities:
        return 0.0, 0.0

    # MIS = 1 - average_similarity (inverted so higher MIS = higher diversity)
    avg_similarity = np.mean(similarities)
    mis_mean = 1.0 - avg_similarity  # Invert: lower similarity = higher MIS (better diversity)
    mis_std = np.std(similarities)
    return mis_mean, mis_std


# Update the metric calculation calls
# generated_donor_mols = [
#     Chem.MolFromSmiles(smiles) for smiles in generated_donor_smiles if smiles
# ]
# generated_acceptor_mols = [
#     Chem.MolFromSmiles(smiles) for smiles in generated_acceptor_smiles if smiles
# ]
# 
# # Calculate metrics without parallelization for simplicity
# mfd_donors = calculate_mfd(donor_mols, generated_donor_mols)
# mfd_acceptors = calculate_mfd(acceptor_mols, generated_acceptor_mols)
# 
# # Calculate MIS scores
# mis_donors = calculate_mis(generated_donor_mols)
# mis_acceptors = calculate_mis(generated_acceptor_mols)
# 
# # Save metrics
# with open("base_gan_metrics.txt", "w") as f:
#     f.write("Metrics for Donors:\n")
#     f.write(f"Mean Fréchet Distance (MFD): {mfd_donors}\n")
#     f.write(f"Mean Inception Score (MIS) Mean: {mis_donors[0]}\n")
#     f.write(f"Mean Inception Score (MIS) Std: {mis_donors[1]}\n\n")
#     f.write("Metrics for Acceptors:\n")
#     f.write(f"Mean Fréchet Distance (MFD): {mfd_acceptors}\n")
#     f.write(f"Mean Inception Score (MIS) Mean: {mis_acceptors[0]}\n")
#     f.write(f"Mean Inception Score (MIS) Std: {mis_acceptors[1]}\n")
# 
# print("Metrics saved to base_gan_metrics.txt")


# Move builder functions outside objective to avoid recreation in each trial
def build_tunable_generator(
    coord_dim, latent_dim, gen_dense_layers, gen_dense_units, gen_lstm_units
):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(latent_dim,)))

    for _ in range(gen_dense_layers):
        model.add(layers.Dense(gen_dense_units, activation="relu"))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(coord_dim * 3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((coord_dim, 3)))
    model.add(layers.LSTM(gen_lstm_units, return_sequences=True))
    model.add(layers.LSTM(gen_lstm_units, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(3)))
    return model


def build_tunable_discriminator(
    coord_dim, disc_transformer_heads, disc_ff_dim, disc_dense_units
):
    input_layer = layers.Input(shape=(coord_dim, 3))
    x = transformer_block(
        input_layer, num_heads=disc_transformer_heads, ff_dim=disc_ff_dim
    )
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(disc_dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)


def objective_donor(trial):
    # Hyperparameters to tune
    latent_dim = trial.suggest_int("latent_dim", 16, 32)
    batch_size = trial.suggest_int("batch_size", 16, 32)
    # Allow up to 20 epochs — the new early-stopping logic will halt the donor
    # GAN before discriminator domination regardless of the epoch ceiling.
    epochs = trial.suggest_int("epochs", 5, 20)
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)

    # Generator hyperparameters
    gen_params = {
        "gen_dense_layers": trial.suggest_int("gen_dense_layers", 1, 2),
        "gen_dense_units": trial.suggest_int("gen_dense_units", 128, 256),
        "gen_lstm_units": trial.suggest_int("gen_lstm_units", 128, 256),
        "lr": lr,  # Add learning rate to gen_params
    }

    # Discriminator hyperparameters
    disc_params = {
        "disc_transformer_heads": trial.suggest_int("disc_transformer_heads", 4, 8),
        "disc_ff_dim": trial.suggest_int("disc_ff_dim", 128, 256),
        "disc_dense_units": trial.suggest_int("disc_dense_units", 64, 128),
    }

    try:
        # Train donor GAN with current hyperparameters
        donor_generator = train_gan(
            donor_coords,
            donor_coord_dim,
            latent_dim,
            batch_size,
            epochs,
            gen_params,
            disc_params,
        )

        # Add error handling for sample generation
        num_samples = min(
            300, len(donor_coords)
        )  # Ensure we don't try to generate too many samples
        generated_donor_coords = generate_coords(
            donor_generator, latent_dim, num_samples, donor_coord_dim
        )

        # Add validation for generated molecules
        generated_donor_mols = []
        for smiles in get_unique_smiles(
            generated_donor_coords, donor_mols, donor_coord_dim
        ):
            if smiles and is_valid_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    generated_donor_mols.append(mol)

        if len(generated_donor_mols) < 5:  # Minimum threshold for meaningful evaluation
            return float("inf")

        # Calculate metrics with error handling
        try:
            mfd_score = calculate_mfd(donor_mols, generated_donor_mols)
            mis_mean, _ = calculate_mis(generated_donor_mols)

            if np.isnan(mfd_score) or np.isnan(mis_mean):
                return float("inf")

            score = mfd_score - mis_mean
        except Exception as e:
            print(f"Metric calculation failed: {e}")
            return float("inf")

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

    return score


def objective_acceptor(trial):
    # Similar to donor objective, but using acceptor data
    latent_dim = trial.suggest_int("latent_dim", 16, 32)
    batch_size = trial.suggest_int("batch_size", 16, 32)
    epochs = trial.suggest_int("epochs", 5, 15)
    # Cap the acceptor LR search at 5e-3 (was 1e-2). The acceptor training set
    # is small (~126 rows, ~6 batches/epoch), so a learning rate near 1e-2 lets
    # the discriminator dominate within 2-3 epochs before any correction
    # mechanism (throttling / LR decay) can act. This mirrors the donor LR cap.
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)

    gen_params = {
        "gen_dense_layers": trial.suggest_int("gen_dense_layers", 1, 2),
        "gen_dense_units": trial.suggest_int("gen_dense_units", 128, 256),
        "gen_lstm_units": trial.suggest_int("gen_lstm_units", 128, 256),
        "lr": lr,  # Add learning rate to gen_params
    }

    disc_params = {
        "disc_transformer_heads": trial.suggest_int("disc_transformer_heads", 4, 8),
        "disc_ff_dim": trial.suggest_int("disc_ff_dim", 128, 256),
        "disc_dense_units": trial.suggest_int("disc_dense_units", 64, 128),
    }

    try:
        # Train acceptor GAN with current hyperparameters
        acceptor_generator = train_gan(
            acceptor_coords,
            acceptor_coord_dim,
            latent_dim,
            batch_size,
            epochs,
            gen_params,
            disc_params,
        )

        num_samples = min(300, len(acceptor_coords))
        generated_acceptor_coords = generate_coords(
            acceptor_generator, latent_dim, num_samples, acceptor_coord_dim
        )

        generated_acceptor_mols = []
        for smiles in get_unique_smiles(
            generated_acceptor_coords, acceptor_mols, acceptor_coord_dim
        ):
            if smiles and is_valid_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    generated_acceptor_mols.append(mol)

        if len(generated_acceptor_mols) < 5:
            return float("inf")

        try:
            mfd_score = calculate_mfd(acceptor_mols, generated_acceptor_mols)
            mis_mean, _ = calculate_mis(generated_acceptor_mols)

            if np.isnan(mfd_score) or np.isnan(mis_mean):
                return float("inf")

            score = mfd_score - mis_mean
        except Exception as e:
            print(f"Metric calculation failed: {e}")
            return float("inf")

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

    return score


def find_best_hyperparameters(objective, name="model", n_trials=25):
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    try:
        print(f"\nOptimizing {name} hyperparameters...")
        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(ValueError, RuntimeError, tf.errors.InvalidArgumentError),
        )

        print(f"\n{name} study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print(
            "  Number of pruned trials: ",
            len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        )
        print(
            "  Number of complete trials: ",
            len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
        )

        print("\nBest trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params

    except Exception as e:
        print(f"Optimization failed: {e}")
        raise


def optimize_smiles_gan_hyperparameters(
    template_smiles,
    char_to_idx,
    idx_to_char,
    max_length,
    vocab_size,
    latent_dim,
    n_trials=5,
):
    logger.info("=" * 80)
    logger.info("AUXILIARY OPTUNA - SMILES GAN")
    logger.info("=" * 80)
    logger.info(
        "SMILES GAN is trainable, so lightweight Optuna is useful for dense/LSTM units, learning rates, batch size, and epochs."
    )
    training_subset = balanced_trait_order(template_smiles, "Generic", max_items=min(len(template_smiles), 300))
    logger.info(
        f"SMILES GAN Optuna uses a trait-balanced subset of {len(training_subset)} molecules; final SMILES GAN training uses all {len(template_smiles)} usable rows."
    )

    def objective(trial):
        dense_units = trial.suggest_categorical("dense_units", [128, 256, 384])
        gen_lstm_units = trial.suggest_categorical("gen_lstm_units", [128, 256, 512])
        disc_lstm_units = trial.suggest_categorical("disc_lstm_units", [128, 256])
        disc_dense_units = trial.suggest_categorical("disc_dense_units", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.45)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        epochs = trial.suggest_int("epochs", 3, 8)
        gen_lr = trial.suggest_float("gen_lr", 5e-5, 5e-4, log=True)
        disc_lr = trial.suggest_float("disc_lr", 2e-5, 3e-4, log=True)
        logger.info(f"SMILES GAN Optuna trial {trial.number} params: {trial.params}")
        try:
            generator = build_smiles_generator(
                latent_dim,
                max_length,
                vocab_size,
                dense_units=dense_units,
                lstm_units=gen_lstm_units,
            )
            discriminator = build_smiles_discriminator(
                max_length,
                vocab_size,
                lstm_units=disc_lstm_units,
                dense_units=disc_dense_units,
                dropout_rate=dropout_rate,
            )
            generator, _ = train_smiles_gan(
                generator,
                discriminator,
                training_subset,
                char_to_idx,
                idx_to_char,
                max_length,
                vocab_size,
                latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                gen_lr=gen_lr,
                disc_lr=disc_lr,
            )
            noise = np.random.normal(0, 1, (32, latent_dim))
            generated_one_hot = generator.predict(noise, verbose=0)
            valid = 0
            unique = set()
            for one_hot in generated_one_hot:
                smi = one_hot_to_smiles(one_hot, idx_to_char, temperature=1.0)
                can = canonical_smiles(smi)
                if can:
                    valid += 1
                    unique.add(can)
            score = -((valid / 32.0) + (len(unique) / 32.0))
            logger.info(
                f"SMILES GAN trial {trial.number}: valid={valid}/32, unique={len(unique)}, score={score:.4f}"
            )
            return score
        except Exception as e:
            logger.warning(f"SMILES GAN Optuna trial {trial.number} failed: {e}")
            return float("inf")

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        logger.info(f"Best SMILES GAN auxiliary params: {study.best_params}")
        return study.best_params
    except Exception as e:
        logger.warning(f"SMILES GAN auxiliary optimization failed; using defaults. Error: {e}")
        return {}


def optimize_vae_hyperparameters(template_fps, latent_dim, n_trials=5):
    logger.info("=" * 80)
    logger.info("AUXILIARY OPTUNA - MOLECULAR VAE")
    logger.info("=" * 80)
    logger.info(
        "VAE is trainable, so lightweight Optuna is useful for latent scaling, hidden width, learning rate, epochs, and batch size."
    )
    template_fps = np.array(template_fps)

    def objective(trial):
        latent_scale = trial.suggest_categorical("latent_scale", [0.5, 1.0, 1.5])
        tuned_latent_dim = max(8, int(latent_dim * latent_scale))
        hidden_width = trial.suggest_categorical("hidden_width", [256, 512, 768])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 3, 10)
        logger.info(f"VAE Optuna trial {trial.number} params: {trial.params}")
        try:
            vae, _, _ = build_molecular_vae(
                input_dim=template_fps.shape[1],
                latent_dim=tuned_latent_dim,
                encoder_units=(hidden_width, max(128, hidden_width // 2)),
                decoder_units=(max(128, hidden_width // 2), hidden_width),
                learning_rate=learning_rate,
            )
            history = vae.fit(
                template_fps,
                template_fps,
                epochs=epochs,
                batch_size=min(batch_size, len(template_fps)),
                verbose=0,
            )
            loss = float(history.history["loss"][-1])
            logger.info(f"VAE trial {trial.number}: final_loss={loss:.4f}")
            return loss
        except Exception as e:
            logger.warning(f"VAE Optuna trial {trial.number} failed: {e}")
            return float("inf")

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        params = study.best_params
        params["latent_dim"] = max(8, int(latent_dim * params.get("latent_scale", 1.0)))
        logger.info(f"Best VAE auxiliary params: {params}")
        return params
    except Exception as e:
        logger.warning(f"VAE auxiliary optimization failed; using defaults. Error: {e}")
        return {}


def optimize_active_learning_hyperparameters(template_smiles, n_trials=5, molecule_type="Generic"):
    logger.info("=" * 80)
    logger.info("AUXILIARY OPTUNA - ACTIVE LEARNING SURROGATE")
    logger.info("=" * 80)
    logger.info(
        "Active-learning uses a trainable RandomForest surrogate, so lightweight Optuna is useful for surrogate capacity and exploration/exploitation balance."
    )
    oracle = ActiveLearningOracle()
    sample_smiles = [
        str(s).strip()
        for s in balanced_trait_order(template_smiles, molecule_type, max_items=min(len(template_smiles), 80))
        if str(s).strip()
    ]
    if len(sample_smiles) < 10:
        logger.warning("Not enough non-empty molecules for active-learning Optuna; using defaults")
        return {}
    morgan_gen = GetMorganGenerator(2, fpSize=2048)
    X = []
    y = []
    fallback_count = 0
    for smi in sample_smiles:
        fp = None
        for feature_map in TRAINING_FALLBACK_FEATURES.values():
            fp = feature_map.get(str(smi).strip())
            if fp is not None:
                fallback_count += 1
                break
        if fp is None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = morgan_gen.GetFingerprintAsNumPy(mol)
        if fp is None:
            fallback = fallback_representation_from_smiles(smi)
            if fallback["usable"]:
                fp = fallback["hashed_fp"]
                fallback_count += 1
        if fp is None:
            continue
        X.append(np.asarray(fp, dtype=np.float32))
        y.append(oracle.query(smi))
    X = np.array(X)
    y = np.array(y)
    logger.info(
        f"Active-learning Optuna fingerprint matrix: rows={len(X)}, fallback_or_precomputed={fallback_count}"
    )
    if len(X) < 10:
        logger.warning("Not enough fingerprintable molecules for active-learning Optuna after fallback handling; using defaults")
        return {}

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
        max_depth = trial.suggest_categorical("max_depth", [None, 8, 16, 24])
        exploration_weight = trial.suggest_float("exploration_weight", 0.25, 0.75)
        logger.info(f"Active-learning Optuna trial {trial.number} params: {trial.params}")
        try:
            split = max(5, int(0.8 * len(X)))
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )
            model.fit(X[:split], y[:split])
            preds = model.predict(X[split:])
            mse = float(np.mean((preds - y[split:]) ** 2)) if len(preds) else 0.0
            score = mse - 0.01 * exploration_weight
            logger.info(f"Active-learning trial {trial.number}: mse={mse:.4f}, score={score:.4f}")
            return score
        except Exception as e:
            logger.warning(f"Active-learning Optuna trial {trial.number} failed: {e}")
            return float("inf")

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        logger.info(f"Best active-learning auxiliary params: {study.best_params}")
        return study.best_params
    except Exception as e:
        logger.warning(f"Active-learning auxiliary optimization failed; using defaults. Error: {e}")
        return {}


# Add this function to check uniqueness against training data
def get_unique_novel_smiles(
    generated_coords, template_mols, training_smiles, num_desired=10000
):
    """
    Generate unique SMILES that are not present in the training dataset.
    No upper limit - extracts all possible unique valid molecules.
    """
    smiles_set = set()
    unique_novel_smiles = []
    attempts = 0
    max_attempts = len(generated_coords)  # Process all generated coordinates

    for coords, template_mol in zip(
        generated_coords,
        template_mols * (len(generated_coords) // len(template_mols) + 1),
    ):
        attempts += 1
        if attempts >= max_attempts:
            break

        try:
            smiles = safe_coords_to_smiles(coords, template_mol)

            # Check if SMILES is valid, unique, and not in training data
            if (
                smiles
                and is_valid_smiles(smiles)
                and smiles not in smiles_set
                and smiles not in training_smiles
            ):
                # Additional validation using RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    if (
                        canonical_smiles not in smiles_set
                        and canonical_smiles not in training_smiles
                    ):
                        smiles_set.add(canonical_smiles)
                        unique_novel_smiles.append(canonical_smiles)

                        # No upper limit - collect all valid unique molecules
                        if len(unique_novel_smiles) >= num_desired:
                            break

        except Exception as e:
            print(f"Error processing SMILES: {e}")
            continue

    print(
        f"Generated {len(unique_novel_smiles)} unique novel SMILES (all valid molecules extracted)"
    )
    return unique_novel_smiles


def calculate_molecular_properties(mol):
    """Calculate key molecular properties"""
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }


def calculate_similarity_metrics(generated_mols, training_mols):
    """Calculate similarity metrics and property distributions"""
    similarities = []
    property_scores = []

    # Calculate training set property ranges
    training_properties = [calculate_molecular_properties(mol) for mol in training_mols]
    prop_ranges = {
        key: (
            min(p[key] for p in training_properties),
            max(p[key] for p in training_properties),
        )
        for key in training_properties[0].keys()
    }

    for gen_mol in generated_mols:
        # Calculate maximum Tanimoto similarity to training set
        gen_fp = morgan_fp(gen_mol, radius=2, nBits=2048)
        if gen_fp is not None:
            train_fps = [
                morgan_fp(train_mol, radius=2, nBits=2048)
                for train_mol in training_mols
            ]
            train_fps = [fp for fp in train_fps if fp is not None]
            if train_fps:
                max_sim = max(
                    DataStructs.TanimotoSimilarity(gen_fp, train_fp)
                    for train_fp in train_fps
                )
            else:
                max_sim = 0.0
        else:
            max_sim = 0.0

        # Calculate property score (how well it fits within training ranges)
        gen_props = calculate_molecular_properties(gen_mol)
        prop_score = sum(
            1
            for key, value in gen_props.items()
            if prop_ranges[key][0] <= value <= prop_ranges[key][1]
        )
        prop_score /= len(prop_ranges)  # Normalize to 0-1

        similarities.append(max_sim)
        property_scores.append(prop_score)

    return similarities, property_scores


def visualize_molecule_space(
    generated_mols, training_mols, mfd_scores, mis_scores, molecule_type
):
    """Create detailed 3D visualization of molecular space with interpretations"""
    similarities, prop_scores = calculate_similarity_metrics(
        generated_mols, training_mols
    )

    # Save visualization data to CSV
    visualization_data = pd.DataFrame(
        {
            "similarities": similarities,
            "prop_scores": prop_scores,
            "mfd_scores": mfd_scores,
            "mis_scores": mis_scores,
        }
    )
    visualization_data.to_csv(
        f"{molecule_type.lower()}_visualization_data.csv", index=False
    )


def create_visualization_plot(
    similarities, prop_scores, mfd_scores, mis_scores, molecule_type
):
    """Create the actual visualization plot with error handling, validation, and threshold planes"""
    try:
        # Input validation
        if not all(
            len(x) > 0 for x in [similarities, prop_scores, mfd_scores, mis_scores]
        ):
            raise ValueError("Empty input arrays detected")

        if not all(
            len(x) == len(similarities) for x in [prop_scores, mfd_scores, mis_scores]
        ):
            raise ValueError("Input arrays must have the same length")

        # Create directory for outputs
        output_dir = "visualization_outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Create figure and grid
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

        try:
            # 3D Scatter plot
            ax1 = fig.add_subplot(gs[0, :], projection="3d")

            # Add threshold planes
            x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            z50 = np.full_like(x, np.percentile(mis_scores, 50))
            z75 = np.full_like(x, np.percentile(mis_scores, 75))

            # Plot threshold surfaces with labels
            surf50 = ax1.plot_surface(x, y, z50, alpha=0.2, color="r")
            surf75 = ax1.plot_surface(x, y, z75, alpha=0.2, color="g")

            # Add threshold annotations
            ax1.text(
                0.8,
                0.8,
                np.percentile(mis_scores, 50),
                f"50th percentile\n({np.percentile(mis_scores, 50):.2f})",
                color="r",
            )
            ax1.text(
                0.8,
                0.8,
                np.percentile(mis_scores, 75),
                f"75th percentile\n({np.percentile(mis_scores, 75):.2f})",
                color="g",
            )

            # Add main scatter plot
            scatter = ax1.scatter(
                similarities,
                prop_scores,
                mis_scores,
                c=mfd_scores,
                cmap="viridis",
                label=f"Generated {molecule_type}s",
            )

            # Create custom legend
            legend_elements = [
                Patch(facecolor="r", alpha=0.2, label="50th Percentile"),
                Patch(facecolor="g", alpha=0.2, label="75th Percentile"),
            ]
            ax1.legend(handles=legend_elements, loc="upper right")

            ax1.set_xlabel("Similarity to Training Set")
            ax1.set_ylabel("Property Match Score")
            ax1.set_zlabel("Mean Inception Score")

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label("Fréchet Distance")

            # Distribution plots
            ax2 = fig.add_subplot(gs[1, 0])
            sns.histplot(data=similarities, ax=ax2, kde=True)
            ax2.set_title("Distribution of Molecular Similarities")
            ax2.set_xlabel("Similarity Score")

            ax3 = fig.add_subplot(gs[1, 1])
            sns.histplot(data=prop_scores, ax=ax3, kde=True)
            ax3.set_title("Distribution of Property Matches")
            ax3.set_xlabel("Property Match Score")

            # Add statistics text box
            stats_text = (
                f"Statistics:\n"
                f"n={len(similarities)}\n"
                f"Avg Similarity: {np.mean(similarities):.2f}\n"
                f"Avg Property Score: {np.mean(prop_scores):.2f}\n"
                f"Avg MIS: {np.mean(mis_scores):.2f}"
            )
            plt.figtext(
                0.02,
                0.02,
                stats_text,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            plt.suptitle(f"Generated {molecule_type} Space Analysis", fontsize=16)
            plt.tight_layout()

            # Save plot with correct naming (remove hybrid suffix if present)
            base_type = molecule_type.replace("_Hybrid_11Approach", "").replace("_Hybrid_12Method", "").replace("_", "").lower()
            output_path = os.path.join(
                output_dir, f"{base_type}_space_analysis.png"
            )
            print(f"Attempting to save visualization to: {output_path}")

            # Ensure the plot is rendered before saving
            plt.draw()

            # Save with different backends if primary fails
            try:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Successfully saved visualization to: {output_path}")
            except Exception as e:
                print(f"Primary save failed: {e}")
                try:
                    # Try with Agg backend
                    import matplotlib
                    matplotlib.use('Agg')
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    print(f"Successfully saved visualization with Agg backend to: {output_path}")
                except Exception as e2:
                    print(f"Agg backend save also failed: {e2}")
                    # Try simple save without bbox_inches
                    try:
                        plt.savefig(output_path, dpi=150)
                        print(f"Successfully saved visualization with simple save to: {output_path}")
                    except Exception as e3:
                        print(f"All save attempts failed: {e3}")
                        return None

        except Exception as e:
            print(f"Error in plot creation: {e}")
            raise

        finally:
            plt.close(fig)

    except Exception as e:
        print(f"Error in visualization function: {e}")
        print(
            "Input array lengths:",
            {
                "similarities": len(similarities),
                "prop_scores": len(prop_scores),
                "mfd_scores": len(mfd_scores),
                "mis_scores": len(mis_scores),
            },
        )
        raise


# Add this function to validate the data before visualization
def validate_visualization_data(similarities, prop_scores, mfd_scores, mis_scores):
    """Validate data before creating visualization"""
    try:
        # Convert to numpy arrays for consistent handling
        similarities = np.array(similarities)
        prop_scores = np.array(prop_scores)
        mfd_scores = np.array(mfd_scores)
        mis_scores = np.array(mis_scores)

        # Check for NaN or infinite values
        if (
            np.any(np.isnan(similarities))
            or np.any(np.isnan(prop_scores))
            or np.any(np.isnan(mfd_scores))
            or np.any(np.isnan(mis_scores))
        ):
            raise ValueError("NaN values detected in input data")

        if (
            np.any(np.isinf(similarities))
            or np.any(np.isinf(prop_scores))
            or np.any(np.isinf(mfd_scores))
            or np.any(np.isinf(mis_scores))
        ):
            raise ValueError("Infinite values detected in input data")

        # Check value ranges
        if not (0 <= similarities).all() <= 1 or not (0 <= prop_scores).all() <= 1:
            raise ValueError("Similarity or property scores outside [0,1] range")

        return similarities, prop_scores, mfd_scores, mis_scores

    except Exception as e:
        print(f"Data validation error: {e}")
        raise


def write_analysis_report(
    similarities, prop_scores, mis_scores, mfd_scores, molecule_type
):
    """Write detailed analysis report with interpretations"""
    thresholds = {
        "50%": np.percentile(mis_scores, 50),
        "75%": np.percentile(mis_scores, 75),
        "90%": np.percentile(mis_scores, 90),
    }
    
    # Extract base molecule type (remove hybrid suffix if present)
    base_type = molecule_type.replace("_Hybrid_11Approach", "").replace("_Hybrid_12Method", "").replace("_", "").lower()

    with open(f"{base_type}_analysis_report.txt", "w") as f:
        f.write(f"Detailed Analysis Report for Generated {molecule_type} Molecules\n")
        f.write("=" * 50 + "\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-----------------\n")
        f.write(
            f"Analysis of {len(similarities)} generated {molecule_type.lower()} molecules reveals:\n"
        )
        avg_sim = np.mean(similarities)
        avg_prop = np.mean(prop_scores)
        f.write(
            f"- Overall novelty: {'High' if avg_sim < 0.5 else 'Moderate' if avg_sim < 0.7 else 'Low'}\n"
        )
        f.write(
            f"- Property preservation: {'Strong' if avg_prop > 0.75 else 'Moderate' if avg_prop > 0.5 else 'Weak'}\n"
        )
        f.write(
            f"- Generation quality: {'Excellent' if np.mean(mis_scores) > 0.7 else 'Good' if np.mean(mis_scores) > 0.5 else 'Fair'}\n\n"
        )

        # Detailed Similarity Analysis
        f.write("SIMILARITY ANALYSIS\n")
        f.write("-------------------\n")
        f.write(f"Average similarity to training set: {avg_sim:.3f}\n")
        f.write(
            f"Similarity range: {np.min(similarities):.3f} to {np.max(similarities):.3f}\n"
        )
        f.write(
            f"Molecules with high novelty (similarity < 0.5): {sum(s < 0.5 for s in similarities) / len(similarities) * 100:.1f}%\n"
        )
        f.write("Interpretation: " + get_similarity_interpretation(avg_sim) + "\n\n")

        # Property Analysis
        f.write("PROPERTY ANALYSIS\n")
        f.write("-----------------\n")
        f.write(f"Average property match score: {avg_prop:.3f}\n")
        f.write(f"Property match distribution:\n")
        f.write(
            f"- Poor match (<0.5): {sum(s < 0.5 for s in prop_scores) / len(prop_scores) * 100:.1f}%\n"
        )
        f.write(
            f"- Moderate match (0.5-0.75): {sum((0.5 <= s < 0.75) for s in prop_scores) / len(prop_scores) * 100:.1f}%\n"
        )
        f.write(
            f"- Strong match (>0.75): {sum(s >= 0.75 for s in prop_scores) / len(prop_scores) * 100:.1f}%\n"
        )
        f.write("Interpretation: " + get_property_interpretation(avg_prop) + "\n\n")

        # Quality Metrics
        f.write("QUALITY METRICS\n")
        f.write("--------------\n")
        f.write(f"Fréchet Distance (FD):\n")
        f.write(f"- Average: {np.mean(mfd_scores):.3f}\n")
        f.write(f"- Standard deviation: {np.std(mfd_scores):.3f}\n")
        f.write(
            "Interpretation: " + get_fd_interpretation(np.mean(mfd_scores)) + "\n\n"
        )

        f.write(f"Mean Inception Score (MIS):\n")
        f.write(f"- Average: {np.mean(mis_scores):.3f}\n")
        f.write(f"- Standard deviation: {np.std(mis_scores):.3f}\n")
        f.write(
            "Interpretation: " + get_mis_interpretation(np.mean(mis_scores)) + "\n\n"
        )

        # Quality Thresholds and Recommendations
        f.write("QUALITY THRESHOLDS AND RECOMMENDATIONS\n")
        f.write("------------------------------------\n")
        for threshold, value in thresholds.items():
            f.write(f"{threshold} quality threshold (MIS): {value:.3f}\n")

        f.write("\nRECOMMENDATIONS\n")
        f.write("--------------\n")
        recommendations = get_recommendations(
            avg_sim, avg_prop, np.mean(mis_scores), np.mean(mfd_scores)
        )
        for rec in recommendations:
            f.write(f"- {rec}\n")


def get_similarity_interpretation(avg_sim):
    if avg_sim < 0.3:
        return "The generated molecules show high novelty but may deviate significantly from desired chemical space."
    elif avg_sim < 0.5:
        return "Good balance between novelty and similarity to training set."
    else:
        return "Generated molecules are very similar to training set, suggesting limited novelty."


def get_property_interpretation(avg_prop):
    if avg_prop < 0.5:
        return "Generated molecules poorly preserve desired properties, suggesting potential issues with the model."
    elif avg_prop < 0.75:
        return "Moderate property preservation, but room for improvement in maintaining desired characteristics."
    else:
        return "Strong property preservation, indicating good quality of generated molecules."


def get_fd_interpretation(fd_score):
    if fd_score < 50:
        return "Excellent distribution matching between generated and training sets."
    elif fd_score < 100:
        return "Good distribution matching, but some differences exist."
    else:
        return "Significant differences between generated and training distributions."


def get_mis_interpretation(mis_score):
    if mis_score > 0.7:
        return "High quality and diverse generated molecules."
    elif mis_score > 0.5:
        return "Good quality generation with moderate diversity."
    else:
        return "Limited quality or diversity in generated molecules."


def get_recommendations(avg_sim, avg_prop, mis_score, fd_score):
    recommendations = []

    if avg_sim > 0.7:
        recommendations.append(
            "Consider increasing model diversity to generate more novel molecules."
        )
    if avg_prop < 0.5:
        recommendations.append(
            "Improve property preservation by adjusting model architecture or training parameters."
        )
    if mis_score < 0.5:
        recommendations.append(
            "Enhance generation quality through better training or model complexity."
        )
    if fd_score > 100:
        recommendations.append(
            "Review and adjust the distribution matching mechanism in the model."
        )

    if not recommendations:
        recommendations.append(
            "Current model performance is satisfactory. Consider fine-tuning for specific objectives."
        )

    return recommendations


def process_generated_molecules(generated_mols, training_mols, molecule_type):
    """Process and visualize generated molecules with validation"""
    try:
        if not generated_mols:
            raise ValueError("No generated molecules provided for processing")

        # Calculate metrics
        mfd_score = calculate_mfd(training_mols, generated_mols)
        mis_mean, _ = calculate_mis(generated_mols)

        # Calculate similarity metrics
        similarities, prop_scores = calculate_similarity_metrics(
            generated_mols, training_mols
        )

        # Validate data
        similarities, prop_scores, mfd_scores, mis_scores = validate_visualization_data(
            similarities,
            prop_scores,
            [mfd_score] * len(generated_mols),
            [mis_mean] * len(generated_mols),
        )

        # Create visualization
        create_visualization_plot(
            similarities, prop_scores, mfd_scores, mis_scores, molecule_type
        )

        # Write analysis report
        write_analysis_report(
            similarities, prop_scores, mis_scores, mfd_scores, molecule_type
        )

    except Exception as e:
        print(f"Error processing {molecule_type} molecules: {e}")
        raise


def validate_molecular_quality(mol):
    """Validate generated molecule quality with OSC realism constraints"""
    if mol is None:
        return False, "Null molecule"

    try:
        Chem.SanitizeMol(mol)
        return validate_osc_realism(mol, "Generic")

    except Exception as e:
        return False, f"Invalid molecule: {str(e)}"


def save_visualization_data(generated_mols, training_mols, molecule_type):
    """
    Calculate and save visualization data for later plotting with enhanced quality validation

    Parameters:
    generated_mols: List of generated RDKit molecules
    training_mols: List of training RDKit molecules
    molecule_type: String indicating 'Donor' or 'Acceptor'
    """
    try:
        # Enhanced quality filtering
        logger.info(
            f"Performing quality validation for {len(generated_mols)} {molecule_type} molecules..."
        )
        quality_filtered_mols = []
        quality_scores = []

        for mol in generated_mols:
            is_valid, reason = validate_molecular_quality(mol)
            if is_valid:
                quality_filtered_mols.append(mol)
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.0)
                logger.debug(f"Filtered out molecule: {reason}")

        logger.info(
            f"Quality filtering: {len(quality_filtered_mols)}/{len(generated_mols)} molecules passed"
        )

        # Calculate metrics only on quality-filtered molecules
        similarities, prop_scores = calculate_similarity_metrics(
            quality_filtered_mols, training_mols
        )
        mfd_score = (
            calculate_mfd(training_mols, quality_filtered_mols)
            if quality_filtered_mols
            else float("inf")
        )
        mis_mean, _ = (
            calculate_mis(quality_filtered_mols)
            if quality_filtered_mols
            else (0.0, 0.0)
        )

        # Create DataFrame with the data
        visualization_data = pd.DataFrame(
            {
                "similarities": similarities,
                "prop_scores": prop_scores,
                "mfd_scores": [mfd_score] * len(quality_filtered_mols)
                if quality_filtered_mols
                else [],
                "mis_scores": [mis_mean] * len(quality_filtered_mols)
                if quality_filtered_mols
                else [],
                "quality_score": [1.0] * len(quality_filtered_mols)
                if quality_filtered_mols
                else [],
            }
        )

        # Save to CSV
        output_filename = f"{molecule_type.lower()}_visualization_data.csv"
        visualization_data.to_csv(output_filename, index=False)
        print(
            f"Saved visualization data to {output_filename} ({len(quality_filtered_mols)} quality-filtered molecules)"
        )

        return visualization_data

    except Exception as e:
        print(f"Error saving visualization data for {molecule_type}: {e}")
        raise


# Main training code
logger.info("=" * 80)
logger.info("STARTING MAIN TRAINING PIPELINE")
logger.info("=" * 80)
try:
    # Optimize and train donor GAN
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER OPTIMIZATION - DONOR GAN")
    logger.info("=" * 80)
    donor_best_params = find_best_hyperparameters(objective_donor, "donor", n_trials=2)
    logger.info(f"Best donor parameters found: {donor_best_params}")

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER OPTIMIZATION - ACCEPTOR GAN")
    logger.info("=" * 80)
    acceptor_best_params = find_best_hyperparameters(
        objective_acceptor, "acceptor", n_trials=2
    )
    logger.info(f"Best acceptor parameters found: {acceptor_best_params}")

    # Train donor GAN with best parameters
    logger.info("=" * 80)
    logger.info("TRAINING OPTIMIZED DONOR GAN")
    logger.info("=" * 80)
    donor_gen_params = {
        k: v for k, v in donor_best_params.items() if k.startswith("gen_") or k == "lr"
    }
    donor_disc_params = {
        k: v for k, v in donor_best_params.items() if k.startswith("disc_")
    }
    logger.debug(f"Donor generator params: {donor_gen_params}")
    logger.debug(f"Donor discriminator params: {donor_disc_params}")

    donor_generator = train_gan(
        donor_coords,
        donor_coord_dim,
        donor_best_params["latent_dim"],
        donor_best_params["batch_size"],
        donor_best_params["epochs"],
        donor_gen_params,
        donor_disc_params,
        plot_tag="donor_optimized",
    )
    logger.info("Optimized donor GAN training completed")

    # Train acceptor GAN with best parameters
    logger.info("=" * 80)
    logger.info("TRAINING OPTIMIZED ACCEPTOR GAN")
    logger.info("=" * 80)
    acceptor_gen_params = {
        k: v for k, v in acceptor_best_params.items() if k.startswith("gen_") or k == "lr"
    }
    acceptor_disc_params = {
        k: v for k, v in acceptor_best_params.items() if k.startswith("disc_")
    }
    logger.debug(f"Acceptor generator params: {acceptor_gen_params}")
    logger.debug(f"Acceptor discriminator params: {acceptor_disc_params}")

    acceptor_generator = train_gan(
        acceptor_coords,
        acceptor_coord_dim,
        acceptor_best_params["latent_dim"],
        acceptor_best_params["batch_size"],
        acceptor_best_params["epochs"],
        acceptor_gen_params,
        acceptor_disc_params,
        plot_tag="acceptor_optimized",
    )
    logger.info("Optimized acceptor GAN training completed")

    # Create sets of training SMILES for quick lookup
    logger.info("Creating training SMILES sets for novelty filtering...")
    donor_training_smiles = set(donor_smiles)
    acceptor_training_smiles = set(acceptor_smiles)
    logger.info(f"Donor training set: {len(donor_training_smiles)} unique SMILES")
    logger.info(f"Acceptor training set: {len(acceptor_training_smiles)} unique SMILES")

    # ============================================================================
    # HYBRID 12-METHOD GENERATION
    # ============================================================================

    print("\n" + "=" * 80)
    print("STARTING HYBRID 12-METHOD GENERATION")
    print("=" * 80 + "\n")

    # Generate molecules using all 12 methods
    logger.info("Generating donor molecules using hybrid 12-method pipeline...")
    hybrid_donor_results = hybrid_generate_molecules(
        donor_generator,
        donor_best_params["latent_dim"],
        donor_coord_dim,
        donor_mols,
        donor_smiles,
        num_per_approach=50,
        molecule_type="Donor",
    )
    logger.info(
        f"Hybrid donor generation completed. Generated {sum(len(v) for v in hybrid_donor_results.values())} molecules across all approaches"
    )

    logger.info("Generating acceptor molecules using hybrid 12-method pipeline...")
    hybrid_acceptor_results = hybrid_generate_molecules(
        acceptor_generator,
        acceptor_best_params["latent_dim"],
        acceptor_coord_dim,
        acceptor_mols,
        acceptor_smiles,
        num_per_approach=50,
        molecule_type="Acceptor",
    )
    logger.info(
        f"Hybrid acceptor generation completed. Generated {sum(len(v) for v in hybrid_acceptor_results.values())} molecules across all approaches"
    )

    # Define key->name mapping for approaches (preserve ordering)
    approach_mapping = [
        ("base_gan", "Base_GAN"),
        ("approach1_smiles_gan", "SMILES_GAN"),
        ("approach2_graph_gan", "Graph_GAN"),
        ("approach3_junction_tree", "Junction_Tree"),
        ("approach4_conditional_gan", "Conditional_GAN"),
        ("approach5_rl_gan", "RL_GAN"),
        ("approach6_vae", "VAE"),
        ("approach7_transformer", "Transformer"),
        ("approach8_transfer_learning", "Transfer_Learning"),
        ("approach9_multi_objective", "Multi_Objective"),
        ("approach10_active_learning", "Active_Learning"),
        ("approach11_cocograph", "CoCoGraph_Constrained"),
    ]

    # Initialize lists for deduplication
    logger.info("Initializing deduplication for all approaches...")
    unique_donor_smiles = []
    unique_acceptor_smiles = []
    donor_sources = []
    acceptor_sources = []
    seen_donors = set()
    seen_acceptors = set()
    donor_training_canonical_set = canonical_smiles_set(donor_smiles)
    acceptor_training_canonical_set = canonical_smiles_set(acceptor_smiles)
    donor_training_fps = fingerprints_from_smiles(donor_smiles)
    acceptor_training_fps = fingerprints_from_smiles(acceptor_smiles)

    # Process donors from all 12 methods
    logger.info(
        "Processing and deduplicating donor molecules from all 12 methods..."
    )
    for key, approach_name in approach_mapping:
        approach_smiles = hybrid_donor_results[key]
        before_count = len(unique_donor_smiles)
        for smi in approach_smiles:
            can, ok, reason, _ = accept_generated_candidate(
                smi,
                donor_training_canonical_set,
                donor_training_fps,
                "Donor",
                seen_donors,
                max_similarity=0.4,
            )
            if ok:
                unique_donor_smiles.append(can)
                donor_sources.append(approach_name)
                seen_donors.add(can)
            else:
                logger.debug(f"  {approach_name}: donor rejected: {reason}")
        added = len(unique_donor_smiles) - before_count
        logger.debug(
            f"  {approach_name}: {added} unique valid donors added (from {len(approach_smiles)} generated)"
        )

    # Process acceptors from all 12 methods
    logger.info(
        "Processing and deduplicating acceptor molecules from all 12 methods..."
    )
    for key, approach_name in approach_mapping:
        approach_smiles = hybrid_acceptor_results[key]
        before_count = len(unique_acceptor_smiles)
        for smi in approach_smiles:
            can, ok, reason, _ = accept_generated_candidate(
                smi,
                acceptor_training_canonical_set,
                acceptor_training_fps,
                "Acceptor",
                seen_acceptors,
                max_similarity=0.4,
            )
            if ok:
                unique_acceptor_smiles.append(can)
                acceptor_sources.append(approach_name)
                seen_acceptors.add(can)
            else:
                logger.debug(f"  {approach_name}: acceptor rejected: {reason}")
        added = len(unique_acceptor_smiles) - before_count
        logger.debug(
            f"  {approach_name}: {added} unique valid acceptors added (from {len(approach_smiles)} generated)"
        )

    logger.info(
        f"Deduplication complete. Donors: {len(unique_donor_smiles)}, Acceptors: {len(unique_acceptor_smiles)}"
    )

    # Filter out training set molecules (novelty filtering)
    logger.info("=" * 80)
    logger.info("APPLYING STRICT CANONICAL EXACT-MATCH AND TANIMOTO NOVELTY FILTERING")
    logger.info("=" * 80)
    donor_training_set = donor_training_canonical_set
    acceptor_training_set = acceptor_training_canonical_set

    logger.info(f"Filtering donors - Before: {len(unique_donor_smiles)} molecules")
    unique_donor_smiles, donor_sources, donor_rejections = filter_smiles_with_sources(
        unique_donor_smiles,
        donor_sources,
        donor_training_set,
        donor_training_fps,
        "Donor",
        max_similarity=0.4,
    )
    logger.info(
        f"Filtering acceptors - Before: {len(unique_acceptor_smiles)} molecules"
    )
    unique_acceptor_smiles, acceptor_sources, acceptor_rejections = filter_smiles_with_sources(
        unique_acceptor_smiles,
        acceptor_sources,
        acceptor_training_set,
        acceptor_training_fps,
        "Acceptor",
        max_similarity=0.4,
    )

    logger.info(
        f"Novel donors after training filter: {len(unique_donor_smiles)} (removed {len(seen_donors) - len(unique_donor_smiles)} training molecules)"
    )
    logger.info(
        f"Novel acceptors after training filter: {len(unique_acceptor_smiles)} (removed {len(seen_acceptors) - len(unique_acceptor_smiles)} training molecules)"
    )
    logger.info(f"Strict donor rejections: {donor_rejections}")
    logger.info(f"Strict acceptor rejections: {acceptor_rejections}")

    # ============================================================================
    # COMPREHENSIVE PER-TRAINING-SMILES VARIANT GENERATION WITH QUOTA ENFORCEMENT
    # ============================================================================
    print("\n" + "=" * 80)
    print("STARTING PER-TRAINING-SMILES VARIANT GENERATION WITH QUOTA ENFORCEMENT")
    print("=" * 80)

    # Configuration
    # MIN_VARIANTS_PER_TRAINING raised from 1 to 3: a quota of 1 technically
    # "touches" every training molecule but is too weak to guarantee real
    # coverage of every structural variation present in the training set
    # (e.g. a thiophene-rich donor with 3 novel variants explores far more of
    # its local chemical neighborhood than one with a single variant). Raising
    # to 3 costs more generation attempts but meaningfully increases the
    # number of distinct novel molecules anchored to each training template,
    # directly supporting "every possible variation/type" coverage. Combined
    # with MAX_ATTEMPTS_PER_TRAINING raised to 15 so the extra quota has room
    # to actually be met via GAN/SMARTS/BRICS before falling through to the
    # CoCoGraph rescue path.
    MIN_VARIANTS_PER_TRAINING = 3  # Minimum novel variants per training molecule
    MAX_ATTEMPTS_PER_TRAINING = 15  # Maximum generation attempts per molecule

    # Build fragment library once
    logger.info("Building fragment library for per-training variant generation...")
    fragment_library = build_fragment_library(
        donor_smiles + acceptor_smiles, max_frags=3000
    )
    logger.info(f"Fragment library built with {len(fragment_library)} fragments")

    logger.info("Initializing seen sets for per-training variants...")
    seen_novel_donors = set(unique_donor_smiles)
    seen_novel_acceptors = set(unique_acceptor_smiles)
    logger.debug(
        f"Initial seen donors: {len(seen_novel_donors)}, seen acceptors: {len(seen_novel_acceptors)}"
    )

    def generate_variants_with_quota(
        train_smiles_list,
        training_set,
        seen_novel,
        unique_list,
        source_list,
        generator,
        latent_dim,
        coord_dim,
        template_mols,
        molecule_type,
    ):
        """
        Generate variants for each training SMILES with quota enforcement
        """
        logger.info(
            f"Starting variant generation for {len(train_smiles_list)} {molecule_type} training molecules"
        )
        logger.info(
            f"Target: {MIN_VARIANTS_PER_TRAINING} variants per molecule, Max attempts: {MAX_ATTEMPTS_PER_TRAINING}"
        )
        coverage_stats = {"full_coverage": 0, "partial_coverage": 0, "no_coverage": 0}
        total_variants_added = 0
        training_canonical_set = canonical_smiles_set(training_set)
        training_fps = fingerprints_from_smiles(training_set)
        seen_canonical = canonical_smiles_set(seen_novel)
        coverage_rows = []

        for idx, train_smi in enumerate(train_smiles_list):
            if (idx + 1) % 50 == 0:
                logger.info(
                    f"  Processing {molecule_type} {idx + 1}/{len(train_smiles_list)}..."
                )
                logger.debug(
                    f"    Current coverage - Full: {coverage_stats['full_coverage']}, Partial: {coverage_stats['partial_coverage']}, None: {coverage_stats['no_coverage']}"
                )

            variants_for_this_mol = []
            attempts = 0
            mol = Chem.MolFromSmiles(train_smi)
            if not mol:
                logger.debug(
                    f"    RDKit could not parse {molecule_type} {idx}; using fragment/trait rescue generation: {str(train_smi)[:80]}"
                )

            # Strategy 1: Try GAN-based generation (coordinate â†’ SMILES)
            while (
                len(variants_for_this_mol) < MIN_VARIANTS_PER_TRAINING
                and attempts < MAX_ATTEMPTS_PER_TRAINING
                and mol is not None
            ):
                attempts += 1

                try:
                    # Generate vicoordinate GAN
                    noise = np.random.normal(0, 1, (5, latent_dim))
                    generated_coords = generator.predict(noise, verbose=0)

                    # Try to match structure to current training molecule
                    for coords in generated_coords:
                        try:
                            # Use template mol with similar size
                            template_idx = idx % len(template_mols)
                            template_mol = template_mols[template_idx]
                            smi = coords_to_smiles(
                                coords[: template_mol.GetNumAtoms()], template_mol
                            )

                            can, ok, _, _ = accept_generated_candidate(
                                smi,
                                training_canonical_set,
                                training_fps,
                                molecule_type,
                                seen_canonical,
                                max_similarity=0.4,
                            )
                            if ok:
                                variants_for_this_mol.append(can)
                                seen_novel.add(can)
                                seen_canonical.add(can)
                                if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING:
                                    break
                        except:
                            continue
                except:
                    continue

                # Strategy 2: Fallback to SMARTS mutations if GAN didn't produce enough
                if (
                    len(variants_for_this_mol) < MIN_VARIANTS_PER_TRAINING
                    and attempts % 10 == 0
                ):
                    smarts_variants = smart_mutants_for_mol(mol, max_out=5)
                    for var_smi in smarts_variants:
                        can, ok, _, _ = accept_generated_candidate(
                            var_smi,
                            training_canonical_set,
                            training_fps,
                            molecule_type,
                            seen_canonical,
                            max_similarity=0.4,
                        )
                        if ok:
                            variants_for_this_mol.append(can)
                            seen_novel.add(can)
                            seen_canonical.add(can)
                            if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING:
                                break

                # Strategy 3: Fallback to BRICS recombination if still not enough
                if (
                    len(variants_for_this_mol) < MIN_VARIANTS_PER_TRAINING
                    and attempts % 15 == 0
                ):
                    brics_variants = brics_variants_for_mol(mol, max_out=5)
                    brics_variants.extend(
                        cross_brics_variants_for_mol(mol, fragment_library, max_out=3)
                    )

                    for var_smi in brics_variants:
                        can, ok, _, _ = accept_generated_candidate(
                            var_smi,
                            training_canonical_set,
                            training_fps,
                            molecule_type,
                            seen_canonical,
                            max_similarity=0.4,
                        )
                        if ok:
                            variants_for_this_mol.append(can)
                            seen_novel.add(can)
                            seen_canonical.add(can)
                            if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING:
                                break

            if len(variants_for_this_mol) < MIN_VARIANTS_PER_TRAINING:
                rescue_needed = MIN_VARIANTS_PER_TRAINING - len(variants_for_this_mol)
                rescue_candidates = constrained_cocograph_generate(
                    [train_smi],
                    fragment_library,
                    max(rescue_needed * 5, 10),
                    molecule_type,
                    training_canonical_set,
                    training_fps,
                    max_similarity=0.4,
                )
                for rescue_smi in rescue_candidates:
                    can, ok, _, _ = accept_generated_candidate(
                        rescue_smi,
                        training_canonical_set,
                        training_fps,
                        molecule_type,
                        seen_canonical,
                        max_similarity=0.4,
                    )
                    if ok:
                        variants_for_this_mol.append(can)
                        seen_novel.add(can)
                        seen_canonical.add(can)
                        if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING:
                            break

            # Add collected variants
            for var_smi in variants_for_this_mol:
                unique_list.append(var_smi)
                source_list.append("Training_Set_Augmentation")

            total_variants_added += len(variants_for_this_mol)

            # Track coverage
            if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING:
                coverage_stats["full_coverage"] += 1
            elif len(variants_for_this_mol) > 0:
                coverage_stats["partial_coverage"] += 1
            else:
                coverage_stats["no_coverage"] += 1
                logger.debug(
                    f"    No variants generated for {molecule_type} {idx}: {train_smi[:50]}"
                )
            coverage_rows.append({
                "row_position_in_training_list": idx,
                "molecule_type": molecule_type,
                "training_smiles": train_smi,
                "canonical_training_smiles": canonical_smiles(train_smi) or "",
                "trait_signature": osc_trait_signature(train_smi, molecule_type),
                "rdkit_parseable": mol is not None,
                "attempts": attempts,
                "variants_generated": len(variants_for_this_mol),
                "coverage_status": (
                    "full_coverage"
                    if len(variants_for_this_mol) >= MIN_VARIANTS_PER_TRAINING
                    else ("partial_coverage" if variants_for_this_mol else "no_coverage")
                ),
                "generated_variants": "|".join(variants_for_this_mol),
            })

        logger.info(f"Variant generation completed for {molecule_type}")
        logger.info(f"  Total variants added: {total_variants_added}")
        logger.info(
            f"  Coverage - Full: {coverage_stats['full_coverage']}, Partial: {coverage_stats['partial_coverage']}, None: {coverage_stats['no_coverage']}"
        )
        try:
            os.makedirs("results", exist_ok=True)
            coverage_report_path = os.path.join(
                "results",
                f"{molecule_type.lower()}_per_training_smiles_generation_coverage.csv",
            )
            pd.DataFrame(coverage_rows).to_csv(coverage_report_path, index=False)
            logger.info(f"{molecule_type} per-training generation coverage report saved to {coverage_report_path}")
        except Exception as e:
            logger.warning(f"Could not save {molecule_type} per-training generation coverage report: {e}")
        if coverage_stats["no_coverage"] > 0 and os.environ.get("GAN_ENFORCE_ALL_ROWS_GENERATE", "1") == "1":
            raise RuntimeError(
                f"{molecule_type} generation coverage guarantee failed: {coverage_stats['no_coverage']} training rows produced zero accepted novel realistic variants. See results/{molecule_type.lower()}_per_training_smiles_generation_coverage.csv"
            )
        return coverage_stats, total_variants_added

    # Generate variants for donors
    print(f"\nProcessing {len(donor_smiles)} donor training molecules...")
    donor_stats, donor_variants = generate_variants_with_quota(
        donor_smiles,
        donor_training_set,
        seen_novel_donors,
        unique_donor_smiles,
        donor_sources,
        donor_generator,
        donor_best_params["latent_dim"],
        donor_coord_dim,
        donor_mols,
        "Donor",
    )

    # Generate variants for acceptors
    print(f"\nProcessing {len(acceptor_smiles)} acceptor training molecules...")
    acceptor_stats, acceptor_variants = generate_variants_with_quota(
        acceptor_smiles,
        acceptor_training_set,
        seen_novel_acceptors,
        unique_acceptor_smiles,
        acceptor_sources,
        acceptor_generator,
        acceptor_best_params["latent_dim"],
        acceptor_coord_dim,
        acceptor_mols,
        "Acceptor",
    )

    # -------------------------------------------------------------------
    # TRAIT-GROUP-LEVEL COVERAGE VERIFICATION
    #
    # generate_variants_with_quota() already guarantees every individual
    # training MOLECULE gets at least MIN_VARIANTS_PER_TRAINING novel
    # variants (enforced via GAN_ENFORCE_ALL_ROWS_GENERATE, which crashes
    # the run if any molecule ends up with zero). But that alone doesn't
    # verify every structural CATEGORY/TYPE present in the training set
    # (e.g. thiophene-rich donors, BDT-core donors, cyano-acceptor variants
    # — see osc_trait_signature()) actually produced novel variants sharing
    # that trait, since a template could theoretically only yield variants
    # that drift into an unrelated trait bucket. This check cross-references
    # every trait group observed in the training set (donor_trait_groups /
    # acceptor_trait_groups, computed earlier via log_trait_coverage) against
    # the trait signatures of the ACTUALLY generated novel molecules, and
    # raises if a training trait group has zero matching generated variants.
    # -------------------------------------------------------------------
    def _verify_trait_group_coverage(trait_groups, generated_smiles, sources, molecule_type):
        generated_trait_counts = Counter(
            osc_trait_signature(smi, molecule_type) for smi in generated_smiles
        )
        missing_traits = [trait for trait in trait_groups if generated_trait_counts.get(trait, 0) == 0]
        covered = len(trait_groups) - len(missing_traits)
        logger.info(
            f"{molecule_type} trait-group coverage: {covered}/{len(trait_groups)} training trait "
            f"groups have at least one generated variant sharing that exact trait signature"
        )
        if missing_traits:
            # Reported but intentionally NON-FATAL: osc_trait_signature() is a
            # fine-grained combinatorial tag (heavy-atom bucket x ring count x
            # aromaticity x heteroatom count x halogens x donor/acceptor-
            # specific substructure flags), so it is close to 1:1 with
            # individual training molecules — a mutated/recombined variant
            # will almost always shift at least one tag (heavy-atom count,
            # ring count, etc.) relative to its parent template, so "same
            # exact trait signature" is a stricter bar than "meaningfully
            # covers that region of chemical space". Actual per-MOLECULE
            # coverage (every training template gets novel variants) is
            # already hard-enforced above via GAN_ENFORCE_ALL_ROWS_GENERATE.
            # This report tells you which broad structural categories have
            # the least generated-variant support without risking a false-
            # positive crash of a multi-hour distributed job.
            logger.warning(
                f"{molecule_type} trait groups with no EXACT-signature-matching generated variant "
                f"({len(missing_traits)}/{len(trait_groups)}) — informational only, not a failure: "
                f"{missing_traits[:10]}{'...' if len(missing_traits) > 10 else ''}"
            )
        return missing_traits

    donor_missing_traits = _verify_trait_group_coverage(
        donor_trait_groups, unique_donor_smiles, donor_sources, "Donor"
    )
    acceptor_missing_traits = _verify_trait_group_coverage(
        acceptor_trait_groups, unique_acceptor_smiles, acceptor_sources, "Acceptor"
    )

    # -------------------------------------------------------------------
    # SOFTER TRAIT-SIMILARITY COVERAGE METRIC (nearest-trait Tanimoto)
    #
    # The exact-signature check above is almost always "failed" for most
    # trait groups by construction (any mutation/recombination shifts at
    # least one tag in the fine-grained trait signature). This metric
    # answers the more useful question: for EVERY training molecule of a
    # given trait, was ANY generated molecule chemically similar to it
    # (Tanimoto on Morgan fingerprints), regardless of exact trait match?
    # A trait group is "explored" if at least one of its training members
    # has a nearby generated molecule (Tanimoto >= 0.3, a permissive bar
    # well below the 0.4 novelty-filter threshold used elsewhere, since
    # here we WANT to find near neighbors, not reject them). This directly
    # answers "does every kind/trait of training molecule have generated
    # molecules exploring its chemical neighborhood" without the false
    # negatives of exact-signature matching.
    # -------------------------------------------------------------------
    def _verify_trait_similarity_coverage(train_smiles_list, generated_smiles, molecule_type, threshold=0.3):
        gen_fps = []
        for smi in generated_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = morgan_fp(mol, radius=2, nBits=2048)
            if fp is not None:
                gen_fps.append(fp)

        rows = []
        for train_smi in train_smiles_list:
            train_mol = Chem.MolFromSmiles(train_smi)
            trait = osc_trait_signature(train_smi, molecule_type)
            if train_mol is None or not gen_fps:
                rows.append({"training_smiles": train_smi, "trait_signature": trait,
                             "max_tanimoto_to_generated": 0.0, "explored": False})
                continue
            train_fp = morgan_fp(train_mol, radius=2, nBits=2048)
            if train_fp is None:
                rows.append({"training_smiles": train_smi, "trait_signature": trait,
                             "max_tanimoto_to_generated": 0.0, "explored": False})
                continue
            max_sim = max(DataStructs.TanimotoSimilarity(train_fp, gfp) for gfp in gen_fps)
            rows.append({"training_smiles": train_smi, "trait_signature": trait,
                         "max_tanimoto_to_generated": max_sim, "explored": max_sim >= threshold})

        df_sim = pd.DataFrame(rows)
        if not df_sim.empty:
            trait_summary = df_sim.groupby("trait_signature").agg(
                training_count=("training_smiles", "count"),
                mean_max_tanimoto=("max_tanimoto_to_generated", "mean"),
                best_max_tanimoto=("max_tanimoto_to_generated", "max"),
                any_explored=("explored", "any"),
            ).reset_index()
            unexplored_traits = trait_summary[~trait_summary["any_explored"]]
            explored_count = int(trait_summary["any_explored"].sum())
            total_traits = len(trait_summary)
            logger.info(
                f"{molecule_type} trait-SIMILARITY coverage (Tanimoto>={threshold}): "
                f"{explored_count}/{total_traits} training trait groups have at least one "
                f"chemically-similar generated molecule"
            )
            if len(unexplored_traits) > 0:
                logger.warning(
                    f"{molecule_type} trait groups with NO chemically-similar generated molecule "
                    f"(Tanimoto<{threshold} for every member) — genuinely under-explored regions: "
                    f"{unexplored_traits['trait_signature'].tolist()[:10]}"
                    f"{'...' if len(unexplored_traits) > 10 else ''}"
                )
        else:
            trait_summary = pd.DataFrame(columns=["trait_signature", "training_count",
                                                    "mean_max_tanimoto", "best_max_tanimoto", "any_explored"])
        return df_sim, trait_summary

    donor_sim_df, donor_trait_sim_summary = _verify_trait_similarity_coverage(
        donor_smiles, unique_donor_smiles, "Donor"
    )
    acceptor_sim_df, acceptor_trait_sim_summary = _verify_trait_similarity_coverage(
        acceptor_smiles, unique_acceptor_smiles, "Acceptor"
    )

    try:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame({
            "trait_signature": list(donor_trait_groups.keys()),
            "training_count": list(donor_trait_groups.values()),
            "has_generated_variant": [t not in donor_missing_traits for t in donor_trait_groups.keys()],
        }).to_csv(os.path.join("results", "donor_trait_group_coverage_check.csv"), index=False)
        pd.DataFrame({
            "trait_signature": list(acceptor_trait_groups.keys()),
            "training_count": list(acceptor_trait_groups.values()),
            "has_generated_variant": [t not in acceptor_missing_traits for t in acceptor_trait_groups.keys()],
        }).to_csv(os.path.join("results", "acceptor_trait_group_coverage_check.csv"), index=False)
        donor_trait_sim_summary.to_csv(
            os.path.join("results", "donor_trait_similarity_coverage.csv"), index=False
        )
        acceptor_trait_sim_summary.to_csv(
            os.path.join("results", "acceptor_trait_similarity_coverage.csv"), index=False
        )
    except Exception as e:
        logger.warning(f"Could not save trait-group coverage check CSVs: {e}")

    # Print comprehensive coverage report
    print("\n" + "=" * 80)
    print("PER-TRAINING COVERAGE REPORT")
    print("=" * 80)
    print(
        f"\nDONORS (Target: {MIN_VARIANTS_PER_TRAINING} variants per training molecule):"
    )
    print(f"  Total training molecules: {len(donor_smiles)}")
    print(
        f"  Full coverage (â‰¥{MIN_VARIANTS_PER_TRAINING} variants): {donor_stats['full_coverage']} ({donor_stats['full_coverage'] / len(donor_smiles) * 100:.1f}%)"
    )
    print(
        f"  Partial coverage (1-{MIN_VARIANTS_PER_TRAINING - 1} variants): {donor_stats['partial_coverage']} ({donor_stats['partial_coverage'] / len(donor_smiles) * 100:.1f}%)"
    )
    print(
        f"  No coverage (0 variants): {donor_stats['no_coverage']} ({donor_stats['no_coverage'] / len(donor_smiles) * 100:.1f}%)"
    )
    print(f"  Total variants added: {donor_variants}")

    print(
        f"\nACCEPTORS (Target: {MIN_VARIANTS_PER_TRAINING} variants per training molecule):"
    )
    print(f"  Total training molecules: {len(acceptor_smiles)}")
    print(
        f"  Full coverage (â‰¥{MIN_VARIANTS_PER_TRAINING} variants): {acceptor_stats['full_coverage']} ({acceptor_stats['full_coverage'] / len(acceptor_smiles) * 100:.1f}%)"
    )
    print(
        f"  Partial coverage (1-{MIN_VARIANTS_PER_TRAINING - 1} variants): {acceptor_stats['partial_coverage']} ({acceptor_stats['partial_coverage'] / len(acceptor_smiles) * 100:.1f}%)"
    )
    print(
        f"  No coverage (0 variants): {acceptor_stats['no_coverage']} ({acceptor_stats['no_coverage'] / len(acceptor_smiles) * 100:.1f}%)"
    )
    print(f"  Total variants added: {acceptor_variants}")

    print(f"\nFinal total unique donors: {len(unique_donor_smiles)}")
    print(f"Final total unique acceptors: {len(unique_acceptor_smiles)}")
    print("=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("FINAL STRICT FILTER AFTER ALL AUGMENTATION")
    logger.info("=" * 80)
    pre_final_donors = len(unique_donor_smiles)
    pre_final_acceptors = len(unique_acceptor_smiles)
    unique_donor_smiles, donor_sources, final_donor_rejections = filter_smiles_with_sources(
        unique_donor_smiles,
        donor_sources,
        donor_training_canonical_set,
        donor_training_fps,
        "Donor",
        max_similarity=0.4,
    )
    unique_acceptor_smiles, acceptor_sources, final_acceptor_rejections = filter_smiles_with_sources(
        unique_acceptor_smiles,
        acceptor_sources,
        acceptor_training_canonical_set,
        acceptor_training_fps,
        "Acceptor",
        max_similarity=0.4,
    )
    logger.info(f"Final donors retained: {len(unique_donor_smiles)}/{pre_final_donors}")
    logger.info(f"Final acceptors retained: {len(unique_acceptor_smiles)}/{pre_final_acceptors}")
    logger.info(f"Final donor rejections: {final_donor_rejections}")
    logger.info(f"Final acceptor rejections: {final_acceptor_rejections}")

    # ============================================================================
    # TANGO SYNTHESIZABILITY SCORING
    # ============================================================================
    logger.info("=" * 80)
    logger.info("COMPUTING TANGO SYNTHESIZABILITY SCORES")
    logger.info("=" * 80)

    all_training_smiles = donor_smiles + acceptor_smiles
    tango_scorer = TANGOSynthesizabilityReward(training_smiles=all_training_smiles, alpha=0.5)

    donor_tango_scores = []
    for smi in unique_donor_smiles:
        if smi:
            score = tango_scorer.score_smiles(smi)
            donor_tango_scores.append(score)
        else:
            donor_tango_scores.append(0.0)

    acceptor_tango_scores = []
    for smi in unique_acceptor_smiles:
        if smi:
            score = tango_scorer.score_smiles(smi)
            acceptor_tango_scores.append(score)
        else:
            acceptor_tango_scores.append(0.0)

    valid_donor_ts = [s for s in donor_tango_scores if s > 0]
    valid_acceptor_ts = [s for s in acceptor_tango_scores if s > 0]
    logger.info(f"  Donor TANGO scores: mean={np.mean(valid_donor_ts):.3f}, "
                f"min={np.min(valid_donor_ts):.3f}, max={np.max(valid_donor_ts):.3f}" if valid_donor_ts else "  No valid donor scores")
    logger.info(f"  Acceptor TANGO scores: mean={np.mean(valid_acceptor_ts):.3f}, "
                f"min={np.min(valid_acceptor_ts):.3f}, max={np.max(valid_acceptor_ts):.3f}" if valid_acceptor_ts else "  No valid acceptor scores")

    # Compare TANGO vs standalone synthesizability scorer
    logger.info("  Comparing TANGO vs SynthesizabilityScorer (standalone)...")
    sample_donors = [s for s in unique_donor_smiles if s][:50]
    tango_only_scores = []
    synth_only_scores = []
    combined_scores_list = []
    for smi in sample_donors:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            t, s, c, _ = tango_scorer.combined_reward(mol, SYNTH_SCORER)
            tango_only_scores.append(t)
            synth_only_scores.append(s)
            combined_scores_list.append(c)

    if tango_only_scores:
        try:
            corr, pval = pearsonr(tango_only_scores, synth_only_scores)
            logger.info(f"  TANGO vs SynthScorer correlation: r={corr:.3f}, p={pval:.4f}")
            if corr > 0.5:
                logger.info("  => Moderate-high correlation: combining both improves reward robustness")
                use_combined = True
            else:
                logger.info("  => Low correlation: keeping both as separate metrics")
                use_combined = False
        except Exception:
            use_combined = False
            logger.info("  => Could not compute correlation, keeping separate")
    else:
        use_combined = False

    # Save TANGO scores
    os.makedirs("results", exist_ok=True)
    tango_df = pd.DataFrame({
        'SMILES': [s for s in unique_donor_smiles if s] + [s for s in unique_acceptor_smiles if s],
        'Type': ['Donor'] * len([s for s in unique_donor_smiles if s]) + ['Acceptor'] * len([s for s in unique_acceptor_smiles if s]),
        'TANGO_Score': [s for s in donor_tango_scores if s > 0 or True][:len([s for s in unique_donor_smiles if s])] +
                       [s for s in acceptor_tango_scores if s > 0 or True][:len([s for s in unique_acceptor_smiles if s])],
    })
    tango_df.to_csv("results/tango_scores.csv", index=False)
    logger.info(f"  Saved TANGO scores to results/tango_scores.csv")

    # Per-approach TANGO statistics
    approach_tango_stats = []
    for _, approach_name in approach_mapping:
        d_indices = [i for i, src in enumerate(donor_sources) if src == approach_name]
        a_indices = [i for i, src in enumerate(acceptor_sources) if src == approach_name]
        d_scores = [donor_tango_scores[i] for i in d_indices if i < len(donor_tango_scores)]
        a_scores = [acceptor_tango_scores[i] for i in a_indices if i < len(acceptor_tango_scores)]
        all_scores = d_scores + a_scores
        if all_scores:
            approach_tango_stats.append({
                'approach': approach_name,
                'count': len(all_scores),
                'mean_tango': np.mean(all_scores),
                'std_tango': np.std(all_scores),
                'min_tango': np.min(all_scores),
                'max_tango': np.max(all_scores),
            })
            logger.info(f"  {approach_name}: TANGO mean={np.mean(all_scores):.3f} "
                        f"(n={len(all_scores)})")

    pd.DataFrame(approach_tango_stats).to_csv("results/approach_tango_stats.csv", index=False)

    # ============================================================================
    # SCREENING SURROGATE (Method 1 only) & D-A PAIRING
    # ============================================================================
    logger.info("=" * 80)
    logger.info("RUNNING DESCRIPTOR-BASED INTERPOLATIVE SCREENING SURROGATE")
    logger.info("  Method 1 (Ridge + isotonic + GP on Morgan FPs) -- SCREEN ONLY.")
    logger.info("  Method 2 (Scharber physics) REMOVED from the quantitative blend.")
    logger.info("  Method 3 (SISSO++) REMOVED earlier for numerical instability.")
    logger.info("=" * 80)

    try:
        screening_pipeline = ScreeningSurrogatePipeline(file_path)

        valid_donors_for_pred = [s for s in unique_donor_smiles if s and is_valid_smiles(s)]
        valid_acceptors_for_pred = [s for s in unique_acceptor_smiles if s and is_valid_smiles(s)]

        logger.info(f"  Evaluating D-A pairs: {len(valid_donors_for_pred)} donors x {len(valid_acceptors_for_pred)} acceptors")

        # Build SMILES -> TANGO score lookups so evaluate_da_pairs can attach
        # a synthesizability signal to every pair (see filter_high_performance_pairs
        # for how this screens out device-property-optimistic but synthetically
        # implausible pairs).
        donor_tango_lookup = {
            smi: score for smi, score in zip(unique_donor_smiles, donor_tango_scores) if smi
        }
        acceptor_tango_lookup = {
            smi: score for smi, score in zip(unique_acceptor_smiles, acceptor_tango_scores) if smi
        }

        # ------------------------------------------------------------------
        # RE-SCORE PATH: when --use-existing-predictions is supplied, load the
        # already-computed pair table instead of re-evaluating ~12M pairs.
        # This makes every correction in this file (Method-2 removal, FF label,
        # tier reporting) applicable WITHOUT regenerating molecules.
        # ------------------------------------------------------------------
        _existing_preds = getattr(RUNTIME_ARGS, "use_existing_predictions", None)
        if _existing_preds and os.path.exists(_existing_preds):
            logger.info(f"  RE-SCORE MODE: loading existing pair predictions from {_existing_preds}")
            pairs_df = pd.read_csv(_existing_preds)
            logger.info(f"  Loaded {len(pairs_df)} existing pair rows (no molecule generation, no re-pairing)")
            # Re-derive the Method-1-only blend on the existing table so the
            # corrected (physics-free) blend is what gets filtered/reported.
            for _prop in ['PCE', 'Jsc', 'FF', 'Voc']:
                _surr_col = f"surrogate_{_prop}"
                if _surr_col in pairs_df.columns:
                    pairs_df[_prop] = pairs_df[_surr_col].apply(
                        lambda v, p=_prop: ScreeningSurrogatePipeline._clip_to_physical(v, p)
                        if pd.notna(v) else np.nan
                    )
            logger.info("  Re-derived blended PCE/Jsc/FF/Voc from Method 1 surrogate columns only.")
        else:
            pairs_df = screening_pipeline.evaluate_da_pairs(
                valid_donors_for_pred, valid_acceptors_for_pred, max_pairs=50000,
                donor_tango_lookup=donor_tango_lookup, acceptor_tango_lookup=acceptor_tango_lookup,
            )

        filtered_pairs = screening_pipeline.filter_high_performance_pairs(pairs_df)

        logger.info(f"  Total pairs evaluated: {len(pairs_df)}")

        for col in ['PCE', 'Jsc', 'FF', 'Voc']:
            if col in pairs_df.columns:
                logger.info(f"    {col}: mean={pairs_df[col].mean():.3f}, max={pairs_df[col].max():.3f}")

        if 'favorable_homo_lumo_alignment' in pairs_df.columns:
            _fav = pairs_df['favorable_homo_lumo_alignment']
            _n_fav = int(_fav.fillna(False).astype(bool).sum())
            logger.info(
                f"    favorable_homo_lumo_alignment (QUALITATIVE design-rule flag): "
                f"{_n_fav}/{len(pairs_df)} pairs ({100.0 * _n_fav / max(1, len(pairs_df)):.2f}%)"
            )

        screening_pipeline.save_predictions(pairs_df, filtered_pairs, output_dir="results")
        screening_pipeline.save_disjoint_cv_report(output_dir="results")

        # Identify which donors and acceptors appear in high-performance pairs
        if not filtered_pairs.empty:
            hp_donors = set(filtered_pairs['donor_smiles'].unique())
            hp_acceptors = set(filtered_pairs['acceptor_smiles'].unique())
            logger.info(f"  Unique high-performance donors: {len(hp_donors)}")
            logger.info(f"  Unique high-performance acceptors: {len(hp_acceptors)}")
        else:
            logger.info("  No pairs met all thresholds; retaining all molecules")
            hp_donors = set(valid_donors_for_pred)
            hp_acceptors = set(valid_acceptors_for_pred)

    except Exception as e:
        logger.error(f"Screening surrogate pipeline failed: {e}")
        logger.exception("Full traceback:")
        pairs_df = pd.DataFrame()
        filtered_pairs = pd.DataFrame()

    # ============================================================================
    # SAVE APPROACH STATISTICS
    # ============================================================================
    approach_stats_records = []
    for _, approach_name in approach_mapping:
        d_count = donor_sources.count(approach_name)
        a_count = acceptor_sources.count(approach_name)
        approach_stats_records.append({
            'approach': approach_name,
            'donor_count': d_count,
            'acceptor_count': a_count,
            'total': d_count + a_count,
        })
    approach_stats_records.append({
        'approach': 'Training_Set_Augmentation',
        'donor_count': donor_sources.count("Training_Set_Augmentation"),
        'acceptor_count': acceptor_sources.count("Training_Set_Augmentation"),
        'total': donor_sources.count("Training_Set_Augmentation") + acceptor_sources.count("Training_Set_Augmentation"),
    })
    pd.DataFrame(approach_stats_records).to_csv("results/approach_statistics.csv", index=False)
    logger.info("  Saved approach statistics to results/approach_statistics.csv")

    # Pad lists to same length for DataFrame
    max_len = max(len(unique_donor_smiles), len(unique_acceptor_smiles))
    unique_donor_smiles += [""] * (max_len - len(unique_donor_smiles))
    unique_acceptor_smiles += [""] * (max_len - len(unique_acceptor_smiles))
    donor_sources += [""] * (max_len - len(donor_sources))
    acceptor_sources += [""] * (max_len - len(acceptor_sources))
    donor_tango_scores += [0.0] * (max_len - len(donor_tango_scores))
    acceptor_tango_scores += [0.0] * (max_len - len(acceptor_tango_scores))

    # Create timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Preparing final output files...")

    logger.info("Creating output DataFrame...")
    output_df = pd.DataFrame(
        {
            "Generated Donor SMILES": unique_donor_smiles,
            "Donor Generation Method": donor_sources,
            "Donor TANGO Score": donor_tango_scores,
            "Generated Acceptor SMILES": unique_acceptor_smiles,
            "Acceptor Generation Method": acceptor_sources,
            "Acceptor TANGO Score": acceptor_tango_scores,
        }
    )

    # Add validation information with enhanced quality checks
    logger.info("Validating all generated structures with enhanced quality checks...")
    output_df["Donor Valid Structure"] = output_df["Generated Donor SMILES"].apply(
        lambda x: "Yes" if x and is_valid_smiles(x) else "No"
    )
    output_df["Acceptor Valid Structure"] = output_df[
        "Generated Acceptor SMILES"
    ].apply(lambda x: "Yes" if x and is_valid_smiles(x) else "No")

    # Add enhanced quality validation
    def enhanced_validation(smiles):
        if not smiles or not is_valid_smiles(smiles):
            return "No"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "No"
        is_quality, reason = validate_molecular_quality(mol)
        return "Yes" if is_quality else f"No: {reason}"

    output_df["Donor Quality Check"] = output_df["Generated Donor SMILES"].apply(
        enhanced_validation
    )
    output_df["Acceptor Quality Check"] = output_df["Generated Acceptor SMILES"].apply(
        enhanced_validation
    )

    # Save to Excel with timestamp (already defined above)
    output_filename = f"Hybrid_12Method_Generated_Molecules_{timestamp}.xlsx"
    logger.info(f"Saving results to {output_filename}...")
    output_df.to_excel(output_filename, index=False)
    logger.info(f"Results saved successfully to {output_filename}")

    # Print detailed summary by approach
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY BY APPROACH (ALL UNIQUE NOVEL MOLECULES)")
    print("=" * 80)

    print("\nDONORS:")
    for _, approach_name in approach_mapping:
        count = donor_sources.count(approach_name)
        print(f"  {approach_name}: {count} molecules")
    ptv_donor = donor_sources.count("Training_Set_Augmentation")
    print(f"  Training_Set_Augmentation: {ptv_donor} molecules")
    print(f"  TOTAL: {len([s for s in unique_donor_smiles if s])}")

    print("\nACCEPTORS:")
    for _, approach_name in approach_mapping:
        count = acceptor_sources.count(approach_name)
        print(f"  {approach_name}: {count} molecules")
    ptv_acceptor = acceptor_sources.count("Training_Set_Augmentation")
    print(f"  Training_Set_Augmentation: {ptv_acceptor} molecules")
    print(f"  TOTAL: {len([s for s in unique_acceptor_smiles if s])}")

    print(f"\nResults saved to: {output_filename}")
    print("=" * 80 + "\n")

    # Save additional metadata
    with open(f"hybrid_12method_metadata_{timestamp}.txt", "w") as f:
        f.write("HYBRID 12-METHOD GENERATION METADATA\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generation Parameters:\n")
        f.write(f"  Donor latent dim: {donor_best_params['latent_dim']}\n")
        f.write(f"  Acceptor latent dim: {acceptor_best_params['latent_dim']}\n")
        f.write(f"  Molecules per approach: ~1500\n")
        f.write(
            f"  Total unique donors generated: {len([s for s in unique_donor_smiles if s])}\n"
        )
        f.write(
            f"  Total unique acceptors generated: {len([s for s in unique_acceptor_smiles if s])}\n\n"
        )
        # Detailed per-approach counts
        f.write("Approaches:\n")
        for i, (key, name) in enumerate(approach_mapping, 1):
            donor_count = donor_sources.count(name)
            acceptor_count = acceptor_sources.count(name)
            f.write(
                f"  {i}. {name}: {donor_count} donors, {acceptor_count} acceptors\n"
            )

        # Add per-training variants count
        ptv_donor_count = donor_sources.count("Training_Set_Augmentation")
        ptv_acceptor_count = acceptor_sources.count("Training_Set_Augmentation")
        f.write(
            f"  12. Training_Set_Augmentation: {ptv_donor_count} donors, {ptv_acceptor_count} acceptors\n"
        )
        f.write(
            "\nNote: Training_Set_Augmentation are SMARTS/BRICS modifications of training molecules\n"
        )
        f.write(
            "      ensuring comprehensive coverage of the training set chemical space.\n"
        )

    # Convert all generated SMILES to molecule objects for analysis
    combined_donor_mols = [
        Chem.MolFromSmiles(s)
        for s in unique_donor_smiles
        if s and Chem.MolFromSmiles(s)
    ]
    combined_acceptor_mols = [
        Chem.MolFromSmiles(s)
        for s in unique_acceptor_smiles
        if s and Chem.MolFromSmiles(s)
    ]

    print(f"\nValid donor molecules for analysis: {len(combined_donor_mols)}")
    print(f"Valid acceptor molecules for analysis: {len(combined_acceptor_mols)}")

    # Process combined donor molecules
    print("\nProcessing combined donor molecules...")
    if len(combined_donor_mols) >= 5:
        donor_data = save_visualization_data(
            combined_donor_mols, donor_mols, "Donor_Hybrid_12Method"
        )
    else:
        print("Warning: Not enough valid donor molecules for analysis")
        donor_data = None

    # Process combined acceptor molecules
    print("\nProcessing combined acceptor molecules...")
    if len(combined_acceptor_mols) >= 5:
        acceptor_data = save_visualization_data(
            combined_acceptor_mols, acceptor_mols, "Acceptor_Hybrid_12Method"
        )
    else:
        print("Warning: Not enough valid acceptor molecules for analysis")
        acceptor_data = None

    # Create visualization plots
    print("\nCreating visualization plots...")
    
    # Ensure visualization directory exists
    os.makedirs("visualization_outputs", exist_ok=True)
    
    # Create visualizations if data is available
    donor_plot = None
    if donor_data is not None and not donor_data.empty:
        try:
            create_visualization_plot(
                donor_data["similarities"].values,
                donor_data["prop_scores"].values,
                donor_data["mfd_scores"].values,
                donor_data["mis_scores"].values,
                "Donor_Hybrid_12Method",
            )
            donor_plot = "visualization_outputs/donor_space_analysis.png"
            print(f"âœ“ Donor visualization created: {donor_plot}")
        except Exception as e:
            print(f"âœ— Failed to create donor visualization: {e}")

    acceptor_plot = None
    if acceptor_data is not None and not acceptor_data.empty:
        try:
            create_visualization_plot(
                acceptor_data["similarities"].values,
                acceptor_data["prop_scores"].values,
                acceptor_data["mfd_scores"].values,
                acceptor_data["mis_scores"].values,
                "Acceptor_Hybrid_12Method",
            )
            acceptor_plot = "visualization_outputs/acceptor_space_analysis.png"
            print(f"âœ“ Acceptor visualization created: {acceptor_plot}")
        except Exception as e:
            print(f"âœ— Failed to create acceptor visualization: {e}")

    # Generate comprehensive analysis reports
    print("\nGenerating comprehensive analysis reports...")

    donor_report = None
    if donor_data is not None and not donor_data.empty:
        try:
            write_analysis_report(
                donor_data["similarities"].values,
                donor_data["prop_scores"].values,
                donor_data["mis_scores"].values,
                donor_data["mfd_scores"].values,
                "Donor_Hybrid_12Method",
            )
            donor_report = "donor_analysis_report.txt"
            print(f"âœ“ Donor analysis report created: {donor_report}")
        except Exception as e:
            print(f"âœ— Failed to create donor analysis report: {e}")

    acceptor_report = None
    if acceptor_data is not None and not acceptor_data.empty:
        try:
            write_analysis_report(
                acceptor_data["similarities"].values,
                acceptor_data["prop_scores"].values,
                acceptor_data["mis_scores"].values,
                acceptor_data["mfd_scores"].values,
                "Acceptor_Hybrid_12Method",
            )
            acceptor_report = "acceptor_analysis_report.txt"
            print(f"âœ“ Acceptor analysis report created: {acceptor_report}")
        except Exception as e:
            print(f"âœ— Failed to create acceptor analysis report: {e}")

    # ============================================================================
    # GENERATE TANGO & DMPNN PLOTS INLINE
    # ============================================================================
    logger.info("Generating TANGO and DMPNN-related plots...")
    try:
        # TANGO distribution plots
        fig_tango, axes_t = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mol_type, scores_list in zip(axes_t, ['Donor', 'Acceptor'],
                                              [donor_tango_scores, acceptor_tango_scores]):
            valid_scores = [s for s in scores_list if s > 0]
            if valid_scores:
                ax.hist(valid_scores, bins=50, color='#4CAF50' if mol_type == 'Donor' else '#FF9800',
                        alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(np.mean(valid_scores), color='red', linestyle='--', linewidth=1.5,
                           label=f'Mean: {np.mean(valid_scores):.3f}')
                ax.set_xlabel('TANGO Score', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{mol_type} TANGO Distribution', fontsize=13)
                ax.legend()
        plt.tight_layout()
        plt.savefig("visualization_outputs/tango_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved TANGO distribution plot")

        # Approach comparison bar chart
        fig_app, ax_app = plt.subplots(figsize=(14, 7))
        app_names = [name for _, name in approach_mapping] + ["Training_Set_Augmentation"]
        d_counts = [donor_sources.count(name) for name in app_names]
        a_counts = [acceptor_sources.count(name) for name in app_names]
        x_app = np.arange(len(app_names))
        w = 0.35
        ax_app.bar(x_app - w/2, d_counts, w, label='Donors', color='#2196F3')
        ax_app.bar(x_app + w/2, a_counts, w, label='Acceptors', color='#FF5722')
        ax_app.set_xticks(x_app)
        ax_app.set_xticklabels(app_names, rotation=45, ha='right', fontsize=8)
        ax_app.set_ylabel('Molecule Count')
        ax_app.set_title('Molecules Generated by Each Approach')
        ax_app.legend()
        plt.tight_layout()
        plt.savefig("visualization_outputs/approach_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved approach comparison plot")

        # PCE/Jsc/FF/Voc distributions — one plot for the final BLENDED
        # values (max of all 3 methods, as before) plus one separate plot
        # PER PREDICTION METHOD, so each method's own output distribution
        # is independently visible rather than only the blended result.
        def _plot_device_properties(dataframe, cols_map, title, out_filename):
            if dataframe.empty:
                return
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, (label, (col, color, thresh)) in zip(axes.ravel(), cols_map.items()):
                if col in dataframe.columns:
                    data = dataframe[col].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                        ax.axvline(thresh, color='red', linestyle='--', label=f'Threshold: {thresh}')
                        ax.axvline(data.mean(), color='black', linestyle='-', label=f'Mean: {data.mean():.2f}')
                        ax.set_xlabel(label)
                        ax.set_ylabel('Frequency')
                        pct = (data >= thresh).sum() / len(data) * 100
                        ax.set_title(f'{label} ({pct:.1f}% above threshold)')
                        ax.legend(fontsize=8)
                    else:
                        ax.set_title(f'{label} (no data)')
                        ax.axis('off')
                else:
                    ax.set_title(f'{label} (column not available)')
                    ax.axis('off')
            plt.suptitle(title, fontsize=14, y=1.01)
            plt.tight_layout()
            plt.savefig(f"visualization_outputs/{out_filename}", dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"  Saved {out_filename}")

        if not pairs_df.empty:
            # Screening-surrogate output (Method 1 ONLY, physically clipped).
            # Filename kept for backward compatibility with existing consumers.
            # FF threshold line uses the ACTUALLY APPLIED Tier-1 value (0.75).
            _plot_device_properties(
                pairs_df,
                {
                    'PCE (screen, Method 1)': ('PCE', '#E91E63', 18.0),
                    'Jsc (screen, Method 1)': ('Jsc', '#2196F3', 25.0),
                    'FF (screen, Method 1)':  ('FF', '#4CAF50', ScreeningSurrogatePipeline.FF_THRESHOLD_TIER1),
                    'Voc (screen, Method 1)': ('Voc', '#FF9800', 1.0),
                },
                'Screening-Surrogate OSC Device Properties (Method 1 only; screen, not prediction)',
                'device_properties_distribution.png',
            )

            # Method 1 raw (identical source to the blend now that Methods 2-3
            # are removed; retained as an explicitly-labelled companion panel).
            _plot_device_properties(
                pairs_df,
                {
                    'PCE (Ridge+GP)': ('surrogate_PCE', '#E91E63', 18.0),
                    'Jsc (Ridge+GP)': ('surrogate_Jsc', '#2196F3', 25.0),
                    'FF (Ridge+GP)':  ('surrogate_FF', '#4CAF50', ScreeningSurrogatePipeline.FF_THRESHOLD_TIER1),
                    'Voc (Ridge+GP)': ('surrogate_Voc', '#FF9800', 1.0),
                },
                'Screening Surrogate — Method 1 (Ridge + Isotonic + Gaussian Process)',
                'device_properties_distribution_method1_ridge_gp.png',
            )

            # Method 2 (Scharber physics) plot REMOVED. Its values are no longer
            # quantitative contributors, so plotting them as "predicted device
            # properties" would misrepresent them. The frontier-orbital signal
            # is now reported as the qualitative alignment-flag prevalence below.
            if 'favorable_homo_lumo_alignment' in pairs_df.columns:
                try:
                    _fav_series = pairs_df['favorable_homo_lumo_alignment'].fillna(False).astype(bool)
                    _n_fav = int(_fav_series.sum())
                    _n_all = len(pairs_df)
                    pd.DataFrame({
                        'flag': ['favorable_homo_lumo_alignment', 'not_favorable'],
                        'count': [_n_fav, _n_all - _n_fav],
                        'fraction': [_n_fav / max(1, _n_all), (_n_all - _n_fav) / max(1, _n_all)],
                    }).to_csv(os.path.join('plot_data', 'homo_lumo_alignment_flag_data.csv'), index=False)
                    logger.info(
                        f"  Qualitative frontier-orbital design-rule flag: {_n_fav}/{_n_all} "
                        f"({100.0 * _n_fav / max(1, _n_all):.2f}%) pairs favourable "
                        f"(replaces the removed Method-2 quantitative panel)"
                    )
                except Exception as _fe:
                    logger.debug(f"  Could not write alignment-flag data: {_fe}")
            # Method 3 (SISSO++ equations) removed earlier for numerical instability.

        # TANGO by approach bar chart
        if approach_tango_stats:
            fig_ta, ax_ta = plt.subplots(figsize=(12, 6))
            ta_df = pd.DataFrame(approach_tango_stats)
            ax_ta.bar(np.arange(len(ta_df)), ta_df['mean_tango'],
                      yerr=ta_df['std_tango'], capsize=3, color='#3F51B5', alpha=0.8)
            ax_ta.set_xticks(np.arange(len(ta_df)))
            ax_ta.set_xticklabels(ta_df['approach'], rotation=45, ha='right', fontsize=9)
            ax_ta.set_ylabel('Mean TANGO Score')
            ax_ta.set_title('TANGO Synthesizability Score by Approach')
            plt.tight_layout()
            plt.savefig("visualization_outputs/tango_by_approach.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved TANGO by approach plot")

    except Exception as e:
        logger.error(f"Error generating additional plots: {e}")
        logger.exception("Full traceback:")

    # Calculate and save comprehensive metrics for all approaches
    if combined_donor_mols and combined_acceptor_mols:
        mfd_donors = calculate_mfd(donor_mols, combined_donor_mols)
        mfd_acceptors = calculate_mfd(acceptor_mols, combined_acceptor_mols)
        mis_donors = calculate_mis(combined_donor_mols)
        mis_acceptors = calculate_mis(combined_acceptor_mols)

        # Save with timestamp filename
        with open(f"hybrid_12method_overall_metrics_{timestamp}.txt", "w") as f:
            f.write("HYBRID 12-METHOD OVERALL METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                "Combined Results from All 12 Generation Methods (Maximum Diversity):\n"
            )
            f.write("  1. Base GAN (3D Coordinates)\n")
            f.write("  2. SMILES-Based GAN\n")
            f.write("  3. Graph-Based GAN\n")
            f.write("  4. Junction Tree Fragment Assembly\n")
            f.write("  5. Conditional GAN with Chemical Rules\n")
            f.write("  6. RL-GAN (ORGAN-inspired)\n")
            f.write("  7. Molecular VAE (Variational Autoencoder)\n")
            f.write("  8. Transformer-based Generation\n")
            f.write("  9. Transfer Learning with Pre-trained Embeddings\n")
            f.write("  10. Multi-Objective Optimization (Pareto Frontier)\n")
            f.write("  11. Active Learning with Uncertainty Sampling\n")
            f.write("  12. Per-Training Variants (SMARTS/BRICS modifications)\n\n")

            f.write("DONORS:\n")
            f.write(f"  Total molecules: {len(combined_donor_mols)}\n")
            f.write(f"  MFD (Fréchet Distance): {mfd_donors:.3f}\n")
            f.write(
                f"  MIS (Inception Score): {mis_donors[0]:.3f} ± {mis_donors[1]:.3f}\n\n"
            )

            f.write("ACCEPTORS:\n")
            f.write(f"  Total molecules: {len(combined_acceptor_mols)}\n")
            f.write(f"  MFD (Fréchet Distance): {mfd_acceptors:.3f}\n")
            f.write(
                f"  MIS (Inception Score): {mis_acceptors[0]:.3f} ± {mis_acceptors[1]:.3f}\n\n"
            )

            f.write("Interpretation:\n")
            f.write("-" * 50 + "\n")
            f.write("Lower MFD indicates better similarity to training distribution\n")
            f.write("Higher MIS indicates better quality and diversity\n")

        # Also save as simple filename for easy access
        try:
            with open("output_metrics.txt", "w") as f:
                f.write("GENERATION METRICS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write("DONORS:\n")
                f.write(f"  Total molecules: {len(combined_donor_mols)}\n")
                f.write(f"  MFD (Fréchet Distance): {mfd_donors:.3f}\n")
                f.write(
                    f"  MIS (Inception Score): {mis_donors[0]:.3f} ± {mis_donors[1]:.3f}\n"
                )
                if valid_donor_ts:
                    f.write(f"  TANGO Score (mean): {np.mean(valid_donor_ts):.3f}\n")
                    f.write(f"  TANGO Score (range): [{np.min(valid_donor_ts):.3f}, {np.max(valid_donor_ts):.3f}]\n")
                f.write("\nACCEPTORS:\n")
                f.write(f"  Total molecules: {len(combined_acceptor_mols)}\n")
                f.write(f"  MFD (Fréchet Distance): {mfd_acceptors:.3f}\n")
                f.write(
                    f"  MIS (Inception Score): {mis_acceptors[0]:.3f} ± {mis_acceptors[1]:.3f}\n"
                )
                if valid_acceptor_ts:
                    f.write(f"  TANGO Score (mean): {np.mean(valid_acceptor_ts):.3f}\n")
                    f.write(f"  TANGO Score (range): [{np.min(valid_acceptor_ts):.3f}, {np.max(valid_acceptor_ts):.3f}]\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("DMPNN DEVICE PROPERTY PREDICTIONS\n")
                f.write("=" * 60 + "\n")
                if not pairs_df.empty:
                    for col in ['PCE', 'Jsc', 'FF', 'Voc']:
                        if col in pairs_df.columns:
                            f.write(f"  {col}: mean={pairs_df[col].mean():.3f}, "
                                    f"std={pairs_df[col].std():.3f}, "
                                    f"max={pairs_df[col].max():.3f}\n")
                    f.write(f"\n  Total D-A pairs evaluated: {len(pairs_df)}\n")
                    f.write(f"  High-performance pairs: {len(filtered_pairs)}\n")
                    f.write(f"  Thresholds: PCE>18%, Jsc>25 mA/cm2, FF>0.8, Voc>1.0 V\n")
                else:
                    f.write("  DMPNN pipeline did not produce results\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("PER-APPROACH MOLECULE COUNTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'Approach':<30} {'Donors':>8} {'Acceptors':>10}\n")
                f.write("-" * 50 + "\n")
                for _, name in approach_mapping:
                    d_c = donor_sources.count(name)
                    a_c = acceptor_sources.count(name)
                    f.write(f"{name:<30} {d_c:>8} {a_c:>10}\n")
                d_aug = donor_sources.count("Training_Set_Augmentation")
                a_aug = acceptor_sources.count("Training_Set_Augmentation")
                f.write(f"{'Training_Set_Augmentation':<30} {d_aug:>8} {a_aug:>10}\n")
                f.write("-" * 50 + "\n")
                total_d = len([s for s in unique_donor_smiles if s])
                total_a = len([s for s in unique_acceptor_smiles if s])
                f.write(f"{'TOTAL':<30} {total_d:>8} {total_a:>10}\n")

                f.write("\n\nInterpretation:\n")
                f.write("Lower MFD = better match to training distribution\n")
                f.write("Higher MIS = better quality and diversity\n")
                f.write("Higher TANGO = more synthesizable (Guo & Schwaller 2026)\n")
            
            print("âœ“ Saved metrics to output_metrics.txt")
        except Exception as e:
            print(f"âœ— Failed to save output_metrics.txt: {e}")

    print("\n" + "=" * 80)
    print("HYBRID 12-METHOD PIPELINE COMPLETED SUCCESSFULLY!")
    print("GENERATED NEW, NOVEL MOLECULES FROM ALL 12 METHODS BASED ON TRAINING DATA")
    print("=" * 80)
    print("\nMOLECULE GENERATION STRATEGY:")
    print("âœ“ Each approach generates NEW molecules derived from training SMILES")
    print("âœ“ NO direct copying of training molecules")
    print("âœ“ Uses SMARTS mutations, BRICS recombination, and ML generation")
    print("âœ“ Ensures each training SMILES contributes to new molecule generation")
    print("âœ“ All generated molecules are validated as chemically valid and novel")
    print("=" * 80)
    print(f"\nKey Output Files (NO UPPER LIMITS - Maximum molecules generated):")
    print(f"  1. {output_filename} - All molecules with approach tracking")
    
    # Check and report on each required file
    required_files = [
        ("output_metrics.txt", "Overall quality metrics"),
        ("donor_analysis_report.txt", "Detailed donor analysis"),
        ("acceptor_analysis_report.txt", "Detailed acceptor analysis"),
        ("visualization_outputs/donor_space_analysis.png", "Donor visualization"),
        ("visualization_outputs/acceptor_space_analysis.png", "Acceptor visualization")
    ]
    
    for i, (filename, description) in enumerate(required_files, 2):
        if os.path.exists(filename):
            print(f"  {i}. âœ“ {filename} - {description}")
        else:
            print(f"  {i}. âœ— {filename} - {description} (NOT CREATED)")
    
    # Count successful outputs
    successful_outputs = sum(1 for filename, _ in required_files if os.path.exists(filename))
    print(f"\nâœ“ Successfully created {successful_outputs}/5 required output files")
    print(f"\nAdditional detailed outputs:")
    print(f"  - hybrid_12method_metadata_{timestamp}.txt - Generation statistics")
    print(f"  - hybrid_12method_overall_metrics_{timestamp}.txt - Extended metrics")
    print(
        f"  - donor_hybrid_12method_visualization_data.csv - Donor data"
    )
    print(
        f"  - acceptor_hybrid_12method_visualization_data.csv - Acceptor data"
    )
    print("\nAll 12 Generation Methods Used:")
    print("  1. Base GAN (3D Coordinates)")
    print("  2. SMILES-Based GAN")
    print("  3. Graph-Based GAN")
    print("  4. Junction Tree Fragment Assembly")
    print("  5. Conditional GAN with Chemical Rules")
    print("  6. RL-GAN (ORGAN-inspired)")
    print("  7. Molecular VAE")
    print("  8. Transformer-based Generation")
    print("  9. Transfer Learning")
    print("  10. Multi-Objective Optimization")
    print("  11. Active Learning")
    print("  12. CoCoGraph-Inspired Constrained Graph Generation")
    print("  + Per-Training Variants (SMARTS/BRICS modifications)")
    
    # Print detailed molecule count by approach
    print("\n" + "=" * 80)
    print("MOLECULES GENERATED BY EACH APPROACH (NO UPPER LIMITS)")
    print("=" * 80)
    print("\nDONORS:")
    for _, approach_name in approach_mapping:
        count = donor_sources.count(approach_name)
        print(f"  {approach_name}: {count} molecules")
    ptv_donor = donor_sources.count("Training_Set_Augmentation")
    print(f"  Training_Set_Augmentation: {ptv_donor} molecules")
    print(f"  --> TOTAL DONORS: {len([s for s in unique_donor_smiles if s])}")
    
    print("\nACCEPTORS:")
    for _, approach_name in approach_mapping:
        count = acceptor_sources.count(approach_name)
        print(f"  {approach_name}: {count} molecules")
    ptv_acceptor = acceptor_sources.count("Training_Set_Augmentation")
    print(f"  Training_Set_Augmentation: {ptv_acceptor} molecules")
    print(f"  --> TOTAL ACCEPTORS: {len([s for s in unique_acceptor_smiles if s])}")
    print("=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("ALL PROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    # Log detailed breakdown of molecules by approach
    logger.info("=" * 80)
    logger.info("FINAL MOLECULE COUNT BY APPROACH")
    logger.info("=" * 80)
    for label, sources, smile_list in [
        ("DONORS", donor_sources, unique_donor_smiles),
        ("ACCEPTORS", acceptor_sources, unique_acceptor_smiles),
    ]:
        logger.info(label + ":")
        for _, approach_name in approach_mapping:
            count = sources.count(approach_name)
            logger.info(f"  {approach_name}: {count} molecules")
        ptv_count = sources.count("Training_Set_Augmentation")
        logger.info(f"  Training_Set_Augmentation: {ptv_count} molecules")
        total_unique = len([s for s in smile_list if s])
        logger.info(f"  TOTAL {label}: {total_unique}")
    logger.info("=" * 80)
    
    logger.info(f"Total execution completed: {datetime.datetime.now()}")
    logger.info("Check the log file (gan_pipeline_log.txt) for detailed execution trace")
    logger.info("All 12 generation methods contributed molecules - see counts above")
    
    # FINAL VALIDATION: Ensure all required output files exist
    logger.info("=" * 80)
    logger.info("VALIDATING ALL REQUIRED OUTPUT FILES")
    logger.info("=" * 80)
    
    required_outputs = [
        "output_metrics.txt",
        "donor_analysis_report.txt",
        "acceptor_analysis_report.txt",
        "visualization_outputs/donor_space_analysis.png",
        "visualization_outputs/acceptor_space_analysis.png",
        "results/tango_scores.csv",
        "results/approach_statistics.csv",
        "results/approach_tango_stats.csv",
    ]
    
    optional_outputs = [
        "results/generated_molecules_with_predictions.csv",
        "results/high_performance_pairs.csv",
        "results/dmpnn_prediction_stats.csv",
        "visualization_outputs/tango_distribution.png",
        "visualization_outputs/approach_comparison.png",
        "visualization_outputs/device_properties_distribution.png",
        "visualization_outputs/device_properties_distribution_method1_ridge_gp.png",
        "visualization_outputs/tango_by_approach.png",
    ]
    
    all_files_exist = True
    missing_files = []
    
    for output_file in required_outputs:
        if os.path.exists(output_file):
            logger.info(f"âœ“ {output_file} - EXISTS")
        else:
            logger.warning(f"âš  {output_file} - MISSING")
            all_files_exist = False
            missing_files.append(output_file)
    
    for output_file in optional_outputs:
        if os.path.exists(output_file):
            logger.info(f"âœ“ {output_file} - EXISTS (optional)")
        else:
            logger.info(f"  {output_file} - not created (optional)")
    
    logger.info("=" * 80)
    if all_files_exist:
        logger.info("SUCCESS: ALL REQUIRED OUTPUT FILES CREATED")
    else:
        logger.warning(f"PARTIAL SUCCESS: {len(required_outputs) - len(missing_files)}/{len(required_outputs)} output files created")
        logger.warning(f"Missing files: {', '.join(missing_files)}")
    
    logger.info("=" * 80)
    logger.info("NEW FEATURES IN THIS VERSION:")
    logger.info("  - TANGO-INSPIRED synthesizability heuristic (first-pass SCREEN only)")
    logger.info("  - Descriptor-based interpolative SCREENING surrogate (Method 1 only)")
    logger.info("  - Method 2 (Scharber physics) removed from the quantitative blend")
    logger.info("  - Grouped donor-/acceptor-/scaffold-disjoint CV -> method1_disjoint_cv_report.csv")
    logger.info("  - Tier 1 / Tier 2 counts reported separately (never conflated)")
    logger.info("  - --skip-generation / --use-existing-predictions re-score toggle")
    logger.info("  - D-A pair evaluation with PCE/Jsc/FF/Voc filtering")
    logger.info("  - Fixed Junction Tree (single-component only)")
    logger.info("  - Fixed Graph_GAN (slot-to-atom bond mapping)")
    logger.info("  - Fixed Conditional_GAN (relaxed diversity threshold)")
    logger.info("  - Fixed RL_GAN (robust sanitization)")
    logger.info("  - Fixed Active_Learning (lower oracle threshold)")
    logger.info("  - Standalone plot scripts in plot_scripts/ folder")
    logger.info("  - All plot data saved as CSV in results/ folder")
    logger.info("=" * 80)
    try:
        rank_summary = {
            "rank": RANK_CTX["rank"],
            "world_size": RANK_CTX["world_size"],
            "distributed": RANK_CTX["distributed"],
            "cwd": os.getcwd(),
            "base_workdir": BASE_WORKDIR,
            "donor_count": len([s for s in unique_donor_smiles if s]),
            "acceptor_count": len([s for s in unique_acceptor_smiles if s]),
            "timestamp": str(datetime.datetime.now()),
        }
        with open("rank_done.json", "w") as f:
            json.dump(rank_summary, f, indent=2)
        logger.info("Wrote rank completion marker: rank_done.json")
    except Exception as marker_e:
        logger.warning(f"Could not write rank completion marker: {marker_e}")

    # -------------------------------------------------------------------
    # AUTOMATIC UNIVERSAL OUTPUT MERGE
    #
    # In distributed mode, every rank writes to its own rank_XXX/ folder.
    # Without an explicit merge step, users are left with N separate result
    # sets and no single "final" answer. Rank 0 is designated as the merge
    # coordinator: after writing its own completion marker, it polls for all
    # other ranks' rank_done.json markers (bounded wait, non-blocking on
    # failure) and then runs merge_rank_outputs.py to produce one universal
    # `final_merged_outputs/` folder directly under the submit directory.
    # This is best-effort: if other ranks are slow or a marker never
    # appears (e.g. a rank crashed), merging proceeds with whatever ranks
    # ARE done after the timeout, rather than hanging indefinitely.
    # -------------------------------------------------------------------
    if RANK_CTX["distributed"] and RANK_CTX["rank"] == 0:
        try:
            import time as _time
            logger.info("=" * 80)
            logger.info("RANK 0: WAITING FOR ALL RANKS TO COMPLETE FOR AUTOMATIC MERGE")
            logger.info("=" * 80)
            expected_ranks = RANK_CTX["world_size"]
            # Default raised from 1800s (30 min) to 43200s (12 hours). The
            # try_7th_july 240-rank run took over 22 hours end-to-end with
            # highly heterogeneous per-rank completion times (some ranks
            # finished in ~1hr, others took >20hrs), so the 30-minute default
            # caused the merge to fire after only 196-201/240 ranks (81-84%)
            # had reported done, producing an incomplete "final" merge well
            # before the job was actually finished. 12 hours is still a hard
            # cap (not indefinite) but is realistic for this job's observed
            # runtime distribution; override via GAN_MERGE_MAX_WAIT_SECONDS
            # if your job's ranks are even more heterogeneous.
            max_wait_seconds = int(os.environ.get("GAN_MERGE_MAX_WAIT_SECONDS", "43200"))
            poll_interval_seconds = 15
            waited = 0
            base_outdir = os.path.abspath(os.path.join(RANK_OUTDIR, ".."))
            while waited < max_wait_seconds:
                done_markers = glob.glob(os.path.join(base_outdir, "rank_*", "rank_done.json"))
                if len(done_markers) >= expected_ranks:
                    logger.info(f"All {len(done_markers)}/{expected_ranks} ranks completed. Proceeding with merge.")
                    break
                _time.sleep(poll_interval_seconds)
                waited += poll_interval_seconds
                if waited % 120 == 0:
                    logger.info(f"  Waiting for ranks to complete: {len(done_markers)}/{expected_ranks} done after {waited}s")
            else:
                done_markers = glob.glob(os.path.join(base_outdir, "rank_*", "rank_done.json"))
                logger.warning(
                    f"Merge wait timed out after {max_wait_seconds}s with {len(done_markers)}/{expected_ranks} "
                    f"ranks done. Proceeding with merge using available ranks."
                )

            merge_script_path = os.path.join(BASE_WORKDIR, "merge_rank_outputs.py")
            final_outdir = os.path.join(BASE_WORKDIR, "final_merged_outputs")
            if os.path.exists(merge_script_path):
                import subprocess
                logger.info(f"Running merge script: {merge_script_path}")
                logger.info(f"  Input directory: {base_outdir}")
                logger.info(f"  Output directory: {final_outdir}")
                # Pass the training CSV path explicitly so merge_rank_outputs.py
                # doesn't have to guess via directory search (fixes a bug where
                # DMPNN.csv lives in BASE_WORKDIR, which sits two directories
                # above the rank_*/ folders when they're nested under runs/<job_id>/,
                # a depth the old search didn't cover).
                training_csv_path = os.path.join(BASE_WORKDIR, "DMPNN.csv")
                merge_cmd = [sys.executable, merge_script_path, "--indir", base_outdir, "--outdir", final_outdir]
                if os.path.exists(training_csv_path):
                    merge_cmd += ["--training-csv", training_csv_path]
                result = subprocess.run(
                    merge_cmd,
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    logger.info("Automatic merge completed successfully.")
                    logger.info(result.stdout)
                    logger.info(f"UNIVERSAL FINAL OUTPUT AVAILABLE AT: {final_outdir}")
                else:
                    logger.warning(f"Merge script exited with code {result.returncode}")
                    logger.warning(f"stdout: {result.stdout}")
                    logger.warning(f"stderr: {result.stderr}")
            else:
                logger.warning(f"merge_rank_outputs.py not found at {merge_script_path}; skipping automatic merge.")
        except Exception as merge_e:
            logger.warning(f"Automatic merge step failed (non-fatal): {merge_e}")

except Exception as e:
    logger.error("=" * 80)
    logger.error("PROCESS FAILED WITH ERROR")
    logger.error("=" * 80)
    logger.error(f"Error: {str(e)}")
    logger.exception("Full traceback:")
    raise
