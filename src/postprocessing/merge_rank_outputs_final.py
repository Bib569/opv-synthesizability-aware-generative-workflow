# -----------------------------------------------------------------------------
# REPOSITORY NOTE (comment added when this repository was assembled; no code changed)
# Pipeline stage 1/2 helper: merges the per-rank outputs of the 240 SLURM ranks.
# Run from the flat run directory built by `sh hpc/setup_run_dir.sh` (see docs/PIPELINE.md);
# the script expects DMPNN.csv / Guzik_data.csv and its sibling scripts in that directory.
# CLI: --indir <rank dirs> --outdir <merged dir> --stage {generation,postgen} [--training-csv DMPNN.csv].
# Inputs are the rank_*/ directories written by GAN_240_final.py / postgen_pipeline.py on the cluster
# (not in the repository); see hpc/sub_140.sh and hpc/sub_postgen_full_140.sh for the exact calls.
# -----------------------------------------------------------------------------
"""
Merge all rank_XXX/ outputs into a single universal final_merged_outputs/ folder.

This produces a folder structure matching what a single rank folder contains:
  - All CSVs from results/ (merged, deduplicated)
  - All visualization PNGs (copied from rank with most molecules)
  - All analysis reports (from rank with most molecules)
  - GAN loss/accuracy CSVs and PNGs (averaged across ranks)
  - Hybrid Excel output (from rank with most molecules)
  - output_metrics.txt, hybrid metadata, etc.
"""

import os
import glob
import json
import shutil
import argparse
import pandas as pd
import numpy as np

# Optional heavy dependencies for TRUE report/plot regeneration from pooled data.
# If unavailable, the merge still completes; only the recompute step is skipped
# and the single-best-rank copies (already done above) remain as a fallback.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from sklearn.decomposition import PCA
    from scipy.linalg import sqrtm
    REGEN_AVAILABLE = True
except ImportError as _imp_e:
    REGEN_AVAILABLE = False
    _REGEN_IMPORT_ERROR = str(_imp_e)


def read_csv_safe(path):
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def concat_csvs(csv_paths):
    dfs = []
    for p in csv_paths:
        df = read_csv_safe(p)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# METRIC RECOMPUTATION — mirrors the exact formulas used by GAN_240.py so that
# regenerated reports/plots are numerically consistent with what a single-rank
# run would have produced, but computed over the FULL merged/pooled molecule
# set across all ranks instead of just one rank's subset.
# =============================================================================

def _morgan_fp(mol, radius=2, nBits=2048):
    try:
        return GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    except Exception:
        return None


def _calculate_mfd(real_mols, generated_mols):
    """Mean Fréchet Distance — identical algorithm to GAN_240.py's calculate_mfd."""
    real_mols = [m for m in real_mols if m is not None]
    generated_mols = [m for m in generated_mols if m is not None]
    if not real_mols or not generated_mols:
        return float("inf")

    real_fps, generated_fps = [], []
    for mol in real_mols:
        fp = _morgan_fp(mol)
        if fp is not None:
            real_fps.append(np.array(fp))
    for mol in generated_mols:
        fp = _morgan_fp(mol)
        if fp is not None:
            generated_fps.append(np.array(fp))

    if not real_fps or not generated_fps:
        return float("inf")

    real_fps_array = np.array(real_fps)
    generated_fps_array = np.array(generated_fps)
    all_fps = np.vstack([real_fps_array, generated_fps_array])

    pca = PCA(n_components=2)
    all_fps_pca = pca.fit_transform(all_fps)
    real_fps_pca = all_fps_pca[: len(real_fps_array)]
    generated_fps_pca = all_fps_pca[len(real_fps_array):]

    mu_real = np.mean(real_fps_pca, axis=0)
    sigma_real = np.cov(real_fps_pca, rowvar=False)
    mu_generated = np.mean(generated_fps_pca, axis=0)
    sigma_generated = np.cov(generated_fps_pca, rowvar=False)

    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return float(fid)


def _calculate_mis(generated_mols, max_pairs_sample=5000):
    """
    Mean Internal Similarity (diversity) — identical algorithm to GAN_240.py's
    calculate_mis. For very large pooled sets (potentially tens of thousands
    of molecules across 240 ranks), full O(n^2) pairwise comparison is
    infeasible; we take a deterministic random sample capped at
    max_pairs_sample molecules before computing pairwise Tanimoto, which
    gives a statistically representative MIS without hanging on merge.
    """
    mols = [m for m in generated_mols if m is not None]
    if len(mols) < 2:
        return 0.0, 0.0

    if len(mols) > max_pairs_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(mols), max_pairs_sample, replace=False)
        mols = [mols[i] for i in idx]

    fps = [f for f in (_morgan_fp(m) for m in mols) if f is not None]
    if len(fps) < 2:
        return 0.0, 0.0

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if not similarities:
        return 0.0, 0.0

    avg_similarity = np.mean(similarities)
    mis_mean = 1.0 - avg_similarity
    mis_std = np.std(similarities)
    return float(mis_mean), float(mis_std)


def _calculate_molecular_properties(mol):
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }


def _calculate_similarity_metrics(generated_mols, training_mols):
    """Identical algorithm to GAN_240.py's calculate_similarity_metrics."""
    similarities, property_scores = [], []
    training_properties = [_calculate_molecular_properties(m) for m in training_mols if m is not None]
    if not training_properties:
        return [0.0] * len(generated_mols), [0.0] * len(generated_mols)

    prop_ranges = {
        key: (
            min(p[key] for p in training_properties),
            max(p[key] for p in training_properties),
        )
        for key in training_properties[0].keys()
    }

    train_fps = [f for f in (_morgan_fp(m) for m in training_mols if m is not None) if f is not None]

    for gen_mol in generated_mols:
        if gen_mol is None:
            similarities.append(0.0)
            property_scores.append(0.0)
            continue
        gen_fp = _morgan_fp(gen_mol)
        if gen_fp is not None and train_fps:
            max_sim = max(DataStructs.TanimotoSimilarity(gen_fp, tf) for tf in train_fps)
        else:
            max_sim = 0.0

        gen_props = _calculate_molecular_properties(gen_mol)
        prop_score = sum(
            1 for key, value in gen_props.items()
            if prop_ranges[key][0] <= value <= prop_ranges[key][1]
        ) / len(prop_ranges)

        similarities.append(max_sim)
        property_scores.append(prop_score)

    return similarities, property_scores


def _get_similarity_interpretation(avg_sim):
    if avg_sim < 0.3:
        return "The generated molecules show high novelty but may deviate significantly from desired chemical space."
    elif avg_sim < 0.5:
        return "Good balance between novelty and similarity to training set."
    else:
        return "Generated molecules are very similar to training set, suggesting limited novelty."


def _get_property_interpretation(avg_prop):
    if avg_prop < 0.5:
        return "Generated molecules poorly preserve desired properties, suggesting potential issues with the model."
    elif avg_prop < 0.75:
        return "Moderate property preservation, but room for improvement in maintaining desired characteristics."
    else:
        return "Strong property preservation, indicating good quality of generated molecules."


def _get_fd_interpretation(fd_score):
    if fd_score < 50:
        return "Excellent distribution matching between generated and training sets."
    elif fd_score < 100:
        return "Good distribution matching, but some differences exist."
    else:
        return "Significant differences between generated and training distributions."


def _get_mis_interpretation(mis_score):
    if mis_score > 0.7:
        return "High quality and diverse generated molecules."
    elif mis_score > 0.5:
        return "Good quality generation with moderate diversity."
    else:
        return "Limited quality or diversity in generated molecules."


def _get_recommendations(avg_sim, avg_prop, mis_score, fd_score):
    recommendations = []
    if avg_sim > 0.7:
        recommendations.append("Consider increasing model diversity to generate more novel molecules.")
    if avg_prop < 0.5:
        recommendations.append("Improve property preservation by adjusting model architecture or training parameters.")
    if mis_score < 0.5:
        recommendations.append("Enhance generation quality through better training or model complexity.")
    if fd_score > 100:
        recommendations.append("Review and adjust the distribution matching mechanism in the model.")
    if not recommendations:
        recommendations.append("Current model performance is satisfactory. Consider fine-tuning for specific objectives.")
    return recommendations


def _write_analysis_report(outdir, similarities, prop_scores, mis_scores, mfd_scores, molecule_type):
    """Identical report format to GAN_240.py's write_analysis_report, written
    into the merged outdir instead of the working directory of a single rank."""
    base_type = molecule_type.replace("_Hybrid_11Approach", "").replace("_Hybrid_12Method", "").replace("_", "").lower()
    path = os.path.join(outdir, f"{base_type}_analysis_report.txt")
    with open(path, "w") as f:
        f.write(f"Detailed Analysis Report for Generated {molecule_type} Molecules\n")
        f.write("=" * 50 + "\n\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-----------------\n")
        f.write(f"Analysis of {len(similarities)} generated {molecule_type.lower()} molecules reveals:\n")
        avg_sim = np.mean(similarities)
        avg_prop = np.mean(prop_scores)
        f.write(f"- Overall novelty: {'High' if avg_sim < 0.5 else 'Moderate' if avg_sim < 0.7 else 'Low'}\n")
        f.write(f"- Property preservation: {'Strong' if avg_prop > 0.75 else 'Moderate' if avg_prop > 0.5 else 'Weak'}\n")
        f.write(f"- Generation quality: {'Excellent' if np.mean(mis_scores) > 0.7 else 'Good' if np.mean(mis_scores) > 0.5 else 'Fair'}\n\n")

        f.write("SIMILARITY ANALYSIS\n")
        f.write("-------------------\n")
        f.write(f"Average similarity to training set: {avg_sim:.3f}\n")
        f.write(f"Similarity range: {np.min(similarities):.3f} to {np.max(similarities):.3f}\n")
        f.write(f"Molecules with high novelty (similarity < 0.5): {sum(s < 0.5 for s in similarities) / len(similarities) * 100:.1f}%\n")
        f.write("Interpretation: " + _get_similarity_interpretation(avg_sim) + "\n\n")

        f.write("PROPERTY ANALYSIS\n")
        f.write("-----------------\n")
        f.write(f"Average property match score: {avg_prop:.3f}\n")
        f.write("Property match distribution:\n")
        f.write(f"- Poor match (<0.5): {sum(s < 0.5 for s in prop_scores) / len(prop_scores) * 100:.1f}%\n")
        f.write(f"- Moderate match (0.5-0.75): {sum((0.5 <= s < 0.75) for s in prop_scores) / len(prop_scores) * 100:.1f}%\n")
        f.write(f"- Strong match (>0.75): {sum(s >= 0.75 for s in prop_scores) / len(prop_scores) * 100:.1f}%\n")
        f.write("Interpretation: " + _get_property_interpretation(avg_prop) + "\n\n")

        f.write("QUALITY METRICS\n")
        f.write("--------------\n")
        f.write("Fréchet Distance (FD):\n")
        f.write(f"- Average: {np.mean(mfd_scores):.3f}\n")
        f.write(f"- Standard deviation: {np.std(mfd_scores):.3f}\n")
        f.write("Interpretation: " + _get_fd_interpretation(np.mean(mfd_scores)) + "\n\n")

        f.write("Mean Inception Score (MIS):\n")
        f.write(f"- Average: {np.mean(mis_scores):.3f}\n")
        f.write(f"- Standard deviation: {np.std(mis_scores):.3f}\n")
        f.write("Interpretation: " + _get_mis_interpretation(np.mean(mis_scores)) + "\n\n")

        f.write("QUALITY THRESHOLDS AND RECOMMENDATIONS\n")
        f.write("------------------------------------\n")
        for pct, label in [(50, "50%"), (75, "75%"), (90, "90%")]:
            f.write(f"{label} quality threshold (MIS): {np.percentile(mis_scores, pct):.3f}\n")

        f.write("\nRECOMMENDATIONS\n")
        f.write("--------------\n")
        for rec in _get_recommendations(avg_sim, avg_prop, np.mean(mis_scores), np.mean(mfd_scores)):
            f.write(f"- {rec}\n")
    return path


def regenerate_reports_and_plots(outdir, best_rank, indir, training_csv_override=None):
    """
    Recompute MFD/MIS/similarity/TANGO metrics over the FULL pooled molecule
    set (all ranks combined) and regenerate output_metrics.txt,
    donor/acceptor_analysis_report.txt, and the four visualization PNGs so
    they reflect the true combined-ensemble result rather than one rank's
    subset. Falls back silently (leaving the best-rank copies already in
    place) if rdkit/matplotlib/sklearn/scipy are unavailable, or if the
    training CSV / SMILES columns cannot be found.
    """
    if not REGEN_AVAILABLE:
        print(f"  [WARN] Report/plot regeneration skipped: missing dependency ({_REGEN_IMPORT_ERROR}). "
              f"Best-rank copies are used instead.")
        return

    merged_preds_path = os.path.join(outdir, "results", "generated_molecules_with_predictions.csv")
    merged_preds = read_csv_safe(merged_preds_path)
    if merged_preds is None or merged_preds.empty:
        print("  [WARN] No merged predictions found; skipping report/plot regeneration.")
        return

    # Locate the original training CSV (DMPNN.csv or Training_dataset_GAN.csv).
    #
    # Bug fixed: the previous search only checked `indir` (e.g.
    # runs/16312, the directory containing rank_*/ folders) and its
    # immediate parent (runs/). But GAN_240.py writes DMPNN.csv to
    # BASE_WORKDIR (the submit directory, e.g. try_7th_july/), which sits
    # ONE LEVEL ABOVE `runs/` — i.e. `indir/../..`, not covered by the old
    # 2-directory search. This silently disabled pooled report/plot
    # regeneration in the real try_7th_july run ("[WARN] Could not locate
    # original training CSV"), leaving output_metrics.txt etc. as best-rank
    # copies only. Fixed by: (1) accepting an explicit --training-csv
    # override, (2) also searching the directory this merge script itself
    # lives in (BASE_WORKDIR, since GAN_240.py invokes
    # `sys.executable merge_rank_outputs.py` from BASE_WORKDIR), and (3)
    # searching indir/../.. in addition to indir and indir/...
    training_csv = None
    if training_csv_override and os.path.exists(training_csv_override):
        training_csv = os.path.abspath(training_csv_override)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            indir,
            os.path.abspath(os.path.join(indir, "..")),
            os.path.abspath(os.path.join(indir, "..", "..")),
            script_dir,
        ]
        for candidate_dir in search_dirs:
            for pattern in ["DMPNN.csv", "Training_dataset_GAN.csv", "*.csv"]:
                matches = glob.glob(os.path.join(candidate_dir, pattern))
                matches = [m for m in matches if "generated_molecules" not in m and "high_performance" not in m]
                if matches:
                    training_csv = matches[0]
                    break
            if training_csv:
                break

    if training_csv is None:
        print("  [WARN] Could not locate original training CSV in indir/parent/grandparent/script-dir; "
              "skipping report/plot regeneration. Pass --training-csv explicitly to fix this.")
        return

    print(f"  Regenerating reports/plots from pooled data (training ref: {training_csv})")

    train_df = read_csv_safe(training_csv)
    donor_col = next((c for c in ["donor(SMILES)"] if c in train_df.columns), None)
    acceptor_col = next((c for c in ["acceptor(SMILES)"] if c in train_df.columns), None)
    if donor_col is None or acceptor_col is None:
        print("  [WARN] Training CSV missing donor(SMILES)/acceptor(SMILES) columns; skipping regeneration.")
        return

    training_donor_smiles = train_df[donor_col].dropna().astype(str).tolist()
    training_acceptor_smiles = train_df[acceptor_col].dropna().astype(str).tolist()
    training_donor_mols = [Chem.MolFromSmiles(s) for s in training_donor_smiles]
    training_donor_mols = [m for m in training_donor_mols if m is not None]
    training_acceptor_mols = [Chem.MolFromSmiles(s) for s in training_acceptor_smiles]
    training_acceptor_mols = [m for m in training_acceptor_mols if m is not None]

    pooled_donor_smiles = sorted(set(merged_preds["donor_smiles"].dropna().astype(str)))
    pooled_acceptor_smiles = sorted(set(merged_preds["acceptor_smiles"].dropna().astype(str)))
    pooled_donor_mols = [m for m in (Chem.MolFromSmiles(s) for s in pooled_donor_smiles) if m is not None]
    pooled_acceptor_mols = [m for m in (Chem.MolFromSmiles(s) for s in pooled_acceptor_smiles) if m is not None]

    print(f"    Pooled unique donors: {len(pooled_donor_mols)}, acceptors: {len(pooled_acceptor_mols)}")

    if not pooled_donor_mols or not pooled_acceptor_mols or not training_donor_mols or not training_acceptor_mols:
        print("  [WARN] Insufficient valid molecules for MFD/MIS recomputation; skipping.")
        return

    # --- Recompute MFD/MIS over the full pooled set ---
    mfd_donors = _calculate_mfd(training_donor_mols, pooled_donor_mols)
    mfd_acceptors = _calculate_mfd(training_acceptor_mols, pooled_acceptor_mols)
    mis_donors = _calculate_mis(pooled_donor_mols)
    mis_acceptors = _calculate_mis(pooled_acceptor_mols)
    print(f"    Pooled MFD donors={mfd_donors:.3f} acceptors={mfd_acceptors:.3f}")
    print(f"    Pooled MIS donors={mis_donors[0]:.3f}±{mis_donors[1]:.3f} acceptors={mis_acceptors[0]:.3f}±{mis_acceptors[1]:.3f}")

    # --- Recompute similarity/property scores for analysis reports ---
    donor_sims, donor_props = _calculate_similarity_metrics(pooled_donor_mols, training_donor_mols)
    acceptor_sims, acceptor_props = _calculate_similarity_metrics(pooled_acceptor_mols, training_acceptor_mols)
    donor_mfd_list = [mfd_donors] * len(pooled_donor_mols)
    donor_mis_list = [mis_donors[0]] * len(pooled_donor_mols)
    acceptor_mfd_list = [mfd_acceptors] * len(pooled_acceptor_mols)
    acceptor_mis_list = [mis_acceptors[0]] * len(pooled_acceptor_mols)

    _write_analysis_report(outdir, donor_sims, donor_props, donor_mis_list, donor_mfd_list, "Donor_Hybrid_12Method")
    _write_analysis_report(outdir, acceptor_sims, acceptor_props, acceptor_mis_list, acceptor_mfd_list, "Acceptor_Hybrid_12Method")
    print("  Wrote donor_analysis_report.txt and acceptor_analysis_report.txt (pooled)")

    # Overwrite the visualization_data CSVs (originally copied from the best
    # rank) with the true pooled similarity/property/MFD/MIS data so anyone
    # inspecting these files sees the ensemble result, not one rank's subset.
    pd.DataFrame({
        "similarities": donor_sims,
        "prop_scores": donor_props,
        "mfd_scores": donor_mfd_list,
        "mis_scores": donor_mis_list,
        "quality_score": [1.0] * len(donor_sims),
    }).to_csv(os.path.join(outdir, "donor_hybrid_12method_visualization_data.csv"), index=False)
    pd.DataFrame({
        "similarities": acceptor_sims,
        "prop_scores": acceptor_props,
        "mfd_scores": acceptor_mfd_list,
        "mis_scores": acceptor_mis_list,
        "quality_score": [1.0] * len(acceptor_sims),
    }).to_csv(os.path.join(outdir, "acceptor_hybrid_12method_visualization_data.csv"), index=False)
    print("  Overwrote donor/acceptor_hybrid_12method_visualization_data.csv with pooled data")

    # --- TANGO scores from merged tango_scores.csv (already pooled) ---
    merged_tango = read_csv_safe(os.path.join(outdir, "results", "tango_scores.csv"))
    donor_tango_scores, acceptor_tango_scores = [], []
    if merged_tango is not None and not merged_tango.empty and "Type" in merged_tango.columns:
        donor_tango_scores = merged_tango[merged_tango["Type"] == "Donor"]["TANGO_Score"].dropna().tolist()
        acceptor_tango_scores = merged_tango[merged_tango["Type"] == "Acceptor"]["TANGO_Score"].dropna().tolist()

    # --- Recompute high-performance / DMPNN stats over the pooled predictions ---
    pairs_df = merged_preds
    filtered_pairs = read_csv_safe(os.path.join(outdir, "results", "high_performance_pairs.csv"))
    if filtered_pairs is None:
        filtered_pairs = pd.DataFrame()

    # --- Write pooled output_metrics.txt ---
    with open(os.path.join(outdir, "output_metrics.txt"), "w") as f:
        f.write("GENERATION METRICS SUMMARY (POOLED ACROSS ALL RANKS)\n")
        f.write("=" * 60 + "\n\n")
        f.write("DONORS:\n")
        f.write(f"  Total unique molecules (pooled): {len(pooled_donor_mols)}\n")
        f.write(f"  MFD (Fréchet Distance): {mfd_donors:.3f}\n")
        f.write(f"  MIS (Inception Score): {mis_donors[0]:.3f} ± {mis_donors[1]:.3f}\n")
        if donor_tango_scores:
            f.write(f"  TANGO Score (mean): {np.mean(donor_tango_scores):.3f}\n")
            f.write(f"  TANGO Score (range): [{np.min(donor_tango_scores):.3f}, {np.max(donor_tango_scores):.3f}]\n")

        f.write("\nACCEPTORS:\n")
        f.write(f"  Total unique molecules (pooled): {len(pooled_acceptor_mols)}\n")
        f.write(f"  MFD (Fréchet Distance): {mfd_acceptors:.3f}\n")
        f.write(f"  MIS (Inception Score): {mis_acceptors[0]:.3f} ± {mis_acceptors[1]:.3f}\n")
        if acceptor_tango_scores:
            f.write(f"  TANGO Score (mean): {np.mean(acceptor_tango_scores):.3f}\n")
            f.write(f"  TANGO Score (range): [{np.min(acceptor_tango_scores):.3f}, {np.max(acceptor_tango_scores):.3f}]\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("DMPNN DEVICE PROPERTY PREDICTIONS (POOLED)\n")
        f.write("=" * 60 + "\n")
        if not pairs_df.empty:
            f.write("\n-- FINAL BLENDED (max of Methods 1-2, physically clipped) --\n")
            for col in ["PCE", "Jsc", "FF", "Voc", "pair_tango", "donor_tango", "acceptor_tango"]:
                if col in pairs_df.columns:
                    f.write(f"  {col}: mean={pairs_df[col].mean():.3f}, "
                            f"std={pairs_df[col].std():.3f}, max={pairs_df[col].max():.3f}\n")

            # Per-method breakdown, if the individual method columns are
            # present (they are whenever GAN_240.py's evaluate_da_pairs()
            # ran the Method 1/2 pipeline, i.e. every normal run). A third
            # method (SISSO++ equations) was implemented and evaluated in an
            # earlier iteration but was REMOVED after verification showed
            # numerically unstable outputs (division-by-near-zero blowups
            # reaching Jsc values of ±400,000+ and physically-impossible
            # negative Voc) on genuinely novel generated molecules outside
            # its narrow 300-row training range. The blended values were
            # unaffected (hard PHYSICAL_BOUNDS clipping absorbed the
            # blowups, and SISSO's in-range values never won the max()
            # against Methods 1/2), so only Methods 1-2 remain here.
            method_cols = {
                "Method 1 (Ridge + Gaussian Process surrogate)": [
                    "surrogate_PCE", "surrogate_Jsc", "surrogate_FF", "surrogate_Voc"],
                "Method 2 (Scharber-model physics estimate)": [
                    "phys_physics_PCE", "phys_physics_Jsc", "phys_physics_FF", "phys_physics_Voc"],
            }
            for method_label, cols in method_cols.items():
                present = [c for c in cols if c in pairs_df.columns and pairs_df[c].notna().any()]
                if not present:
                    continue
                f.write(f"\n-- {method_label} --\n")
                for col in present:
                    data = pairs_df[col].dropna()
                    f.write(f"  {col}: mean={data.mean():.3f}, std={data.std():.3f}, "
                            f"min={data.min():.3f}, max={data.max():.3f}\n")

            f.write(f"\n  Total D-A pairs evaluated (pooled, deduplicated): {len(pairs_df)}\n")
            f.write(f"  High-performance pairs (pooled): {len(filtered_pairs)}\n")
            if "pair_tango" in pairs_df.columns:
                f.write("  Thresholds: PCE>18%, Jsc>25 mA/cm2, FF>0.8, Voc>1.0 V, "
                        "pair_tango(synthesizability)>=0.5 [Tier1] / >=0.35 [Tier2]\n")
                f.write("  (pair_tango = min(donor_tango, acceptor_tango); a pair is only as\n"
                        "   synthesizable as its harder-to-make partner, so 'high-performance'\n"
                        "   here means both good predicted device properties AND a plausible\n"
                        "   synthesis route for both molecules — not device properties alone.)\n")
            else:
                f.write("  Thresholds: PCE>18%, Jsc>25 mA/cm2, FF>0.8, Voc>1.0 V\n")
                f.write("  [No pair_tango column found in this run's pooled data — "
                        "synthesizability gate was not applied.]\n")
        else:
            f.write("  No pooled predictions available\n")

        approach_stats = read_csv_safe(os.path.join(outdir, "results", "approach_statistics.csv"))
        if approach_stats is not None and not approach_stats.empty:
            f.write("\n" + "=" * 60 + "\n")
            f.write("PER-APPROACH MOLECULE COUNTS (SUMMED ACROSS ALL RANKS)\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Approach':<30} {'Donors':>8} {'Acceptors':>10}\n")
            f.write("-" * 50 + "\n")
            for _, row in approach_stats.iterrows():
                f.write(f"{str(row['approach']):<30} {int(row.get('donor_count', 0)):>8} {int(row.get('acceptor_count', 0)):>10}\n")
            f.write("-" * 50 + "\n")
            total_d = int(approach_stats.get('donor_count', pd.Series([0])).sum())
            total_a = int(approach_stats.get('acceptor_count', pd.Series([0])).sum())
            f.write(f"{'TOTAL':<30} {total_d:>8} {total_a:>10}\n")

        f.write("\n\nInterpretation:\n")
        f.write("Lower MFD = better match to training distribution\n")
        f.write("Higher MIS = better quality and diversity\n")
        f.write("Higher TANGO = more synthesizable (Guo & Schwaller 2026)\n")
        f.write(f"\nNOTE: All statistics above are computed on the POOLED/MERGED dataset\n")
        f.write(f"across {len(glob.glob(os.path.join(indir, 'rank_*')))} ranks, not a single rank's subset.\n")
    print("  Wrote output_metrics.txt (pooled)")

    # --- Regenerate the four visualization PNGs from pooled data ---
    # Alongside each PNG, save the exact underlying data to plot_data/ as a
    # CSV so any plot can be manually recreated (different bins, colors,
    # styling, etc.) with a standalone plotting script, without needing to
    # re-run this merge or the original GAN pipeline.
    viz_dir = os.path.join(outdir, "visualization_outputs")
    os.makedirs(viz_dir, exist_ok=True)
    plot_data_dir = os.path.join(outdir, "plot_data")
    os.makedirs(plot_data_dir, exist_ok=True)

    try:
        # Save raw TANGO scores used for the histogram (one row per molecule)
        tango_plot_df = pd.concat([
            pd.DataFrame({"TANGO_Score": donor_tango_scores, "Type": "Donor"}),
            pd.DataFrame({"TANGO_Score": acceptor_tango_scores, "Type": "Acceptor"}),
        ], ignore_index=True)
        tango_plot_df.to_csv(os.path.join(plot_data_dir, "tango_distribution_data.csv"), index=False)

        fig_tango, axes_t = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mol_type, scores_list in zip(axes_t, ["Donor", "Acceptor"],
                                              [donor_tango_scores, acceptor_tango_scores]):
            valid_scores = [s for s in scores_list if s > 0]
            if valid_scores:
                ax.hist(valid_scores, bins=50, color="#4CAF50" if mol_type == "Donor" else "#FF9800",
                        alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.axvline(np.mean(valid_scores), color="red", linestyle="--", linewidth=1.5,
                           label=f"Mean: {np.mean(valid_scores):.3f}")
                ax.set_xlabel("TANGO Score", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"{mol_type} TANGO Distribution (Pooled)", fontsize=13)
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "tango_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_tango)
        print("  Regenerated tango_distribution.png (pooled) + saved tango_distribution_data.csv")
    except Exception as e:
        print(f"  [WARN] Failed to regenerate tango_distribution.png: {e}")

    try:
        approach_stats = read_csv_safe(os.path.join(outdir, "results", "approach_statistics.csv"))
        if approach_stats is not None and not approach_stats.empty:
            # The plot data IS approach_statistics.csv, but save a copy under
            # plot_data/ too so all plot inputs live in one consistent place.
            approach_stats.to_csv(os.path.join(plot_data_dir, "approach_comparison_data.csv"), index=False)

            fig_app, ax_app = plt.subplots(figsize=(14, 7))
            x_app = np.arange(len(approach_stats))
            w = 0.35
            ax_app.bar(x_app - w / 2, approach_stats["donor_count"], w, label="Donors", color="#2196F3")
            ax_app.bar(x_app + w / 2, approach_stats["acceptor_count"], w, label="Acceptors", color="#FF5722")
            ax_app.set_xticks(x_app)
            ax_app.set_xticklabels(approach_stats["approach"], rotation=45, ha="right", fontsize=8)
            ax_app.set_ylabel("Molecule Count")
            ax_app.set_title("Molecules Generated by Each Approach (Summed Across All Ranks)")
            ax_app.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "approach_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close(fig_app)
            print("  Regenerated approach_comparison.png (pooled) + saved approach_comparison_data.csv")
    except Exception as e:
        print(f"  [WARN] Failed to regenerate approach_comparison.png: {e}")

    def _regen_device_properties_plot(cols_map, title, out_png_name, out_csv_name):
        """Regenerate one device-properties 4-panel histogram (blended, or
        one of Method 1/2/3) from the pooled pairs_df, saving both the PNG
        and its underlying plot-source CSV. Mirrors the identical helper in
        GAN_240.py so per-rank and pooled outputs stay visually consistent."""
        if pairs_df.empty:
            return
        cols_present = [col for (_, (col, _, _)) in cols_map.items() if col in pairs_df.columns]
        if cols_present:
            pairs_df[cols_present].to_csv(os.path.join(plot_data_dir, out_csv_name), index=False)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (label, (col, color, thresh)) in zip(axes.ravel(), cols_map.items()):
            if col in pairs_df.columns:
                data = pairs_df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
                    ax.axvline(thresh, color="red", linestyle="--", label=f"Threshold: {thresh}")
                    ax.axvline(data.mean(), color="black", linestyle="-", label=f"Mean: {data.mean():.2f}")
                    ax.set_xlabel(label)
                    ax.set_ylabel("Frequency")
                    pct = (data >= thresh).sum() / len(data) * 100
                    ax.set_title(f"{label} ({pct:.1f}% above threshold, pooled)")
                    ax.legend(fontsize=8)
                else:
                    ax.set_title(f"{label} (no data)")
                    ax.axis("off")
            else:
                ax.set_title(f"{label} (column not available)")
                ax.axis("off")
        plt.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, out_png_name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Regenerated {out_png_name} (pooled) + saved {out_csv_name}")

    try:
        # Final blended (max of Methods 1-2, physically clipped). A third
        # method (SISSO++ equations) was implemented and evaluated in an
        # earlier iteration but was REMOVED after verification showed
        # numerically unstable outputs on genuinely novel generated
        # molecules; see the note in GAN_240.py's evaluate_da_pairs() for
        # full details. Only the two verified-stable methods remain.
        _regen_device_properties_plot(
            {
                "PCE (blended, final)": ("PCE", "#E91E63", 18.0),
                "Jsc (blended, final)": ("Jsc", "#2196F3", 25.0),
                "FF (blended, final)":  ("FF", "#4CAF50", 0.8),
                "Voc (blended, final)": ("Voc", "#FF9800", 1.0),
            },
            "Predicted OSC Device Properties — Final Blended (Pooled, max of Methods 1-2)",
            "device_properties_distribution.png",
            "device_properties_distribution_data.csv",
        )
        # Method 1: Ridge + Gaussian Process surrogate
        _regen_device_properties_plot(
            {
                "PCE (Ridge+GP)": ("surrogate_PCE", "#E91E63", 18.0),
                "Jsc (Ridge+GP)": ("surrogate_Jsc", "#2196F3", 25.0),
                "FF (Ridge+GP)":  ("surrogate_FF", "#4CAF50", 0.8),
                "Voc (Ridge+GP)": ("surrogate_Voc", "#FF9800", 1.0),
            },
            "Predicted OSC Device Properties — Method 1 (Ridge + Gaussian Process, Pooled)",
            "device_properties_distribution_method1_ridge_gp.png",
            "device_properties_distribution_method1_ridge_gp_data.csv",
        )
        # Method 2: Scharber-model physics estimate
        _regen_device_properties_plot(
            {
                "PCE (Physics/Scharber)": ("phys_physics_PCE", "#E91E63", 18.0),
                "Jsc (Physics/Scharber)": ("phys_physics_Jsc", "#2196F3", 25.0),
                "FF (Physics/Scharber)":  ("phys_physics_FF", "#4CAF50", 0.8),
                "Voc (Physics/Scharber)": ("phys_physics_Voc", "#FF9800", 1.0),
            },
            "Predicted OSC Device Properties — Method 2 (Scharber-model Physics, Pooled)",
            "device_properties_distribution_method2_physics.png",
            "device_properties_distribution_method2_physics_data.csv",
        )
    except Exception as e:
        print(f"  [WARN] Failed to regenerate device_properties_distribution PNGs: {e}")

    try:
        approach_tango = read_csv_safe(os.path.join(outdir, "results", "approach_tango_stats.csv"))
        if approach_tango is not None and not approach_tango.empty:
            approach_tango.to_csv(os.path.join(plot_data_dir, "tango_by_approach_data.csv"), index=False)

            fig_ta, ax_ta = plt.subplots(figsize=(12, 6))
            ax_ta.bar(np.arange(len(approach_tango)), approach_tango["mean_tango"],
                      yerr=approach_tango.get("std_tango"), capsize=3, color="#3F51B5", alpha=0.8)
            ax_ta.set_xticks(np.arange(len(approach_tango)))
            ax_ta.set_xticklabels(approach_tango["approach"], rotation=45, ha="right", fontsize=9)
            ax_ta.set_ylabel("Mean TANGO Score")
            ax_ta.set_title("TANGO Synthesizability Score by Approach (Pooled)")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "tango_by_approach.png"), dpi=150, bbox_inches="tight")
            plt.close(fig_ta)
            print("  Regenerated tango_by_approach.png (pooled) + saved tango_by_approach_data.csv")
    except Exception as e:
        print(f"  [WARN] Failed to regenerate tango_by_approach.png: {e}")

    # --- Save data for the GAN loss/accuracy plots (already averaged CSVs
    # in outdir; copy references into plot_data/ for a single consistent
    # location containing all plot-source data) ---
    for gan_csv in [
        "gan_losses_donor_optimized.csv",
        "gan_losses_acceptor_optimized.csv",
        "discriminator_accuracy_donor_optimized.csv",
        "discriminator_accuracy_acceptor_optimized.csv",
    ]:
        src = os.path.join(outdir, gan_csv)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(plot_data_dir, gan_csv))
    print(f"  Saved all plot-source CSVs to {plot_data_dir}")


# =============================================================================
# PER-APPROACH AND AUGMENTATION-EXCLUDED POOLING
#
# WHY THIS EXISTS: Training_Set_Augmentation contributes ~215k donors / ~213k
# acceptors summed across 240 ranks, versus ~11.7k-12k for each of the other 12
# approaches -- roughly an 18x imbalance. That is not a bug: the 12 generative
# approaches share a fixed per-rank budget (num_per_approach=50), whereas
# Training_Set_Augmentation carries a COVERAGE GUARANTEE of >= 3 novel variants
# for every one of the ~300 training molecules, on every rank, unsharded
# (~300 x 3 = ~900/rank). It is working as designed.
#
# The consequence for REPORTING, though, is real: because the augmentation
# molecules dominate the deduplicated pool, a single pooled diversity/distance
# number is disproportionately characterised by BRICS/SMARTS near-neighbours of
# the training set rather than by the exploratory GAN/VAE/Transformer/RL output.
#
# Fix implemented here (no molecule regeneration required, and no data thrown
# away): report BOTH
#   (a) pooled over ALL approaches            -- the honest full-pool number
#   (b) pooled EXCLUDING Training_Set_Augmentation -- the exploratory-only view
# plus (c) a per-approach breakdown of MFD / IntDiv / TANGO, extending the
# pattern that already existed for TANGO alone.
# =============================================================================

AUGMENTATION_APPROACH_NAME = "Training_Set_Augmentation"


def _collect_per_molecule_sources(rank_dirs, merged_dirs=None):
    """Build {side: {smiles: approach}} from source-labelled CSVs.

    Looks for the per-rank files that carry a per-molecule approach label, and
    ALSO in any already-merged directories passed in `merged_dirs` -- long-
    finished runs frequently have their rank_* directories cleaned up while the
    merged copies survive, and without this fallback the per-approach breakdown
    would be silently unavailable in exactly that (common) situation.

    Falls back to an empty mapping (per-approach breakdown then simply skipped,
    with a warning) rather than guessing an attribution.
    """
    out = {"Donor": {}, "Acceptor": {}}
    candidates = [
        ("Donor", "donor_generation_sources.csv"),
        ("Acceptor", "acceptor_generation_sources.csv"),
        ("Donor", "donor_hybrid_12method_sources.csv"),
        ("Acceptor", "acceptor_hybrid_12method_sources.csv"),
    ]
    search_roots = list(rank_dirs or [])
    for md in (merged_dirs or []):
        if md and os.path.isdir(md):
            search_roots.append(md)
    for side, fname in candidates:
        paths = [os.path.join(r, "results", fname) for r in search_roots]
        paths += [os.path.join(r, fname) for r in search_roots]
        paths += [os.path.join(r, "plot_data", fname) for r in search_roots]
        df = concat_csvs(paths)
        if df is None or df.empty:
            continue
        smi_col = next((c for c in ("smiles", "SMILES", "generated_smiles") if c in df.columns), None)
        src_col = next((c for c in ("source", "approach", "Source", "Approach") if c in df.columns), None)
        if smi_col and src_col:
            for smi, src in zip(df[smi_col].astype(str), df[src_col].astype(str)):
                out[side].setdefault(smi, src)
    return out


def compute_pooled_and_per_approach_stats(outdir, results_dir, training_donor_mols,
                                          training_acceptor_mols, rank_dirs):
    """Compute per-approach + pooled(all) + pooled(excl. augmentation) statistics.

    Writes approach_full_stats.csv and returns a dict of pooled summaries for
    inclusion in output_metrics.txt.
    """
    if not REGEN_AVAILABLE:
        print("  [WARN] Per-approach pooling skipped: missing dependency.")
        return {}

    merged_preds = read_csv_safe(os.path.join(results_dir, "generated_molecules_with_predictions.csv"))
    if merged_preds is None or merged_preds.empty:
        print("  [WARN] No merged predictions; per-approach pooling skipped.")
        return {}

    tango = read_csv_safe(os.path.join(results_dir, "tango_scores.csv"))
    tango_lookup = {}
    if tango is not None and not tango.empty and "SMILES" in tango.columns:
        for smi, sc in zip(tango["SMILES"].astype(str),
                           pd.to_numeric(tango.get("TANGO_Score"), errors="coerce")):
            if pd.notna(sc):
                tango_lookup[smi] = float(sc)

    sources = _collect_per_molecule_sources(rank_dirs, merged_dirs=[outdir, results_dir])
    have_sources = bool(sources["Donor"]) or bool(sources["Acceptor"])
    if not have_sources:
        print("  [WARN] No per-molecule source labels found in rank outputs; the per-approach "
              "MFD/IntDiv breakdown cannot be attributed and is skipped. Pooled(all) and "
              "pooled(excluding augmentation) still require source labels, so those are skipped too.")

    rows = []
    summary = {}

    for side, smi_col, train_mols in (("Donor", "donor_smiles", training_donor_mols),
                                      ("Acceptor", "acceptor_smiles", training_acceptor_mols)):
        if smi_col not in merged_preds.columns:
            continue
        pooled_all = sorted(set(merged_preds[smi_col].dropna().astype(str)))
        side_sources = sources.get(side, {})

        def _stats(label, smiles_list):
            mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles_list) if m is not None]
            if len(mols) < 2:
                return None
            mfd = _calculate_mfd(train_mols, mols)
            intdiv, intdiv_std = _calculate_mis(mols)
            tvals = [tango_lookup[s] for s in smiles_list if s in tango_lookup]
            return {
                "side": side, "approach": label,
                "n_unique_molecules": len(smiles_list),
                "n_valid_molecules": len(mols),
                "fp_pca_frechet_distance": mfd,
                "internal_diversity": intdiv,
                "internal_diversity_std": intdiv_std,
                "mean_pairwise_tanimoto": (1.0 - intdiv) if intdiv else np.nan,
                "tango_inspired_mean": float(np.mean(tvals)) if tvals else np.nan,
                "tango_inspired_std": float(np.std(tvals)) if tvals else np.nan,
                "tango_n": len(tvals),
            }

        r = _stats("POOLED_ALL_APPROACHES", pooled_all)
        if r:
            rows.append(r)
            summary["%s_pooled_all" % side] = r

        if side_sources:
            by_approach = {}
            for s in pooled_all:
                by_approach.setdefault(side_sources.get(s, "UNLABELLED"), []).append(s)
            for approach, smis in sorted(by_approach.items()):
                r = _stats(approach, smis)
                if r:
                    rows.append(r)
            excl = [s for s in pooled_all
                    if side_sources.get(s, "UNLABELLED") != AUGMENTATION_APPROACH_NAME]
            if excl:
                r = _stats("POOLED_EXCLUDING_%s" % AUGMENTATION_APPROACH_NAME, excl)
                if r:
                    rows.append(r)
                    summary["%s_pooled_excl_aug" % side] = r

    if rows:
        df = pd.DataFrame(rows)
        p = os.path.join(results_dir, "approach_full_stats.csv")
        df.to_csv(p, index=False)
        print("  Wrote %s (%d rows: per-approach + pooled(all) + pooled(excl. augmentation))"
              % (p, len(df)))
        plot_dir = os.path.join(outdir, "plot_data")
        os.makedirs(plot_dir, exist_ok=True)
        df.to_csv(os.path.join(plot_dir, "approach_full_stats_data.csv"), index=False)
    return summary


def write_tier_breakdown(results_dir, outdir):
    """Write an explicit Tier-1 / Tier-2 breakdown (never a conflated total).

    The previously reported single '25.1% high-performance' figure combined a
    STRICT Tier 1 and a much looser Tier 2, so it could not be read as "fraction
    meeting the strict criteria". Both tiers are now reported separately, with
    their own thresholds spelled out and the FF threshold stated as the value the
    code actually applies (0.75), not the nominal 0.8.
    """
    hp = read_csv_safe(os.path.join(results_dir, "high_performance_pairs.csv"))
    allp = read_csv_safe(os.path.join(results_dir, "generated_molecules_with_predictions.csv"))
    if hp is None or hp.empty:
        return {}
    n_all = len(allp) if allp is not None else 0
    rec = {"n_pairs_evaluated": n_all, "n_high_performance_total": len(hp)}
    if "tier" in hp.columns:
        counts = hp["tier"].astype(str).value_counts().to_dict()
        n_t1 = int(sum(v for k, v in counts.items() if "Tier1" in k))
        n_t2 = int(sum(v for k, v in counts.items() if "Tier2" in k))
        rec.update({
            "n_tier1_strict": n_t1, "n_tier2_relaxed": n_t2,
            "pct_tier1_of_all_pairs": (100.0 * n_t1 / n_all) if n_all else np.nan,
            "pct_tier2_of_all_pairs": (100.0 * n_t2 / n_all) if n_all else np.nan,
            "pct_tier1_plus_tier2_of_all_pairs": (100.0 * (n_t1 + n_t2) / n_all) if n_all else np.nan,
            "tier1_thresholds": "PCE>=18%, Jsc>=25 mA/cm2, FF>=0.75 (value actually applied; "
                                "nominal literature value 0.8), Voc>=1.0 V, pair_tango>=0.5",
            "tier2_thresholds": "PCE>=14%, Jsc>=18 mA/cm2, FF>=0.60, Voc>=0.85 V, pair_tango>=0.35",
        })
    df = pd.DataFrame([rec]).T.rename(columns={0: "value"})
    p = os.path.join(results_dir, "tier_breakdown.csv")
    df.to_csv(p)
    print("  Wrote %s (Tier1 and Tier2 reported separately)" % p)
    plot_dir = os.path.join(outdir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)
    df.to_csv(os.path.join(plot_dir, "tier_breakdown_data.csv"))
    return rec


# =============================================================================
# POSTGEN STAGE MERGE
# =============================================================================

POSTGEN_SHARD_FILES = [
    ("donor_molecule_predictions.csv", ["smiles"]),
    ("acceptor_molecule_predictions.csv", ["smiles"]),
    ("pair_predictions.csv", ["donor_smiles", "acceptor_smiles"]),
    ("retrosynthesis_audit.csv", ["smiles"]),
    ("generative_metrics.csv", ["set"]),
]


def merge_postgen_stage(indir, outdir):
    """Merge postgen_pipeline.py rank shards and write output_metrics_postgen.txt."""
    print("=" * 78)
    print("MERGE STAGE: POSTGEN")
    print("=" * 78)
    os.makedirs(outdir, exist_ok=True)
    results_dir = os.path.join(outdir, "results_postgen")
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(outdir, "plot_data")
    os.makedirs(plot_dir, exist_ok=True)

    rank_dirs = sorted(glob.glob(os.path.join(indir, "rank_*")))
    if not rank_dirs:
        raise RuntimeError("No rank_* directories found in %s" % indir)
    print("Found %d rank directories in %s" % (len(rank_dirs), indir))

    merged_tables = {}
    for fname, dedup in POSTGEN_SHARD_FILES:
        paths = [os.path.join(r, fname) for r in rank_dirs]
        df = concat_csvs(paths)
        if not df.empty:
            keys = [k for k in dedup if k in df.columns]
            if keys:
                df = df.drop_duplicates(subset=keys, keep="first")
            else:
                df = df.drop_duplicates(keep="first")
        out_path = os.path.join(results_dir, fname)
        df.to_csv(out_path, index=False)
        df.to_csv(os.path.join(plot_dir, fname.replace(".csv", "_data.csv")), index=False)
        merged_tables[fname] = df
        print("  Merged %-40s : %d rows" % (fname, len(df)))

    # Root-level artefacts produced once by rank 0.
    for fname in ("model_selection_horse_race.csv", "property_model_cv_report.csv",
                  "stage_a_dataset_diagnostics.csv", "rl_finetune_candidates.csv"):
        src = os.path.join(indir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(results_dir, fname))
            shutil.copy2(src, os.path.join(plot_dir, fname))
            print("  Copied %-40s" % fname)

    write_postgen_metrics_report(outdir, results_dir, merged_tables)
    print("=" * 78)
    print("POSTGEN MERGE COMPLETE -> %s" % outdir)
    print("=" * 78)
    return 0


def write_postgen_metrics_report(outdir, results_dir, tables):
    """Write output_metrics_postgen.txt: the honest, correctly-named metric report."""
    path = os.path.join(outdir, "output_metrics_postgen.txt")
    gm = tables.get("generative_metrics.csv", pd.DataFrame())
    pair = tables.get("pair_predictions.csv", pd.DataFrame())
    don = tables.get("donor_molecule_predictions.csv", pd.DataFrame())
    acc = tables.get("acceptor_molecule_predictions.csv", pd.DataFrame())
    audit = tables.get("retrosynthesis_audit.csv", pd.DataFrame())
    race = read_csv_safe(os.path.join(results_dir, "property_model_cv_report.csv"))

    with open(path, "w", encoding="utf-8") as f:
        f.write("POST-GENERATION METRICS REPORT (POOLED ACROSS ALL RANKS)\n")
        f.write("=" * 78 + "\n\n")
        f.write("METRIC NAMING (corrections applied, stated explicitly)\n")
        f.write("-" * 78 + "\n")
        f.write("  Internal Diversity (IntDiv)      = 1 - mean pairwise Tanimoto.\n")
        f.write("      Previously mislabelled 'Mean Inception Score'. It is NOT an\n")
        f.write("      inception score (no classifier, no softmax, no KL divergence);\n")
        f.write("      it is the MOSES-standard internal-diversity statistic.\n")
        f.write("  FP-PCA-FD                        = Frechet distance in a 2-component PCA\n")
        f.write("      projection of Morgan-fingerprint space. Previously reported as\n")
        f.write("      'Frechet Distance'. It is NOT Frechet ChemNet Distance.\n")
        f.write("  PFD (Property-Embedding Frechet) = Frechet distance in the penultimate\n")
        f.write("      activation space of a property model trained on THIS domain\n")
        f.write("      (OPV donors/acceptors). PRIMARY domain-appropriate distance.\n")
        f.write("  FCD (Frechet ChemNet Distance)   = the real thing, via the `fcd` package.\n")
        f.write("      CAVEAT: ChemNet is ChEMBL-bioactivity-trained, so this measures\n")
        f.write("      distance in a drug-likeness embedding, not an OPV-relevant one.\n")
        f.write("      Reported for cross-field comparability only.\n")
        f.write("  TANGO-inspired heuristic         = building-block-overlap screen. NOT the\n")
        f.write("      published retrosynthesis-constrained TANGO framework, and NOT a\n")
        f.write("      probability that a synthesis route exists.\n\n")

        if not gm.empty:
            f.write("=" * 78 + "\n")
            f.write("GENERATIVE METRICS\n")
            f.write("=" * 78 + "\n")
            cols = ["set", "n_proposed", "validity", "uniqueness", "novelty",
                    "internal_diversity", "scaffold_diversity",
                    "fp_pca_frechet_distance", "property_embedding_frechet_distance",
                    "frechet_chemnet_distance", "nn_similarity_mean",
                    "wasserstein_norm_mean", "mahalanobis_gen_mean",
                    "legacy_property_match_mean", "sascore_mean"]
            avail = [c for c in cols if c in gm.columns]
            for _, row in gm.iterrows():
                f.write("\n-- %s --\n" % row.get("set", "?"))
                for c in avail:
                    if c == "set":
                        continue
                    v = row.get(c)
                    if pd.isna(v):
                        continue
                    f.write("  %-42s : %s\n" % (c, ("%.4f" % v) if isinstance(v, float) else v))
            f.write("\nNOTE: rows named '*_pooled_excluding_Training_Set_Augmentation' give the\n")
            f.write("exploratory-methods-only view. Training_Set_Augmentation dominates the\n")
            f.write("full pool (~18x the per-rank output of each of the other 12 approaches,\n")
            f.write("by design: it carries a >=3-variants-per-training-molecule coverage\n")
            f.write("guarantee rather than a fixed per-rank budget), so the pooled-all number\n")
            f.write("is disproportionately characterised by near-neighbours of the training\n")
            f.write("set. Both views are reported; no generated data is discarded.\n\n")

        if race is not None and not race.empty:
            f.write("=" * 78 + "\n")
            f.write("PROPERTY-MODEL SELECTION (group-disjoint CV horse race)\n")
            f.write("=" * 78 + "\n")
            f.write("Each target was contested between a classical featurised ensemble\n")
            f.write("(Ridge/RandomForest/GaussianProcess on Morgan fingerprints) and a real\n")
            f.write("D-MPNN, decided on HELD-OUT group-disjoint R^2. No family was assumed\n")
            f.write("superior; the D-MPNN is skipped where a target has too few unique\n")
            f.write("molecules for a GNN to generalise (that reason is recorded per target).\n\n")
            f.write("%-26s %-16s %6s %7s %9s %9s %-10s\n"
                    % ("target", "source", "n", "groups", "clas_R2", "dmpnn_R2", "winner"))
            f.write("-" * 78 + "\n")
            for _, r in race.iterrows():
                def _f(x):
                    try:
                        return "%9.4f" % float(x)
                    except Exception:
                        return "%9s" % "-"
                f.write("%-26s %-16s %6s %7s %s %s %-10s\n"
                        % (str(r.get("target", ""))[:26], str(r.get("source", ""))[:16],
                           r.get("n_train", ""), r.get("n_cv_groups", ""),
                           _f(r.get("classical_r2")), _f(r.get("dmpnn_r2")),
                           str(r.get("winner", ""))))
            n_c = int((race["winner"] == "classical").sum()) if "winner" in race.columns else 0
            n_d = int((race["winner"] == "dmpnn").sum()) if "winner" in race.columns else 0
            f.write("\n  Winners: classical ensemble %d targets, D-MPNN %d targets\n" % (n_c, n_d))

        for label, df in (("DONOR", don), ("ACCEPTOR", acc)):
            if df.empty:
                continue
            f.write("\n" + "=" * 78 + "\n")
            f.write("PREDICTED %s MOLECULAR PROPERTIES (pooled, n=%d molecules)\n" % (label, len(df)))
            f.write("=" * 78 + "\n")
            for c in df.columns:
                if c == "smiles":
                    continue
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if s.empty:
                    continue
                f.write("  %-32s mean=%10.4f std=%10.4f min=%10.4f max=%10.4f n=%d\n"
                        % (c, s.mean(), s.std(), s.min(), s.max(), len(s)))

        if not pair.empty:
            f.write("\n" + "=" * 78 + "\n")
            f.write("PREDICTED DEVICE PROPERTIES AND ENERGY LOSS (pooled, n=%d pairs)\n" % len(pair))
            f.write("=" * 78 + "\n")
            for c in pair.columns:
                if c in ("donor_smiles", "acceptor_smiles", "tier"):
                    continue
                s = pd.to_numeric(pair[c], errors="coerce").dropna()
                if s.empty:
                    continue
                f.write("  %-32s mean=%10.4f std=%10.4f min=%10.4f max=%10.4f n=%d\n"
                        % (c, s.mean(), s.std(), s.min(), s.max(), len(s)))
            if "eloss_pred_unphysical" in pair.columns:
                nb = int(pair["eloss_pred_unphysical"].fillna(False).astype(bool).sum())
                f.write("\n  E_LOSS PHYSICS GATE: %d/%d predicted pairs fell below the\n"
                        % (nb, len(pair)))
                f.write("  detailed-balance radiative floor (0.25 eV) and are EXCLUDED from\n")
                f.write("  'eloss_pred_physical'. Rationale: Voc cannot exceed the optical gap\n")
                f.write("  and a non-zero radiative loss is unavoidable, so such values are\n")
                f.write("  model artefacts. They are flagged, not silently rewritten.\n")

        if not audit.empty:
            f.write("\n" + "=" * 78 + "\n")
            f.write("RETROSYNTHESIS AUDIT (shortlist)\n")
            f.write("=" * 78 + "\n")
            backends = audit["audit_backend"].astype(str).value_counts().to_dict() \
                if "audit_backend" in audit.columns else {}
            f.write("  molecules audited : %d\n" % len(audit))
            f.write("  backends used     : %s\n" % backends)
            if "route_found" in audit.columns:
                rf = pd.to_numeric(audit["route_found"], errors="coerce")
                n_ev = int(rf.notna().sum())
                n_ok = int((rf == 1.0).sum())
                if n_ev:
                    f.write("  route found       : %d/%d (%.1f%%)\n"
                            % (n_ok, n_ev, 100.0 * n_ok / n_ev))
            for c in ("sascore", "rascore", "n_route_steps", "tango_inspired"):
                if c in audit.columns:
                    s = pd.to_numeric(audit[c], errors="coerce").dropna()
                    if not s.empty:
                        f.write("  %-16s : mean=%.4f min=%.4f max=%.4f n=%d\n"
                                % (c, s.mean(), s.min(), s.max(), len(s)))
            f.write("\n  EVIDENCE TIER: only molecules audited with the AiZynthFinder backend\n")
            f.write("  carry route-search evidence. RAscore is a ChEMBL-domain distilled\n")
            f.write("  classifier (its authors state it does not transfer to exotic chemistry,\n")
            f.write("  and OPV scaffolds are out of that domain). SAScore is a purely\n")
            f.write("  structural complexity heuristic. Synthesizability claims must be\n")
            f.write("  qualified by whichever backend actually produced the number.\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("Generated: %s\n" % pd.Timestamp.now().isoformat())
    print("  Wrote %s" % path)

    # Per-side interpretive reports, in the same spirit as the legacy
    # donor/acceptor_analysis_report.txt but for the CORRECTED metric set.
    for side in ("donor", "acceptor"):
        write_postgen_analysis_report(outdir, side, gm, don, acc, pair, audit, race)


# =============================================================================
# INTERPRETIVE REPORTING
#
# The legacy donor/acceptor_analysis_report.txt files paired every number with a
# plain-language reading ("Interpretation: ..."). The new metric set previously
# emitted values only, so a reader had no stated basis for judging whether e.g.
# IntDiv=0.848 or PFD=499 is good, bad or meaningless. These helpers restore
# that, with thresholds that are stated explicitly rather than implied, and with
# the honest caveats (screen-only surrogate, out-of-domain FCD/RAscore,
# heuristic TANGO) attached to the specific numbers they qualify.
# =============================================================================

def _interp_validity(v):
    if not np.isfinite(v):
        return "not computed."
    if v >= 0.99:
        return ("essentially every proposed string is a chemically parseable molecule. "
                "Note this is expected to be ~1.0 here because the generation pipeline "
                "already applied RDKit validity filters before writing the pool, so this "
                "confirms the filter worked rather than measuring raw generator quality.")
    if v >= 0.9:
        return "high validity; a small minority of proposals fail RDKit parsing."
    return ("low validity -- a substantial fraction of proposals are not parseable "
            "molecules, which would indicate a decoding problem upstream.")


def _interp_uniqueness(v):
    if not np.isfinite(v):
        return "not computed."
    if v >= 0.99:
        return ("virtually no duplicate structures after canonicalisation; the pool is "
                "not padded by repeats.")
    if v >= 0.9:
        return "high uniqueness with a small duplicate population."
    return ("substantial duplication -- the effective library is much smaller than the "
            "raw count suggests.")


def _interp_novelty(v):
    if not np.isfinite(v):
        return "not computed."
    if v >= 0.99:
        return ("no generated molecule is an exact canonical match to a training molecule, "
                "i.e. nothing was memorised and echoed back. This is an EXACT-MATCH test "
                "only -- it says nothing about how structurally close the molecules are; "
                "read it together with nn_similarity_mean below.")
    if v >= 0.8:
        return "mostly novel, with a minority reproducing training structures exactly."
    return ("low novelty -- a large fraction of the pool duplicates training structures, "
            "suggesting memorisation.")


def _interp_intdiv(v):
    if not np.isfinite(v):
        return "not computed."
    if v >= 0.85:
        return ("very high internal diversity (mean pairwise Tanimoto ~%.3f). The library "
                "spans a broad fingerprint space rather than clustering on one motif."
                % (1.0 - v))
    if v >= 0.7:
        return ("good internal diversity (mean pairwise Tanimoto ~%.3f); a healthy spread "
                "for a generative library." % (1.0 - v))
    if v >= 0.5:
        return ("moderate diversity (mean pairwise Tanimoto ~%.3f); noticeable clustering."
                % (1.0 - v))
    return ("low diversity (mean pairwise Tanimoto ~%.3f) -- the pool is dominated by "
            "near-identical variants of a few scaffolds." % (1.0 - v))


def _interp_nn_sim(v, frac_below):
    if not np.isfinite(v):
        return "not computed."
    txt = ("mean nearest-neighbour Tanimoto to the training set is %.3f" % v)
    if np.isfinite(frac_below):
        txt += (", with %.1f%% of molecules below 0.5" % (100.0 * frac_below))
    if v < 0.3:
        txt += (". This is far from the training fingerprint neighbourhoods: the workflow "
                "moved well beyond local analogue generation. Low similarity is NOT by "
                "itself evidence of usefulness -- it must be read alongside the property "
                "and synthesizability columns.")
    elif v < 0.6:
        txt += (". Moderate structural departure: a mix of analogues and genuinely new "
                "scaffolds.")
    else:
        txt += (". The pool stays close to training structures, i.e. mostly local "
                "analogue generation.")
    return txt


def _interp_scaffold_div(v):
    if not np.isfinite(v):
        return "not computed."
    if v >= 0.7:
        return ("high Bemis-Murcko scaffold diversity -- the library is not a single "
                "scaffold decorated many ways.")
    if v >= 0.4:
        return "moderate scaffold diversity."
    return ("low scaffold diversity -- most molecules share a small number of core "
            "frameworks.")


def _interp_sascore(v):
    if not np.isfinite(v):
        return "not computed."
    if v <= 4.0:
        return ("mean SAScore %.2f on the 1 (easy) to 10 (hard) Ertl-Schuffenhauer scale, "
                "i.e. the pool looks synthetically approachable by fragment-frequency "
                "standards. CAVEAT: SAScore is a purely structural complexity heuristic -- "
                "it never demonstrates that a route exists." % v)
    if v <= 6.0:
        return ("mean SAScore %.2f (1=easy, 10=hard): moderate predicted synthetic "
                "complexity. Structural heuristic only, not route evidence." % v)
    return ("mean SAScore %.2f (1=easy, 10=hard): high predicted synthetic complexity; "
            "many molecules would be demanding to make." % v)


def _interp_distances(fp_pca, pfd, fcd):
    lines = []
    lines.append("  The three distances are NOT comparable to each other in magnitude -- each")
    lines.append("  is a Frechet distance computed in a DIFFERENT embedding space, so only")
    lines.append("  relative comparisons WITHIN one metric (e.g. donor vs acceptor, or")
    lines.append("  approach vs approach) are meaningful.")
    if np.isfinite(fp_pca):
        lines.append("    FP-PCA-FD = %.3f : distance in a 2-component PCA projection of" % fp_pca)
        lines.append("        Morgan-fingerprint space. This is the metric the earlier reports")
        lines.append("        mislabelled as 'Frechet Distance'. Only 2 PCs are retained, so it")
        lines.append("        captures the coarsest structural drift and ignores everything the")
        lines.append("        first two components do not explain. Lower = closer to training.")
    if np.isfinite(pfd):
        lines.append("    PFD       = %.3f : PRIMARY domain-appropriate distance. Computed in" % pfd)
        lines.append("        the penultimate activation space of a property model trained on")
        lines.append("        THIS dataset's OPV donors/acceptors, so the axes correspond to")
        lines.append("        features that actually predict device-relevant properties. Its")
        lines.append("        absolute scale is arbitrary (set by the model's activations), so")
        lines.append("        use it comparatively, never as an absolute quality number.")
    if np.isfinite(fcd):
        lines.append("    FCD       = %.3f : real Frechet ChemNet Distance. CAVEAT: ChemNet is" % fcd)
        lines.append("        trained on ChEMBL bioactivity, i.e. drug-like space. OPV")
        lines.append("        donors/acceptors are out of that domain, so this measures drug-")
        lines.append("        likeness-space distance. Reported ONLY for cross-field")
        lines.append("        comparability with the generative-chemistry literature.")
    else:
        lines.append("    FCD       = not available for this set. ChemNet's one-hot encoder is")
        lines.append("        hard-capped at 350 characters; SMILES longer than that are excluded")
        lines.append("        before the call (see the FCD exclusion counts in the log). If every")
        lines.append("        molecule on a side exceeds the cap, FCD cannot be computed at all.")
    return lines


def write_postgen_analysis_report(outdir, side, gm, don, acc, pair, audit, race):
    """Write <side>_postgen_analysis_report.txt: values PLUS interpretation.

    Mirrors the readability of the legacy donor/acceptor_analysis_report.txt but
    for the corrected metric set (IntDiv instead of the mislabelled 'MIS',
    three separately-named distances instead of one ambiguous 'Frechet
    Distance', MOSES validity/uniqueness/novelty, SAScore, and the tiered
    synthesizability evidence).
    """
    path = os.path.join(outdir, "%s_postgen_analysis_report.txt" % side)
    mol_df = don if side == "donor" else acc

    row = None
    if gm is not None and not gm.empty and "set" in gm.columns:
        want = "%s_pooled_all_approaches" % side
        hits = gm[gm["set"].astype(str) == want]
        if hits.empty:
            hits = gm[gm["set"].astype(str).str.startswith(side)]
        if not hits.empty:
            row = hits.iloc[0]

    def _g(key, default=float("nan")):
        if row is None or key not in row.index:
            return default
        try:
            return float(row[key])
        except Exception:
            return default

    with open(path, "w", encoding="utf-8") as f:
        f.write("Post-Generation Analysis Report -- %s molecules\n" % side.upper())
        f.write("=" * 78 + "\n\n")
        f.write("SCOPE AND STATUS OF THESE NUMBERS\n")
        f.write("-" * 78 + "\n")
        f.write("  Device-property values quoted anywhere in this file come from a\n")
        f.write("  DESCRIPTOR-BASED INTERPOLATIVE SCREENING SURROGATE (Ridge + Gaussian\n")
        f.write("  Process on Morgan fingerprints), not a validated predictor. Its honest\n")
        f.write("  held-out error is in method1_disjoint_cv_report.csv. Use it to rank and\n")
        f.write("  shortlist candidates; do not quote it as predicted device performance.\n")
        f.write("  The Scharber-type physics estimate (previously 'Method 2') has been\n")
        f.write("  REMOVED from every quantitative path because its distributions were\n")
        f.write("  dominated by hard-coded clipping ceilings rather than emergent physics.\n\n")

        if row is None:
            f.write("No pooled generative-metric row was found for the %s side; only the\n"
                    % side)
            f.write("per-molecule property table is summarised below.\n\n")
        else:
            n_prop = _g("n_proposed")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 78 + "\n")
            f.write("  Molecules analysed (pooled) : %s\n"
                    % (("%d" % n_prop) if np.isfinite(n_prop) else "n/a"))
            f.write("  Validity / Uniqueness / Novelty : %.4f / %.4f / %.4f\n"
                    % (_g("validity"), _g("uniqueness"), _g("novelty")))
            f.write("  Internal Diversity (IntDiv)  : %.4f\n" % _g("internal_diversity"))
            f.write("  Mean SAScore                 : %.3f\n" % _g("sascore_mean"))
            f.write("\n")

            f.write("MOSES-STANDARD GENERATIVE METRICS\n")
            f.write("-" * 78 + "\n")
            f.write("  Validity  = %.4f\n" % _g("validity"))
            f.write("    Interpretation: %s\n\n" % _interp_validity(_g("validity")))
            f.write("  Uniqueness = %.4f\n" % _g("uniqueness"))
            f.write("    Interpretation: %s\n\n" % _interp_uniqueness(_g("uniqueness")))
            f.write("  Novelty    = %.4f\n" % _g("novelty"))
            f.write("    Interpretation: %s\n\n" % _interp_novelty(_g("novelty")))
            f.write("  Internal Diversity (IntDiv) = %.4f\n" % _g("internal_diversity"))
            f.write("    Definition: 1 - mean pairwise Morgan-Tanimoto similarity over a\n")
            f.write("      seeded sample of the pool (sample cap and seed are recorded in\n")
            f.write("      generative_metrics.csv so the number is reproducible).\n")
            f.write("    NOTE ON NAMING: earlier reports called this the 'Mean Inception\n")
            f.write("      Score'. That was wrong -- there is no classifier, no softmax and\n")
            f.write("      no KL divergence anywhere in it. It is the MOSES internal-\n")
            f.write("      diversity statistic and is named as such here.\n")
            f.write("    Interpretation: %s\n\n" % _interp_intdiv(_g("internal_diversity")))
            f.write("  Scaffold diversity = %.4f\n" % _g("scaffold_diversity"))
            f.write("    Interpretation: %s\n\n"
                    % _interp_scaffold_div(_g("scaffold_diversity")))

            f.write("SIMILARITY TO THE TRAINING SET\n")
            f.write("-" * 78 + "\n")
            f.write("  nn_similarity_mean = %.4f  (range %.4f - %.4f)\n"
                    % (_g("nn_similarity_mean"), _g("nn_similarity_min"),
                       _g("nn_similarity_max")))
            f.write("    Definition: maximum Tanimoto similarity of each generated molecule\n")
            f.write("      to ANY training molecule (radius-2, 2048-bit Morgan), averaged.\n")
            f.write("      It is a nearest-neighbour measure, not a distribution distance.\n")
            f.write("    Interpretation: %s\n\n"
                    % _interp_nn_sim(_g("nn_similarity_mean"),
                                     _g("nn_similarity_frac_below_0p5")))

            f.write("DISTRIBUTION-DISTANCE METRICS (three, deliberately separate)\n")
            f.write("-" * 78 + "\n")
            for line in _interp_distances(_g("fp_pca_frechet_distance"),
                                          _g("property_embedding_frechet_distance"),
                                          _g("frechet_chemnet_distance")):
                f.write(line + "\n")
            f.write("\n")

            f.write("DESCRIPTOR-DOMAIN AGREEMENT\n")
            f.write("-" * 78 + "\n")
            lpm = _g("legacy_property_match_mean")
            if np.isfinite(lpm):
                f.write("  legacy_property_match_mean = %.4f\n" % lpm)
                f.write("    Definition: fraction of 6 RDKit descriptors (MW, LogP, TPSA, HBA,\n")
                f.write("      HBD, RotBonds) falling inside the training min-max interval.\n")
                f.write("    LIMITATION: it counts binary in/out conditions, so it can only\n")
                f.write("      take discrete values separated by 1/6 = 0.167 -- which is why\n")
                f.write("      its histogram shows sharp spikes rather than a smooth spread.\n")
                f.write("      It says nothing about frontier-orbital energies, absorption or\n")
                f.write("      device properties. Retained only for continuity with the older\n")
                f.write("      reports; prefer the two distribution-aware measures below.\n")
                f.write("    Interpretation: on average %.2f of the 6 descriptors stay inside\n"
                        % (6.0 * lpm))
                f.write("      the training ranges.\n\n")
            wnm = _g("wasserstein_norm_mean")
            if np.isfinite(wnm):
                f.write("  wasserstein_norm_mean = %.4f\n" % wnm)
                f.write("    Definition: mean per-descriptor 1-Wasserstein distance between the\n")
                f.write("      generated and training descriptor distributions, normalised by\n")
                f.write("      the training spread. Distribution-aware, continuous, and it\n")
                f.write("      does not collapse to discrete steps the way the 6-condition\n")
                f.write("      count above does. Lower = the generated descriptor\n")
                f.write("      distributions track the training ones more closely.\n\n")
            mgm = _g("mahalanobis_gen_mean")
            if np.isfinite(mgm):
                f.write("  mahalanobis_gen_mean = %.4f\n" % mgm)
                f.write("    Definition: mean Mahalanobis distance of generated molecules to\n")
                f.write("      the training descriptor manifold, which accounts for descriptor\n")
                f.write("      CORRELATION instead of treating each axis independently. Large\n")
                f.write("      values mean the pool sits outside the correlated descriptor\n")
                f.write("      envelope the downstream surrogate was fitted on -- i.e. those\n")
                f.write("      predictions are extrapolations and should be trusted less.\n\n")

            f.write("SYNTHESIZABILITY\n")
            f.write("-" * 78 + "\n")
            f.write("  SAScore mean = %.3f\n" % _g("sascore_mean"))
            f.write("    Interpretation: %s\n" % _interp_sascore(_g("sascore_mean")))
            fle = _g("sascore_frac_le_6")
            if np.isfinite(fle):
                f.write("    %.1f%% of molecules fall at or below SAScore 6.\n"
                        % (100.0 * fle))
            f.write("\n  EVIDENCE TIERS -- what each synthesizability number can support:\n")
            f.write("    TIER R1 AiZynthFinder MCTS route search : real retrosynthesis\n")
            f.write("            evidence. Only molecules audited with this backend may be\n")
            f.write("            described as having a found route.\n")
            f.write("    TIER R2 RAscore : a classifier distilled FROM AiZynthFinder outcomes,\n")
            f.write("            so it approximates 'is a route findable' in milliseconds.\n")
            f.write("            Its own authors state it does not transfer outside its ChEMBL\n")
            f.write("            training domain, and these OPV scaffolds are outside it.\n")
            f.write("            Indicative only.\n")
            f.write("    TIER R3 SAScore / TANGO-inspired heuristic : structural complexity and\n")
            f.write("            building-block overlap. Fast screens. Neither is a probability\n")
            f.write("            that a synthesis route exists, and the TANGO-inspired score is\n")
            f.write("            NOT the published retrosynthesis-constrained TANGO framework.\n\n")

        if mol_df is not None and not mol_df.empty:
            f.write("PREDICTED %s MOLECULAR PROPERTIES\n" % side.upper())
            f.write("-" * 78 + "\n")
            f.write("  (n reported per column: a property is only predicted for molecules the\n")
            f.write("   corresponding model could featurise; a low n means that model was\n")
            f.write("   unavailable or its source dataset was missing for this run.)\n\n")
            for c in mol_df.columns:
                if c == "smiles":
                    continue
                s = pd.to_numeric(mol_df[c], errors="coerce").dropna()
                if s.empty:
                    f.write("  %-30s : NO VALUES -- model unavailable for this run.\n" % c)
                    continue
                f.write("  %-30s : mean=%10.4f std=%9.4f min=%10.4f max=%10.4f n=%d\n"
                        % (c, s.mean(), s.std(), s.min(), s.max(), len(s)))
            f.write("\n")

        if race is not None and not race.empty and "winner" in race.columns:
            f.write("WHICH MODEL PREDICTS EACH PROPERTY\n")
            f.write("-" * 78 + "\n")
            sub = race[race["target"].astype(str).str.startswith(
                "don_" if side == "donor" else "acc_")] if "target" in race.columns else race
            if not sub.empty:
                for _, r in sub.iterrows():
                    f.write("  %-24s winner=%-10s heldout_R2(classical)=%s\n"
                            % (str(r.get("target", ""))[:24], str(r.get("winner", "")),
                               ("%.4f" % float(r["classical_r2"]))
                               if "classical_r2" in r.index and pd.notna(r["classical_r2"])
                               else "n/a"))
                f.write("\n  A NEGATIVE held-out R2 means the model predicts worse than simply\n")
                f.write("  predicting the training mean, i.e. it does not generalise to unseen\n")
                f.write("  molecules of that group. For donor-side targets this is expected:\n")
                f.write("  only ~38 unique donor structures exist in DMPNN.csv, which is far\n")
                f.write("  too few to learn a generalisable donor-property function. Those\n")
                f.write("  columns are PROVISIONAL and must not be presented as predictions.\n\n")

        f.write("=" * 78 + "\n")
        f.write("Generated: %s\n" % pd.Timestamp.now().isoformat())
    print("  Wrote %s" % path)


def find_best_rank(rank_dirs):
    """Find rank with highest total molecule count (most complete run)."""
    best_rank = rank_dirs[0]
    best_count = 0
    for rd in rank_dirs:
        marker = os.path.join(rd, "rank_done.json")
        if os.path.exists(marker):
            try:
                with open(marker) as f:
                    data = json.load(f)
                count = data.get("donor_count", 0) + data.get("acceptor_count", 0)
                if count > best_count:
                    best_count = count
                    best_rank = rd
            except Exception:
                pass
    return best_rank


def copy_if_exists(src, dst):
    """Copy file if source exists, creating destination directories as needed."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge all rank_XXX/ outputs into a single universal folder"
    )
    parser.add_argument("--indir", required=True, help="Base directory containing rank_*/ outputs")
    parser.add_argument("--outdir", required=True, help="Merged output directory")
    parser.add_argument(
        "--training-csv", default=None,
        help="Explicit path to the training CSV (DMPNN.csv or Training_dataset_GAN.csv) "
             "used for pooled report/plot regeneration. If omitted, the script searches "
             "--indir, its parent, and the directory this script itself lives in "
             "(which is BASE_WORKDIR/submit-dir when GAN_240.py invokes this script)."
    )
    # ------------------------------------------------------------------
    # STAGE SELECTOR (added)
    #   generation : merge GAN_240_final.py's rank outputs (default; the
    #                historical behaviour, now with the per-approach and
    #                "excluding Training_Set_Augmentation" pooling fixes)
    #   postgen    : merge postgen_pipeline.py's rank shards (property
    #                predictions, E_loss, SAScore, MOSES metrics, PFD/FCD,
    #                retrosynthesis audit)
    # ------------------------------------------------------------------
    parser.add_argument("--stage", choices=["generation", "postgen"], default="generation",
                        help="Which pipeline's rank outputs to merge (default: generation).")
    parser.add_argument("--recompute-pooled-stats-only", action="store_true",
                        help="Generation stage only: recompute pooled/per-approach statistics and "
                             "rewrite output_metrics.txt WITHOUT re-merging CSVs or regenerating "
                             "plots. This is the fast path for label/aggregation corrections that "
                             "need no molecule regeneration.")
    args = parser.parse_args()

    if args.stage == "postgen":
        return merge_postgen_stage(os.path.abspath(args.indir), os.path.abspath(args.outdir))

    indir = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    results_dir = os.path.join(outdir, "results")
    os.makedirs(results_dir, exist_ok=True)
    viz_dir = os.path.join(outdir, "visualization_outputs")
    os.makedirs(viz_dir, exist_ok=True)

    rank_dirs = sorted(glob.glob(os.path.join(indir, "rank_*")))
    if not rank_dirs:
        # A full merge genuinely needs the per-rank outputs. The pooled-stats
        # fast path does not: it works from the already-merged CSVs, and the
        # only thing it loses is the per-MOLECULE approach attribution (which
        # lives in the per-rank *_generation_sources.csv files). Long-finished
        # runs often have their rank_* directories cleaned up, so refusing to
        # start in that case would block the recompute for no good reason --
        # warn and continue with pooled-only statistics instead.
        if args.recompute_pooled_stats_only:
            print(f"[WARN] No rank_* directories found in {indir}.")
            print("       Continuing in --recompute-pooled-stats-only mode using the")
            print("       already-merged CSVs. CONSEQUENCE: per-molecule approach labels")
            print("       are unavailable, so the PER-APPROACH breakdown (and therefore")
            print("       per_approach_full_stats_*.png) will be skipped; pooled(all)")
            print("       statistics are still computed.")
            best_rank = None
        else:
            raise RuntimeError(
                f"No rank_* directories found in {indir}. A full merge requires the "
                f"per-rank outputs. If they have been cleaned up, re-run with "
                f"--recompute-pooled-stats-only to recompute pooled statistics from the "
                f"already-merged CSVs instead."
            )
    else:
        print(f"Found {len(rank_dirs)} rank directories in {indir}")
        # Identify the "best" rank (most molecules) for non-mergeable single-file outputs
        best_rank = find_best_rank(rank_dirs)
        print(f"Best rank (most molecules): {best_rank}")

    # -------------------------------------------------------------------------
    # FAST PATH: recompute pooled/per-approach statistics only.
    # For the label/aggregation corrections (per-approach breakdown, pooled
    # excluding Training_Set_Augmentation, Tier1/Tier2 split, FF threshold text)
    # nothing needs re-merging and no molecule needs regenerating -- the merged
    # CSVs already on disk contain everything required.
    # -------------------------------------------------------------------------
    if args.recompute_pooled_stats_only:
        print()
        print("--recompute-pooled-stats-only: skipping CSV re-merge and plot regeneration.")
        train_donor_mols, train_acceptor_mols = [], []
        tcsv = args.training_csv
        if tcsv is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for cand in (os.path.join(script_dir, "DMPNN.csv"),
                         os.path.join(indir, "DMPNN.csv"),
                         os.path.abspath(os.path.join(indir, "..", "DMPNN.csv")),
                         os.path.abspath(os.path.join(indir, "..", "..", "DMPNN.csv"))):
                if os.path.exists(cand):
                    tcsv = cand
                    break
        if tcsv and os.path.exists(tcsv) and REGEN_AVAILABLE:
            tdf = read_csv_safe(tcsv)
            if tdf is not None:
                if "donor(SMILES)" in tdf.columns:
                    train_donor_mols = [m for m in (Chem.MolFromSmiles(str(s))
                                                    for s in tdf["donor(SMILES)"].dropna())
                                        if m is not None]
                if "acceptor(SMILES)" in tdf.columns:
                    train_acceptor_mols = [m for m in (Chem.MolFromSmiles(str(s))
                                                       for s in tdf["acceptor(SMILES)"].dropna())
                                           if m is not None]
            print("  Training reference: %s (%d donor / %d acceptor molecules)"
                  % (tcsv, len(train_donor_mols), len(train_acceptor_mols)))
        else:
            print("  [WARN] No training CSV located; distance metrics will be skipped.")
        compute_pooled_and_per_approach_stats(
            outdir, results_dir, train_donor_mols, train_acceptor_mols, rank_dirs)
        write_tier_breakdown(results_dir, outdir)
        print()
        print("=" * 70)
        print("POOLED-STATS RECOMPUTE COMPLETE (no re-merge, no regeneration)")
        print("=" * 70)
        return 0

    # =========================================================================
    # 1. MERGE PREDICTION CSVs (concatenate + deduplicate)
    # =========================================================================
    # Simple dedup: safe for files where the same key implies identical data
    # across ranks (deterministic, training-set-derived characterization).
    simple_dedup_configs = [
        ("generated_molecules_with_predictions.csv", ["donor_smiles", "acceptor_smiles"]),
        ("high_performance_pairs.csv", ["donor_smiles", "acceptor_smiles"]),
        ("tango_scores.csv", ["SMILES", "Type"]),
        ("donor_preprocessing_report.csv", ["raw_smiles"]),
        ("acceptor_preprocessing_report.csv", ["raw_smiles"]),
        ("donor_training_trait_coverage.csv", ["trait_signature"]),
        ("acceptor_training_trait_coverage.csv", ["trait_signature"]),
    ]

    for filename, dedup_keys in simple_dedup_configs:
        csv_paths = [os.path.join(r, "results", filename) for r in rank_dirs]
        merged = concat_csvs(csv_paths)
        if not merged.empty:
            usable_keys = [k for k in dedup_keys if k in merged.columns]
            if usable_keys:
                merged = merged.drop_duplicates(subset=usable_keys, keep="first")
            else:
                merged = merged.drop_duplicates(keep="first")
        merged.to_csv(os.path.join(results_dir, filename), index=False)
        print(f"  Merged {filename}: {len(merged)} rows")

    # Per-training-smiles coverage CSVs: every rank processes the SAME full
    # training set but generates DIFFERENT example variants for each training
    # molecule (different GAN, different random draws). A plain
    # drop_duplicates(keep="first") would discard 239/240 ranks' worth of
    # distinct example variants. Instead we aggregate: sum variants_generated
    # across ranks, promote coverage_status to the best value seen, and keep
    # one representative example (all actual novel molecules are still fully
    # preserved in generated_molecules_with_predictions.csv regardless).
    coverage_status_rank = {"full_coverage": 2, "partial_coverage": 1, "no_coverage": 0}
    for filename in [
        "donor_per_training_smiles_generation_coverage.csv",
        "acceptor_per_training_smiles_generation_coverage.csv",
    ]:
        csv_paths = [os.path.join(r, "results", filename) for r in rank_dirs]
        merged = concat_csvs(csv_paths)
        if not merged.empty and "training_smiles" in merged.columns:
            if "variants_generated" in merged.columns:
                merged["variants_generated"] = pd.to_numeric(
                    merged["variants_generated"], errors="coerce"
                ).fillna(0)

            # Best coverage_status per training_smiles (full > partial > no_coverage)
            status_lookup = None
            if "coverage_status" in merged.columns:
                status_rank = merged["coverage_status"].map(coverage_status_rank).fillna(0)
                best_idx = status_rank.groupby(merged["training_smiles"]).idxmax()
                status_lookup = merged.loc[best_idx, ["training_smiles", "coverage_status"]]

            agg_map = {}
            if "variants_generated" in merged.columns:
                agg_map["variants_generated"] = "sum"
            if "generated_variants" in merged.columns:
                agg_map["generated_variants"] = "first"
            for other_col in ["canonical_training_smiles", "trait_signature", "rdkit_parseable",
                               "attempts", "row_position_in_training_list", "molecule_type"]:
                if other_col in merged.columns:
                    agg_map[other_col] = "first"

            if agg_map:
                merged = merged.groupby("training_smiles", as_index=False).agg(agg_map)
            else:
                merged = merged.drop_duplicates(subset=["training_smiles"], keep="first")

            if status_lookup is not None:
                merged = merged.merge(status_lookup, on="training_smiles", how="left")
        merged.to_csv(os.path.join(results_dir, filename), index=False)
        print(f"  Aggregated {filename}: {len(merged)} unique training molecules")

    # Aggregate approach statistics (sum counts across ranks)
    app_stats_csvs = [os.path.join(r, "results", "approach_statistics.csv") for r in rank_dirs]
    merged_app_stats = concat_csvs(app_stats_csvs)
    if not merged_app_stats.empty and "approach" in merged_app_stats.columns:
        num_cols = [c for c in merged_app_stats.columns if c != "approach"]
        merged_app_stats = merged_app_stats.groupby("approach", as_index=False)[num_cols].sum()
    merged_app_stats.to_csv(os.path.join(results_dir, "approach_statistics.csv"), index=False)

    # Aggregate tango stats (mean across ranks)
    app_tango_csvs = [os.path.join(r, "results", "approach_tango_stats.csv") for r in rank_dirs]
    merged_app_tango = concat_csvs(app_tango_csvs)
    if not merged_app_tango.empty and "approach" in merged_app_tango.columns:
        agg = {
            "count": "sum",
            "mean_tango": "mean",
            "std_tango": "mean",
            "min_tango": "min",
            "max_tango": "max",
        }
        usable = {k: v for k, v in agg.items() if k in merged_app_tango.columns}
        if usable:
            merged_app_tango = merged_app_tango.groupby("approach", as_index=False).agg(usable)
    merged_app_tango.to_csv(os.path.join(results_dir, "approach_tango_stats.csv"), index=False)

    # DMPNN prediction stats — recompute from merged predictions
    merged_preds = read_csv_safe(os.path.join(results_dir, "generated_molecules_with_predictions.csv"))
    if merged_preds is not None and not merged_preds.empty:
        stats = {}
        # Include the final blended PCE/Jsc/FF/Voc, synthesizability
        # (pair_tango/donor_tango/acceptor_tango), AND each of the 2
        # individual prediction methods' raw (unblended) outputs:
        #   Method 1 (Ridge+GP surrogate): surrogate_PCE/Jsc/FF/Voc
        #   Method 2 (Scharber physics):    phys_physics_PCE/Jsc/FF/Voc
        # so the merged stats let you compare both methods' pooled
        # distributions side by side, not just the final blended value.
        # (A third method, SISSO++ equations, was removed after
        # verification showed numerically unstable outputs on genuinely
        # novel generated molecules — see GAN_240.py for full details.)
        stat_cols = (
            ['PCE', 'Jsc', 'FF', 'Voc', 'pair_tango', 'donor_tango', 'acceptor_tango']
            + ['surrogate_PCE', 'surrogate_Jsc', 'surrogate_FF', 'surrogate_Voc']
            + ['phys_physics_PCE', 'phys_physics_Jsc', 'phys_physics_FF', 'phys_physics_Voc']
        )
        for col in stat_cols:
            if col in merged_preds.columns:
                col_data = merged_preds[col].dropna()
                if len(col_data) == 0:
                    continue
                stats[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                }
        pd.DataFrame(stats).to_csv(os.path.join(results_dir, "dmpnn_prediction_stats.csv"))
        print(f"  Recomputed dmpnn_prediction_stats.csv from merged data "
              f"(includes final blended + Method 1/2 individual stats)")

    # =========================================================================
    # 2. AVERAGE GAN LOSS/ACCURACY CSVs across ranks
    # =========================================================================
    for gan_file in [
        "gan_losses_donor_optimized.csv",
        "gan_losses_acceptor_optimized.csv",
        "discriminator_accuracy_donor_optimized.csv",
        "discriminator_accuracy_acceptor_optimized.csv",
    ]:
        csv_paths = [os.path.join(r, gan_file) for r in rank_dirs]
        dfs = [read_csv_safe(p) for p in csv_paths]
        dfs = [d for d in dfs if d is not None and not d.empty]
        if dfs:
            # Average numeric columns across ranks (each rank ran the same epoch count)
            # Use the shortest run length to align, and drop 'epoch' before
            # averaging so it can be cleanly re-added afterward (pd.insert
            # raises ValueError if a column with that name already exists).
            min_len = min(len(d) for d in dfs)
            aligned = [
                d.iloc[:min_len].select_dtypes(include=[np.number]).drop(columns=["epoch"], errors="ignore")
                for d in dfs
            ]
            avg_df = pd.concat(aligned).groupby(level=0).mean()
            avg_df.insert(0, 'epoch', range(len(avg_df)))
            avg_df.to_csv(os.path.join(outdir, gan_file), index=False)
            print(f"  Averaged {gan_file}: {len(avg_df)} epochs from {len(dfs)} ranks")

    # =========================================================================
    # 3. COPY SINGLE-FILE OUTPUTS from best rank
    # =========================================================================
    single_files = [
        "output_metrics.txt",
        "donor_analysis_report.txt",
        "acceptor_analysis_report.txt",
        "donor_hybrid_12method_visualization_data.csv",
        "acceptor_hybrid_12method_visualization_data.csv",
    ]
    for f in single_files:
        copy_if_exists(os.path.join(best_rank, f), os.path.join(outdir, f))

    # -------------------------------------------------------------------
    # REGENERATE the Hybrid Excel file from the POOLED dataset across all
    # ranks, instead of copying only the single best-rank file.
    #
    # Root cause fixed: previously this copied ONE rank's Excel file
    # (~1,500 rows, one rank's num_per_approach=50 slice), which is what
    # a user visually inspects for "generated molecule diversity" — but it
    # represented only 1/240th of the true pooled output (which has
    # 50,000+ unique donors/acceptors after the rank-seeding diversity fix).
    # This made the actual diversity improvement invisible in the one file
    # most likely to be opened and reviewed by hand. Now: read every rank's
    # own Hybrid_12Method_Generated_Molecules_*.xlsx, concatenate, dedupe
    # by (donor SMILES, acceptor SMILES) pair, and write the TRUE pooled
    # result. If the pooled row count is very large, a large but bounded
    # random sample (deterministic seed) is written instead of the full
    # set, since multi-hundred-thousand-row Excel files are unwieldy to
    # actually open/scroll through by hand; the full pooled truth remains
    # available in results/generated_molecules_with_predictions.csv.
    # -------------------------------------------------------------------
    print("  Regenerating Hybrid Excel file from POOLED data across all ranks...")
    MAX_EXCEL_ROWS = 100000
    excel_dfs = []
    for rd in rank_dirs:
        rank_excel_files = sorted(glob.glob(os.path.join(rd, "Hybrid_12Method_Generated_Molecules_*.xlsx")))
        if not rank_excel_files:
            continue
        try:
            df_rank_excel = pd.read_excel(rank_excel_files[-1], engine="openpyxl")
            if not df_rank_excel.empty:
                excel_dfs.append(df_rank_excel)
        except Exception as e:
            print(f"    [WARN] Could not read {rank_excel_files[-1]}: {e}")

    if excel_dfs:
        pooled_excel_df = pd.concat(excel_dfs, ignore_index=True)
        dedup_cols = [c for c in ["Generated Donor SMILES", "Generated Acceptor SMILES"] if c in pooled_excel_df.columns]
        if dedup_cols:
            pooled_excel_df = pooled_excel_df.drop_duplicates(subset=dedup_cols, keep="first")
        else:
            pooled_excel_df = pooled_excel_df.drop_duplicates(keep="first")
        total_pooled_rows = len(pooled_excel_df)

        if total_pooled_rows > MAX_EXCEL_ROWS:
            pooled_excel_df = pooled_excel_df.sample(n=MAX_EXCEL_ROWS, random_state=42).reset_index(drop=True)
            print(f"    Pooled {total_pooled_rows} unique donor-acceptor rows across {len(excel_dfs)} ranks; "
                  f"wrote a deterministic {MAX_EXCEL_ROWS}-row sample to Excel (full set is in "
                  f"results/generated_molecules_with_predictions.csv).")
        else:
            print(f"    Pooled {total_pooled_rows} unique donor-acceptor rows across {len(excel_dfs)} ranks "
                  f"(full set, no sampling needed).")

        timestamp = None
        excel_files = sorted(glob.glob(os.path.join(best_rank, "Hybrid_12Method_Generated_Molecules_*.xlsx")))
        if excel_files:
            base_name = os.path.basename(excel_files[-1])
            out_excel_name = base_name.replace("Hybrid_12Method_Generated_Molecules_", "Hybrid_12Method_Generated_Molecules_POOLED_")
        else:
            out_excel_name = "Hybrid_12Method_Generated_Molecules_POOLED.xlsx"
        pooled_excel_df.to_excel(os.path.join(outdir, out_excel_name), index=False)
        print(f"    Wrote pooled Excel: {out_excel_name} ({len(pooled_excel_df)} rows)")
    else:
        print("    [WARN] No per-rank Excel files found; falling back to best-rank copy.")
        excel_files = sorted(glob.glob(os.path.join(best_rank, "Hybrid_12Method_Generated_Molecules_*.xlsx")))
        if excel_files:
            copy_if_exists(excel_files[-1], os.path.join(outdir, os.path.basename(excel_files[-1])))

    # Copy hybrid metadata
    meta_files = sorted(glob.glob(os.path.join(best_rank, "hybrid_12method_metadata_*.txt")))
    if meta_files:
        copy_if_exists(meta_files[-1], os.path.join(outdir, os.path.basename(meta_files[-1])))

    # Copy overall metrics
    metrics_files = sorted(glob.glob(os.path.join(best_rank, "hybrid_12method_overall_metrics_*.txt")))
    if metrics_files:
        copy_if_exists(metrics_files[-1], os.path.join(outdir, os.path.basename(metrics_files[-1])))

    # =========================================================================
    # 4. COPY GAN PLOTS from best rank (PNGs)
    # =========================================================================
    for png in [
        "gan_losses_donor_optimized.png",
        "gan_losses_acceptor_optimized.png",
        "discriminator_accuracy_donor_optimized.png",
        "discriminator_accuracy_acceptor_optimized.png",
    ]:
        copy_if_exists(os.path.join(best_rank, png), os.path.join(outdir, png))

    # =========================================================================
    # 5. COPY VISUALIZATION OUTPUTS from best rank
    # =========================================================================
    best_viz = os.path.join(best_rank, "visualization_outputs")
    if os.path.isdir(best_viz):
        for f in os.listdir(best_viz):
            src = os.path.join(best_viz, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(viz_dir, f))
        print(f"  Copied visualization_outputs/ from best rank ({len(os.listdir(best_viz))} files)")

    # =========================================================================
    # 6. COPY SIMILARITY ANALYSIS from best rank (if exists)
    # =========================================================================
    best_sim = os.path.join(best_rank, "similarity_analysis")
    if os.path.isdir(best_sim):
        sim_dir = os.path.join(outdir, "similarity_analysis")
        os.makedirs(sim_dir, exist_ok=True)
        for f in os.listdir(best_sim):
            src = os.path.join(best_sim, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(sim_dir, f))
        print(f"  Copied similarity_analysis/ from best rank ({len(os.listdir(best_sim))} files)")

    # =========================================================================
    # 7. COPY EXACT MATCH ANALYSIS from best rank (if exists)
    # =========================================================================
    best_em = os.path.join(best_rank, "exact_match_analysis")
    if os.path.isdir(best_em):
        em_dir = os.path.join(outdir, "exact_match_analysis")
        os.makedirs(em_dir, exist_ok=True)
        for f in os.listdir(best_em):
            src = os.path.join(best_em, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(em_dir, f))
        print(f"  Copied exact_match_analysis/ from best rank ({len(os.listdir(best_em))} files)")

    # =========================================================================
    # 8. REGENERATE REPORTS AND PLOTS FROM POOLED DATA (true merged result)
    # Overwrites the best-rank copies made in steps 3-4/6 with versions
    # recomputed over the full merged/deduplicated molecule set.
    # =========================================================================
    print()
    print("Regenerating reports and plots from pooled/merged data...")
    regenerate_reports_and_plots(outdir, best_rank, indir, training_csv_override=args.training_csv)

    # =========================================================================
    # 8b. PER-APPROACH + AUGMENTATION-EXCLUDED POOLING, AND TIER BREAKDOWN
    #
    # Added corrections (no molecule regeneration needed for any of these):
    #   * per-approach MFD / IntDiv / TANGO, extending the pattern that already
    #     existed for TANGO alone -> approach_full_stats.csv
    #   * a "pooled excluding Training_Set_Augmentation" variant reported
    #     ALONGSIDE the pooled-all number, so a reader gets both the honest
    #     full-pool picture and the exploratory-methods-only picture without
    #     discarding any generated data
    #   * Tier 1 and Tier 2 counts reported SEPARATELY (the old single
    #     "high-performance" figure conflated a strict and a relaxed tier)
    #     -> tier_breakdown.csv
    # =========================================================================
    print()
    print("Computing per-approach and augmentation-excluded pooled statistics...")
    try:
        _train_csv = args.training_csv
        _train_donor_mols, _train_acceptor_mols = [], []
        if _train_csv is None:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            for _cand in (os.path.join(_script_dir, "DMPNN.csv"),
                          os.path.join(indir, "DMPNN.csv"),
                          os.path.abspath(os.path.join(indir, "..", "DMPNN.csv")),
                          os.path.abspath(os.path.join(indir, "..", "..", "DMPNN.csv"))):
                if os.path.exists(_cand):
                    _train_csv = _cand
                    break
        if _train_csv and os.path.exists(_train_csv) and REGEN_AVAILABLE:
            _tdf = read_csv_safe(_train_csv)
            if _tdf is not None:
                if "donor(SMILES)" in _tdf.columns:
                    _train_donor_mols = [m for m in (Chem.MolFromSmiles(str(s))
                                                     for s in _tdf["donor(SMILES)"].dropna())
                                         if m is not None]
                if "acceptor(SMILES)" in _tdf.columns:
                    _train_acceptor_mols = [m for m in (Chem.MolFromSmiles(str(s))
                                                        for s in _tdf["acceptor(SMILES)"].dropna())
                                            if m is not None]
        compute_pooled_and_per_approach_stats(
            outdir, results_dir, _train_donor_mols, _train_acceptor_mols, rank_dirs)
    except Exception as _e:
        print("  [WARN] Per-approach pooling failed: %s" % _e)

    try:
        write_tier_breakdown(results_dir, outdir)
    except Exception as _e:
        print("  [WARN] Tier breakdown failed: %s" % _e)

    # =========================================================================
    # 9. WRITE MERGE SUMMARY
    # =========================================================================
    rank_markers = []
    for rank_dir in rank_dirs:
        marker = os.path.join(rank_dir, "rank_done.json")
        if os.path.exists(marker):
            try:
                with open(marker, "r") as f:
                    rank_markers.append(json.load(f))
            except Exception:
                pass

    # Count total unique molecules from merged data
    merged_hp = read_csv_safe(os.path.join(results_dir, "high_performance_pairs.csv"))
    merged_all = read_csv_safe(os.path.join(results_dir, "generated_molecules_with_predictions.csv"))

    summary = {
        "indir": indir,
        "outdir": outdir,
        "num_rank_dirs": len(rank_dirs),
        "num_rank_markers": len(rank_markers),
        "best_rank": best_rank,
        "total_unique_pairs": int(len(merged_all)) if merged_all is not None else 0,
        "total_high_performance_pairs": int(len(merged_hp)) if merged_hp is not None else 0,
        "rows": {
            "generated_molecules_with_predictions": int(len(merged_all)) if merged_all is not None else 0,
            "high_performance_pairs": int(len(merged_hp)) if merged_hp is not None else 0,
        },
    }

    with open(os.path.join(outdir, "merge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # =========================================================================
    # DONE
    # =========================================================================
    print()
    print("=" * 70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Input: {indir} ({len(rank_dirs)} rank directories)")
    print(f"  Output: {outdir}")
    print(f"  Best rank used for single-file outputs: {best_rank}")
    print(f"  Total unique D-A pairs: {summary['total_unique_pairs']}")
    print(f"  High-performance pairs: {summary['total_high_performance_pairs']}")
    print()
    print("Output folder contents:")
    for root, dirs, files in os.walk(outdir):
        level = root.replace(outdir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{sub_indent}{f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
