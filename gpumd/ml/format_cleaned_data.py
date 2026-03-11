"""
format_cleaned_data.py — Build a supervised-learning dataset from GB structural
descriptors and rNEMD thermal results.

Usage:
    python ml/format_cleaned_data.py --config configs/small_box.yaml

For each unique grain boundary in the config:
  - Selects the lowest-energy relaxed structure from gb_generation.
  - Computes structural descriptors (sigma_theta, sigma_l, H_phi, A_RDF, rho)
    and the mean SOAP vector for three atom subsets:
      "full"   — all atoms in the supercell
      "gb{d}"  — atoms within d Å of either GB plane (one per gb_dist_threshold)
  - Joins with rNEMD aggregate results (R_K_mean, kappa_mean, …) where available,
    filling NaN for GBs not yet simulated.

Output:
    results/<config_name>/ml/dataset.csv

Column layout:
    [metadata]  gb_label, axis_x/y/z, sigma, plane_x/y/z
    [descriptors per subset]
        {prefix}_sigma_theta, _sigma_l, _H_phi, _A_RDF, _rho, _n_atoms
        soap_{prefix}_0 … soap_{prefix}_{SOAP_DIM-1}
    [targets / reference]
        R_K_mean, R_K_std, kappa_mean, kappa_std, J_mean, J_std
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make gpumd root importable (ml/ sits one level below gpumd/)
SCRIPT_DIR = Path(__file__).resolve().parent
GPUMD_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(GPUMD_ROOT))

from ase.io import read
from dscribe.descriptors import SOAP

from utils.descriptors import (
    compute_global_descriptors,
    compute_mean_soap,
    identify_gb_atoms,
)
from utils.work_coordination import gb_label as make_gb_label

# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Build supervised-learning dataset from GB descriptors + rNEMD results"
)
parser.add_argument("--config", type=str, required=True,
                    help="Path to unified YAML config (e.g. configs/small_box.yaml)")
args = parser.parse_args()

yaml_path = Path(args.config).resolve()
config_name = yaml_path.stem

with open(yaml_path) as f:
    config = yaml.safe_load(f)

ml_cfg            = config["ml"]
GB_DIST_THRESHOLDS = list(ml_cfg["gb_dist_thresholds"])   # e.g. [5.0, 10.0]
BOND_CUTOFF        = float(ml_cfg["bond_cutoff"])
RDF_CUTOFF         = float(ml_cfg["rdf_cutoff"])
RDF_NBINS          = int(ml_cfg["rdf_nbins"])
SOAP_CUTOFF        = float(ml_cfg["soap_cutoff"])
SOAP_N_MAX         = int(ml_cfg["soap_n_max"])
SOAP_L_MAX         = int(ml_cfg["soap_l_max"])
DIHEDRAL_NBINS     = int(ml_cfg["dihedral_nbins"])

GB_GEN_DIR = GPUMD_ROOT / "results" / config_name / "gb_generation"
RNEMD_DIR  = GPUMD_ROOT / "results" / config_name / "rnemd"
OUT_DIR    = GPUMD_ROOT / "results" / config_name / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / "dataset.csv"

# ---------------------------------------------------------------------------
# One-time SOAP initialisation
# ---------------------------------------------------------------------------

soap_descriptor = SOAP(
    species=["Si"],
    r_cut=SOAP_CUTOFF,
    n_max=SOAP_N_MAX,
    l_max=SOAP_L_MAX,
    average="off",   # we average manually so we can apply per-subset masks
    periodic=True,
)
SOAP_DIM = soap_descriptor.get_number_of_features()
print(f"SOAP feature dimension: {SOAP_DIM}")

# Subsets: (column_prefix, distance_threshold_or_None)
# None means no masking (use all atoms)
SUBSETS = [("full", None)] + [(f"gb{int(d)}", float(d)) for d in GB_DIST_THRESHOLDS]

# ---------------------------------------------------------------------------
# Collect unique GBs from config (yaml may list the same label multiple times)
# ---------------------------------------------------------------------------

seen_labels = set()
gb_entries = []
for entry in config["grain_boundaries"]:
    axis  = tuple(entry["axis"])
    sigma = entry["sigma"]
    plane = tuple(entry["plane"])
    label = make_gb_label(axis, sigma, plane)
    if label not in seen_labels:
        seen_labels.add(label)
        gb_entries.append((axis, sigma, plane, label))

print(f"\nConfig: {yaml_path.name}  |  {len(gb_entries)} unique GBs")

# ---------------------------------------------------------------------------
# Column ordering (applied consistently on every write)
# ---------------------------------------------------------------------------

META_COLS   = ["gb_label", "axis_x", "axis_y", "axis_z",
               "sigma", "plane_x", "plane_y", "plane_z"]
TARGET_COLS = ["R_K_mean", "R_K_std", "kappa_mean", "kappa_std", "J_mean", "J_std"]

# ---------------------------------------------------------------------------
# Main loop — write one row per GB as it completes
# ---------------------------------------------------------------------------

# Resume from existing file if present, otherwise start fresh
if OUT_PATH.exists():
    df_existing = pd.read_csv(OUT_PATH)
    already_done = set(df_existing["gb_label"].tolist())
    n_written = len(df_existing)
    print(f"Resuming from {OUT_PATH} ({n_written} rows already present)")
else:
    already_done = set()
    n_written = 0

for axis, sigma, plane, label in gb_entries:
    if label in already_done:
        print(f"  SKIP {label}: already in dataset")
        continue

    gb_gen_dir = GB_GEN_DIR / label

    # --- Find lowest-energy relaxed structure ---
    summary_csv = gb_gen_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"  SKIP {label}: gb_generation not done")
        continue

    df_summary = pd.read_csv(summary_csv)
    if df_summary.empty:
        print(f"  SKIP {label}: summary.csv is empty")
        continue

    best_run = int(df_summary.loc[df_summary["energy_ev"].idxmin(), "run_index"])
    traj_path = gb_gen_dir / f"run_{best_run}" / "structure.traj"
    if not traj_path.exists():
        print(f"  SKIP {label}: structure.traj missing for run_{best_run}")
        continue

    atoms = read(str(traj_path))
    print(f"  {label}: {len(atoms)} atoms, run_{best_run} (lowest E)")

    # --- Build row, starting with GB identity ---
    row = {
        "gb_label": label,
        "axis_x":   axis[0],
        "axis_y":   axis[1],
        "axis_z":   axis[2],
        "sigma":    sigma,
        "plane_x":  plane[0],
        "plane_y":  plane[1],
        "plane_z":  plane[2],
    }

    # --- Descriptors for each atom subset ---
    for prefix, threshold in SUBSETS:
        if threshold is None:
            mask = None
        else:
            mask = identify_gb_atoms(atoms, d_threshold=threshold)
            if mask.sum() == 0:
                warnings.warn(
                    f"{label}: no atoms within {threshold} Å of GB — falling back to all atoms"
                )
                mask = None

        # Global structural descriptors
        desc = compute_global_descriptors(
            atoms,
            bond_cutoff=BOND_CUTOFF,
            rdf_cutoff=RDF_CUTOFF,
            rdf_nbins=RDF_NBINS,
            dihedral_nbins=DIHEDRAL_NBINS,
            mask=mask,
        )
        row[f"{prefix}_sigma_theta"] = desc["sigma_theta"]
        row[f"{prefix}_sigma_l"]     = desc["sigma_l"]
        row[f"{prefix}_H_phi"]       = desc["H_phi"]
        row[f"{prefix}_A_RDF"]       = desc["A_RDF"]
        row[f"{prefix}_rho"]         = desc["rho"]
        row[f"{prefix}_n_atoms"]     = desc["n_atoms_used"]

        # Mean SOAP vector — one column per component
        mean_soap = compute_mean_soap(atoms, soap_descriptor, mask=mask)
        for i, v in enumerate(mean_soap):
            row[f"soap_{prefix}_{i}"] = v

    # --- rNEMD aggregate results (NaN if not yet run) ---
    agg_path = RNEMD_DIR / label / "aggregate.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path).iloc[0]
        for col in ["R_K_mean", "R_K_std", "kappa_mean", "kappa_std", "J_mean", "J_std"]:
            row[col] = agg.get(col, np.nan)
    else:
        for col in ["R_K_mean", "R_K_std", "kappa_mean", "kappa_std", "J_mean", "J_std"]:
            row[col] = np.nan

    # Enforce column order: metadata | descriptors | targets
    desc_cols = [c for c in row if c not in META_COLS and c not in TARGET_COLS]
    ordered_cols = META_COLS + desc_cols + TARGET_COLS
    df_row = pd.DataFrame([row])[ordered_cols]

    write_header = not OUT_PATH.exists()
    df_row.to_csv(OUT_PATH, mode="a", index=False, header=write_header)
    n_written += 1
    print(f"    -> written to {OUT_PATH} ({n_written} rows so far)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if n_written == 0:
    print("\nNo completed structures found — dataset is empty.")
else:
    df_final = pd.read_csv(OUT_PATH)
    print(f"\nDataset written to {OUT_PATH}")
    print(f"  {len(df_final)} rows x {len(df_final.columns)} columns")
    n_with_target = df_final["R_K_mean"].notna().sum()
    print(f"  {n_with_target}/{len(df_final)} rows have rNEMD target values")
