"""
format_cleaned_data.py — Build a supervised-learning dataset from GB structural
descriptors, GB energy, and rNEMD thermal results.

Usage:
    python ml/format_cleaned_data.py --config configs/full.yaml

For each unique grain boundary in the config:
  - Selects the lowest-energy relaxed structure from gb_generation.
  - Computes structural descriptors (sigma_theta, sigma_l, H_phi, A_RDF, rho)
    from that run's structure.traj for three atom subsets:
      "full"   — all atoms in the supercell
      "gb{d}"  — atoms within d Å of either GB plane (one per gb_dist_threshold)
  - Derives the GB macroscopic geometry from aimsgb: character (twist / tilt;
    symmetric vs asymmetric tilt), misorientation angle, and inclination angle.
  - Computes the GB energy by relaxing the supercell with CPUNEP + LBFGS
    (fmax=0.01) against a consistently relaxed bulk_si reference.
  - Joins with rNEMD aggregate results (R_K_mean, kappa_mean, …) where available,
    filling NaN for GBs not yet simulated.

Output:
    <results_dir>/<config_name>/dataset.csv

Column layout:
    [metadata]  gb_label, axis_x/y/z, sigma, plane_x/y/z,
                gb_character, tilt_type, misorientation_deg, inclination_deg
    [descriptors per subset]
        {prefix}_sigma_theta, _sigma_l, _H_phi, _A_RDF, _rho, _n_atoms
    [GB energy]  gamma_j_m2, gamma_ev_ang2, E_GB, N_GB, e_bulk, area_ang2
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

from utils.descriptors import (
    compute_global_descriptors,
    coordination_descriptors,
    steinhardt_descriptors,
    identify_gb_atoms,
)
from utils.gb_geometry import compute_gb_geometry
from utils.gb_energy import compute_gb_energy_relaxed, relaxed_bulk_energy_per_atom
from utils.work_coordination import gb_label as make_gb_label, resolve_results_base

# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Build supervised-learning dataset from GB descriptors + GB energy + rNEMD results"
)
parser.add_argument("--config", type=str, required=True,
                    help="Path to unified YAML config (e.g. configs/full.yaml)")
args = parser.parse_args()

yaml_path = Path(args.config).resolve()
config_name = yaml_path.stem

with open(yaml_path) as f:
    config = yaml.safe_load(f)

# `ml` section is optional; fall back to sensible defaults per key so the
# script runs against configs (e.g. full.yaml) that predate the ml block.
ml_cfg             = config.get("ml") or {}
GB_DIST_THRESHOLDS = list(ml_cfg.get("gb_dist_thresholds", [5.0, 10.0]))
BOND_CUTOFF        = float(ml_cfg.get("bond_cutoff", 3.0))
RDF_CUTOFF         = float(ml_cfg.get("rdf_cutoff", 10.0))
RDF_NBINS          = int(ml_cfg.get("rdf_nbins", 200))
DIHEDRAL_NBINS     = int(ml_cfg.get("dihedral_nbins", 36))

# GB-energy relaxation settings
NEP_MODEL_FILE = str(GPUMD_ROOT / config["nep_model"])
RELAX_FMAX     = float(ml_cfg.get("relax_fmax", 0.01))

RESULTS_BASE = resolve_results_base(config, GPUMD_ROOT) / config_name
GB_GEN_DIR   = RESULTS_BASE / "gb_generation"
RNEMD_DIR    = RESULTS_BASE / "rnemd"
BULK_DIR     = GB_GEN_DIR / "bulk_si"
OUT_DIR      = RESULTS_BASE
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = OUT_DIR / "dataset.csv"

# Subsets: (column_prefix, distance_threshold_or_None)
# None means no masking (use all atoms)
SUBSETS = [("full", None)] + [(f"gb{int(d)}", float(d)) for d in GB_DIST_THRESHOLDS]

# ---------------------------------------------------------------------------
# Bulk reference energy (relaxed once, shared by every GB energy calc)
# ---------------------------------------------------------------------------

e_bulk = None
if BULK_DIR.exists():
    try:
        e_bulk, e_bulk_std = relaxed_bulk_energy_per_atom(
            BULK_DIR, NEP_MODEL_FILE, fmax=RELAX_FMAX
        )
        print(f"Relaxed bulk reference: {e_bulk:.6f} ± {e_bulk_std:.6f} eV/atom")
    except FileNotFoundError as exc:
        warnings.warn(f"No bulk reference ({exc}); GB energies will be NaN")
else:
    warnings.warn(f"bulk_si not found at {BULK_DIR}; GB energies will be NaN")

# ---------------------------------------------------------------------------
# Collect unique GBs from config (yaml may list the same label multiple times)
# ---------------------------------------------------------------------------

seen_labels = set()
gb_entries = []
for entry in config["grain_boundaries"]:
    if entry["sigma"] == -1:
        continue  # bulk_si reference, not a GB
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
               "sigma", "plane_x", "plane_y", "plane_z",
               "gb_character", "tilt_type",
               "misorientation_deg", "inclination_deg"]
ENERGY_COLS = ["gamma_j_m2", "gamma_ev_ang2", "E_GB", "N_GB", "e_bulk", "area_ang2"]
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

    # --- Build row, starting with GB identity and macroscopic geometry ---
    geom = compute_gb_geometry(axis, sigma, plane)
    row = {
        "gb_label":           label,
        "axis_x":             axis[0],
        "axis_y":             axis[1],
        "axis_z":             axis[2],
        "sigma":              sigma,
        "plane_x":            plane[0],
        "plane_y":            plane[1],
        "plane_z":            plane[2],
        "gb_character":       geom["character"],
        "tilt_type":          geom["tilt_type"],
        "misorientation_deg": geom["misorientation_deg"],
        "inclination_deg":    geom["inclination_deg"],
    }

    # --- Descriptors for each atom subset (from structure.traj) ---
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

        # Coordination-defect statistics (under/over-coordinated Si in the
        # subset) -- a strong TBR predictor; see ml_pipeline.ipynb section 11.
        coord = coordination_descriptors(atoms, bond_cutoff=BOND_CUTOFF, mask=mask)
        row[f"{prefix}_coord_under"] = coord["coord_under"]
        row[f"{prefix}_coord_over"]  = coord["coord_over"]
        row[f"{prefix}_coord_mean"]  = coord["coord_mean"]
        row[f"{prefix}_coord_std"]   = coord["coord_std"]

        # Steinhardt bond-orientational order (q4, q6) over the subset -- the
        # strongest TBR predictor found so far; see ml_pipeline.ipynb sec 11b.
        bo = steinhardt_descriptors(atoms, bond_cutoff=BOND_CUTOFF, ls=(4, 6), mask=mask)
        for _l in (4, 6):
            row[f"{prefix}_q{_l}_mean"] = bo[f"q{_l}_mean"]
            row[f"{prefix}_q{_l}_std"]  = bo[f"q{_l}_std"]

    # --- GB energy (relax supercell with CPUNEP + LBFGS, then compute γ) ---
    if e_bulk is not None:
        gb = compute_gb_energy_relaxed(
            traj_path, e_bulk, NEP_MODEL_FILE, fmax=RELAX_FMAX
        )
        row["gamma_j_m2"]    = gb["gamma_j_m2"]
        row["gamma_ev_ang2"] = gb["gamma_ev_ang2"]
        row["E_GB"]          = gb["E_GB"]
        row["N_GB"]          = gb["N_GB"]
        row["e_bulk"]        = gb["e_bulk"]
        row["area_ang2"]     = gb["area_ang2"]
        print(f"    γ = {gb['gamma_j_m2']:.3f} J/m²  ({gb['gamma_ev_ang2']:.4f} eV/Å²)")
    else:
        for col in ENERGY_COLS:
            row[col] = np.nan

    # --- rNEMD aggregate results (NaN if not yet run) ---
    agg_path = RNEMD_DIR / label / "aggregate.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path).iloc[0]
        for col in TARGET_COLS:
            row[col] = agg.get(col, np.nan)
    else:
        for col in TARGET_COLS:
            row[col] = np.nan

    # Enforce column order: metadata | descriptors | energy | targets
    fixed = set(META_COLS) | set(ENERGY_COLS) | set(TARGET_COLS)
    desc_cols = [c for c in row if c not in fixed]
    ordered_cols = META_COLS + desc_cols + ENERGY_COLS + TARGET_COLS
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
