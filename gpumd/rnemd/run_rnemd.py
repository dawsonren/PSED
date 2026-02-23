"""
run_rnemd.py — Run Müller-Plathe rNEMD on relaxed GB structures and
compute Kapitza resistance (TBR) and bulk thermal conductivity (kappa).

Usage:
    python run_rnemd.py --config ../configs/small_box.yaml
    python run_rnemd.py --config ../configs/small_box.yaml --gb sigma5_2-10_001

Pipeline:
1. Load parameters from a unified YAML config (same file used by generate_gbs.py).
2. Scan results/<config_name>/gb_generation/ for GB types. For each GB type,
   select the lowest-energy run from summary.csv.
3. For the selected structure:
   a. Repeat the cell along Y/Z (scale_repeat) for cross-section convergence.
   b. Equilibrate briefly at temperature_k using NPT.
   c. Run n_cycles of Müller-Plathe rNEMD: each cycle runs steps_per_cycle
      MD steps, then swaps the hottest atom in the cold slab with the
      coldest atom in the hot slab (via utils/muller_plathe.py).
   d. Record bin temperatures and swapped velocity magnitudes each cycle.
4. After all cycles, compute:
   - Heat flux J from cumulative energy transferred across the swap planes.
   - Bulk kappa from Fourier's law using the temperature gradient in the
     bulk crystal regions (away from the GB).
   - TBR (Kapitza resistance) from the temperature discontinuity at the
     GB plane divided by J.
5. Write per-structure results and a summary.csv per GB type.

TBR derivation
--------------
TODO

"""

import os
import csv
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.visualize.plot import plot_atoms

from calorine.calculators import GPUNEP

# Import the two utility functions from utils/muller_plathe.py
# swap_velocities: swaps hottest atom in cold bin with coldest in hot bin,
#                  returns (v_hot_magnitude, v_cold_magnitude) for energy flux
# bin_atoms:       assigns atom indices to spatial bins along x, returns
#                  array of length NBINS where each element is an index array
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from utils.muller_plathe import swap_velocities, bin_atoms

# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run Müller-Plathe rNEMD on relaxed GB structures"
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to unified YAML config file (e.g. ../configs/small_box.yaml)"
)
parser.add_argument(
    "--gb", type=str, default=None,
    help="Process a specific GB label (e.g. sigma5_2-10_001). "
         "If omitted, all GB types in the results directory are processed."
)
args = parser.parse_args()

# Resolve paths: gpumd root is the parent of rnemd/
SCRIPT_DIR = Path(__file__).resolve().parent
GPUMD_ROOT = SCRIPT_DIR.parent

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

CONFIG_NAME = Path(args.config).stem  # e.g. "small_box"

# ---------------------------------------------------------------------------
# Load parameters from config
# ---------------------------------------------------------------------------

NEP_MODEL_FILE = str(GPUMD_ROOT / config["nep_model"])
GPUMD_EXEC     = os.path.expandvars(config["gpumd_exec"])

GB_RESULTS_DIR    = str(GPUMD_ROOT / "results" / CONFIG_NAME / "gb_generation")
RNEMD_RESULTS_DIR = str(GPUMD_ROOT / "results" / CONFIG_NAME / "rnemd")

rnemd_cfg = config["rnemd"]
TEMPERATURE_K    = rnemd_cfg["temperature_k"]
NBINS            = rnemd_cfg["nbins"]
COLD_BIN         = NBINS // 4
HOT_BIN          = 3 * NBINS // 4
STEPS_PER_CYCLE  = rnemd_cfg["steps_per_cycle"]
TIMESTEP_FS      = rnemd_cfg["timestep_fs"]
N_CYCLES         = rnemd_cfg["n_cycles"]
N_EQUILIBRATION_CYCLES = rnemd_cfg["n_equilibration_cycles"]
SCALE_REPEAT     = rnemd_cfg["scale_repeat"]

# ---------------------------------------------------------------------------
# Single rNEMD cycle
# ---------------------------------------------------------------------------

def run_one_cycle(atoms, run_dir, cycle_index):
    """
    Run STEPS_PER_CYCLE MD steps via GPUMD, read back velocities (calorine
    quirk: velocities are not returned directly — must read velocity.out),
    then return the updated atoms with correct velocities attached.

    The velocity unit correction (/0.098) accounts for a mismatch between
    GPUMD's internal velocity units and ASE's Å/fs convention.
    """
    pass


# ---------------------------------------------------------------------------
# TBR and kappa calculation
# ---------------------------------------------------------------------------

def compute_tbr_and_kappa(temps_avg, velocities_hc, bin_centers_angstrom,
                           cross_section_angstrom2, total_time_fs, n_atoms):
    """
    Compute Kapitza resistance (TBR) and bulk thermal conductivity from
    the converged average temperature profile and cumulative swap velocities.
    """
    pass


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_temperature_profile(temps_times, bin_centers, result, out_dir,
                              label, structure_index):
    """
    Plot the evolving and converged temperature profile with the linear fits
    used to extract ΔT and kappa. This is the primary sanity check for rNEMD.

    What to look for:
      - Converged profile: later cycles (darker colour) should overlap with
        the cumulative average, indicating steady state.
      - Clear linear regions on each side of the GB — curved profiles suggest
        the system hasn't equilibrated or the box is too short.
      - Visible discontinuity at x_GB: if there's no step, TBR is very small
        or the GB was not preserved (check RDF and atom positions).
    """
    pass


# ---------------------------------------------------------------------------
# Per-structure RNEMD runner
# ---------------------------------------------------------------------------

def run_rnemd_on_structure(atoms, structure_index, gb_label_str, out_dir):
    """
    Full rNEMD pipeline for a single relaxed structure:
      1. Repeat cell along Y/Z (scale_repeat).
      2. NVT equilibration at TEMPERATURE_K
      3. N_CYCLES of Müller-Plathe with velocity swapping.
      4. Compute TBR and kappa.
      5. Save outputs and diagnostic plot.

    Returns a dict of results for the summary CSV.
    """
    pass

# TODO : implement the above stubs

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_gb_type(gb_label_str):
    gb_dir = os.path.join(GB_RESULTS_DIR, gb_label_str)

    # Use summary.csv to find the run with the lowest energy
    summary_csv = os.path.join(gb_dir, "summary.csv")
    if not os.path.exists(summary_csv):
        print(f"  WARNING: no summary.csv found in {gb_dir}, skipping.")
        return
    df = pd.read_csv(summary_csv)
    best_run_index = int(df.loc[df["energy_ev"].idxmin(), "run_index"])
    best_energy    = df["energy_ev"].min()

    traj_path = os.path.join(gb_dir, f"run_{best_run_index}", "structure.traj")
    if not os.path.exists(traj_path):
        print(f"  WARNING: structure.traj not found for best run_{best_run_index} in {gb_dir}, skipping.")
        return

    atoms = read(traj_path)
    print(f"\n{'='*60}")
    print(f"Processing {gb_label_str}  (config: {CONFIG_NAME})")
    print(f"  using run_{best_run_index} (lowest E = {best_energy:.4f} eV)")
    print(f"  scale_repeat={SCALE_REPEAT} (Y/Z cross-section)")
    print(f"{'='*60}")

    out_base    = os.path.join(RNEMD_RESULTS_DIR, gb_label_str)
    summary_rows = []

    struct_dir = os.path.join(out_base, f"structure_{best_run_index}")
    print(f"\n--- Structure run_{best_run_index} (E={best_energy:.4f} eV) ---")

    row = run_rnemd_on_structure(atoms, best_run_index, gb_label_str, struct_dir)
    summary_rows.append(row)

    # --- Summary CSV for this GB type ---
    summary_path = os.path.join(out_base, "summary.csv")
    os.makedirs(out_base, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure_index", "run_index", "energy_ev",
            "R_K_SI", "kappa_SI", "J_SI", "delta_T", "n_atoms"
        ])
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSummary written to {summary_path}")
    _print_summary_table(summary_rows, gb_label_str)


def _print_summary_table(rows, label):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  {'struct':>6}  {'run':>4}  {'E [eV]':>10}  "
          f"{'R_K [K·m²/W]':>14}  {'κ [W/m/K]':>10}")
    print(f"{'─'*70}")
    for r in rows:
        print(f"  {r['structure_index']:>6}  {r['run_index']:>4}  "
              f"{r['energy_ev']:>10.4f}  "
              f"{r['R_K_SI']:>14.3e}  "
              f"{r['kappa_SI']:>10.2f}")
    print(f"{'─'*70}")


def main():
    if not os.path.exists(NEP_MODEL_FILE):
        raise FileNotFoundError(f"NEP model not found at '{NEP_MODEL_FILE}'.")

    os.makedirs(RNEMD_RESULTS_DIR, exist_ok=True)

    if args.gb:
        process_gb_type(args.gb)
    else:
        gb_dirs = sorted(glob.glob(os.path.join(GB_RESULTS_DIR, "sigma*")))
        if not gb_dirs:
            raise RuntimeError(f"No GB result folders found in {GB_RESULTS_DIR}.")
        for gb_dir in gb_dirs:
            process_gb_type(os.path.basename(gb_dir))


if __name__ == "__main__":
    main()
