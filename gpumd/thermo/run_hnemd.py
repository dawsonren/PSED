"""
run_hnemd.py — Homogeneous Non-Equilibrium MD (HNEMD) thermal conductivity
via GPUMD's compute_hnemd keyword.

Usage:
    python run_hnemd.py --config ../configs/no_gb.yaml
    python run_hnemd.py --config ../configs/no_gb.yaml --gb bulk_si

Pipeline:
1. Load parameters from a unified YAML config (same file used by generate_gbs.py).
2. Scan results/<config_name>/gb_generation/ for GB types. For each GB type,
   select the lowest-energy run from summary.csv.
3. For the selected structure, loop over N_RUNS (outer) x FORCING_VALUES (inner):
   - Each run gets fresh MB velocities, then the *same* initial state is used
     for every forcing value — this gives a controlled comparison between Fe
     values for linear-response verification.
   - The structure is assumed to already be equilibrated (by generate_gbs.py).
   a. Production using nvt_nhc + compute_hnemd (fixed volume, Nose-Hoover
      chain thermostat absorbs heat injected by the driving force).
   c. Parse kappa.out (running time-average of thermal conductivity tensor
      components) and extract the converged kappa_zz.
   d. Produce a kappa-over-time convergence plot per run.
4. Produce a kappa vs Fe_z plot across forcing values to verify linear response
   (kappa should be independent of Fe_z in the linear regime).
5. Aggregate results across runs (mean +/- std) per forcing value.
6. Write per-run summary.csv and aggregate.csv per GB type.

HNEMD method
------------
A small external driving force Fe is applied, biasing the heat current.
GPUMD directly outputs the running time-average of the thermal conductivity
tensor to kappa.out. Unlike rNEMD (which imposes a flux and measures dT/dx),
HNEMD imposes a force and measures the resulting heat current — no velocity
swapping or temperature-profile fitting is needed.

The nvt_nhc thermostat is required: the driving force continuously injects
energy, so the thermostat must remove it. Langevin thermostats cannot be used
because they disrupt the heat current.

kappa.out format (5 columns, all W/(m*K), when Fe along z):
  col 0: kappa_zx^in   col 1: kappa_zx^out
  col 2: kappa_zy^in   col 3: kappa_zy^out
  col 4: kappa_zz^total  <-- this is what we extract
"""

import os
import csv
import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# NOTE: suppress warnings from re-initializing calorine (weird quirk that Dawson Smith noticed)
warnings.filterwarnings("ignore", message=".*is not empty.*", module="calorine")

from calorine.calculators import GPUNEP

# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run HNEMD thermal conductivity on relaxed structures"
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to unified YAML config file (e.g. ../configs/no_gb.yaml)"
)
parser.add_argument(
    "--gb", type=str, default=None,
    help="Process a specific GB label (e.g. sigma5_2-10_001). "
         "If omitted, all GB types in the results directory are processed."
)
args = parser.parse_args()

# Resolve paths: gpumd root is the parent of thermo/
SCRIPT_DIR = Path(__file__).resolve().parent
GPUMD_ROOT = SCRIPT_DIR.parent

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

CONFIG_NAME = Path(args.config).stem  # e.g. "no_gb"

# ---------------------------------------------------------------------------
# Load parameters from config
# ---------------------------------------------------------------------------

NEP_MODEL_FILE = str(GPUMD_ROOT / config["nep_model"])
GPUMD_EXEC     = os.path.expandvars(config["gpumd_exec"])

GB_RESULTS_DIR    = str(GPUMD_ROOT / "results" / CONFIG_NAME / "gb_generation")
HNEMD_RESULTS_DIR = str(GPUMD_ROOT / "results" / CONFIG_NAME / "hnemd")

hnemd_cfg = config["hnemd"]

FORCING_VALUES     = list(hnemd_cfg["forcing_values"])
TIMESTEP_FS        = float(hnemd_cfg["timestep_fs"])
PRODUCTION_TIME_PS = float(hnemd_cfg["production_time_ps"])
OUTPUT_INTERVAL    = int(hnemd_cfg.get("output_interval", 1000))
N_RUNS             = int(hnemd_cfg.get("n_runs", 3))
TEMPERATURE_K      = float(hnemd_cfg["temperature_k"])

# nvt_nhc production thermostat coupling
TAU_T = float(hnemd_cfg["tau_t"])


# ---------------------------------------------------------------------------
# Single HNEMD simulation (production only — structure already equilibrated)
# ---------------------------------------------------------------------------

def run_hnemd_simulation(atoms, Fe_z, run_dir):
    """
    Run an HNEMD production simulation using nvt_nhc + compute_hnemd.

    The input structure is assumed to already be equilibrated (by
    generate_gbs.py), so no equilibration stage is needed here.

    Returns the atoms object after the simulation.
    """
    os.makedirs(run_dir, exist_ok=True)

    prod_steps = int(PRODUCTION_TIME_PS * 1000 / TIMESTEP_FS)

    md_params = [
        ("time_step", TIMESTEP_FS),
        ("dump_position", OUTPUT_INTERVAL),
        ("ensemble", ['nvt_nhc', TEMPERATURE_K, TEMPERATURE_K, TAU_T]),
        ("compute_hnemd", [OUTPUT_INTERVAL, 0, 0, Fe_z]),
        ("run", prod_steps),
    ]

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=run_dir,
        atoms=atoms,
    )

    result_atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    return result_atoms


# ---------------------------------------------------------------------------
# kappa.out parsing and convergence
# ---------------------------------------------------------------------------

def parse_kappa_out(run_dir):
    """
    Parse kappa.out from an HNEMD production run.

    Each row is a running time-average output every OUTPUT_INTERVAL steps.
    When the driving force is along z, column 4 is kappa_zz (total).

    Returns
    -------
    dict with 'time_ps' array and 'kappa_zz' array (both 1-D).
    """
    kappa_path = os.path.join(run_dir, "kappa.out")
    data = np.loadtxt(kappa_path)

    n_rows = len(data)
    t_ps = np.arange(1, n_rows + 1) * OUTPUT_INTERVAL * TIMESTEP_FS * 1e-3

    return {
        "time_ps": t_ps,
        "kappa_zz": data[:, 4],
        "raw": data,
    }


def check_kappa_convergence(kappa_zz, tolerance=0.05):
    """
    Check whether the kappa running average has plateaued.

    Compares the final value to the value at the simulation midpoint.
    If the relative change is below *tolerance*, the run is considered
    converged.

    Returns
    -------
    converged : bool
    rel_change : float
        |kappa_final - kappa_mid| / |kappa_final|
    """
    n = len(kappa_zz)
    if n < 10:
        return False, np.nan

    final = kappa_zz[-1]
    mid = kappa_zz[n // 2]
    rel_change = abs(final - mid) / abs(final) if abs(final) > 0 else np.nan
    return rel_change < tolerance, rel_change


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_kappa_convergence(kappa_data, Fe_z, run_dir, label, run_index):
    """
    Plot kappa_zz vs time for a single HNEMD run.

    What to look for:
      - The running average should settle into a clear plateau.
      - If it's still drifting at the end, production_time_ps should be
        increased.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kappa_data["time_ps"], kappa_data["kappa_zz"],
            color="steelblue", linewidth=1)

    final_kappa = kappa_data["kappa_zz"][-1]
    ax.axhline(final_kappa, color="tomato", linestyle="--", linewidth=1,
               label=f"final = {final_kappa:.2f} W/(m*K)")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("kappa_zz [W/(m*K)]")
    ax.set_title(f"{label} | Fe_z = {Fe_z:.1e} A^-1 | run {run_index}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "kappa_convergence.png"), dpi=150)
    plt.close()


def plot_linear_response(forcing_results, out_dir, label):
    """
    Plot kappa_zz vs Fe_z across all forcing values.

    In the linear response regime, kappa should be independent of Fe_z
    (flat line).  Departure at large Fe_z signals nonlinear effects.

    What to look for:
      - Flat plateau at smaller Fe_z values — these are in the linear regime.
      - Drop or rise at the largest Fe_z — exclude these from final results.
    """
    Fe_vals = []
    kappa_means = []
    kappa_stds = []

    for Fe_z in sorted(forcing_results):
        runs = forcing_results[Fe_z]
        kappas = [r["kappa_zz"] for r in runs]
        Fe_vals.append(Fe_z)
        kappa_means.append(np.mean(kappas))
        kappa_stds.append(np.std(kappas, ddof=1) if len(kappas) > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Fe_vals, kappa_means, yerr=kappa_stds,
                marker='o', capsize=5, color="steelblue", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Driving force Fe_z [1/A]")
    ax.set_ylabel("kappa_zz [W/(m*K)]")
    ax.set_title(f"{label} — Linear response check")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "linear_response.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-structure HNEMD runner
# ---------------------------------------------------------------------------

def run_hnemd_on_structure(atoms, structure_index, gb_label_str, out_dir):
    """
    Full HNEMD pipeline for a single relaxed structure.

    Loop order: N_RUNS (outer) x FORCING_VALUES (inner).  Each run gets
    fresh MB velocities, then the *same* initial state is used for every
    forcing value — this gives a controlled pairwise comparison between
    Fe values for linear-response verification.

    Returns (all_run_results, forcing_results) where all_run_results is a
    flat list of per-run dicts and forcing_results is
    {Fe_z: [list of per-run result dicts]}.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Structure: {len(atoms)} atoms")

    all_run_results = []
    forcing_results = {Fe_z: [] for Fe_z in FORCING_VALUES}

    for run_idx in range(N_RUNS):
        print(f"\n  === Run {run_idx + 1}/{N_RUNS} ===")

        # Fresh MB velocities — shared across all forcing values for this run
        run_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(run_atoms, temperature_K=TEMPERATURE_K)
        print(f"    Initial T = {run_atoms.get_temperature():.1f} K")

        # Convert ASE velocities (Å/t_ASE) to GPUMD units (Å/fs).
        # Calorine writes vel to model.xyz without converting, but GPUMD
        # reads vel as Å/fs.  Without this, velocities are ~10x too large.
        run_atoms.set_velocities(run_atoms.get_velocities() * units.fs)

        for Fe_z in FORCING_VALUES:
            fe_label = f"Fe_{Fe_z:.2e}"
            run_dir = os.path.join(out_dir, fe_label, f"run_{run_idx}")

            print(f"\n    --- Fe_z = {Fe_z:.1e} A^-1 ---")

            # Copy so each forcing value starts from the same initial state
            fe_atoms = run_atoms.copy()

            # Run production
            print(f"      Production ({PRODUCTION_TIME_PS} ps nvt_nhc)...")
            result_atoms = run_hnemd_simulation(fe_atoms, Fe_z, run_dir)

            # Parse kappa.out
            kappa_data = parse_kappa_out(run_dir)
            kappa_zz = float(kappa_data["kappa_zz"][-1])

            # Convergence check
            converged, rel_change = check_kappa_convergence(
                kappa_data["kappa_zz"])
            if converged:
                print(f"      kappa_zz = {kappa_zz:.2f} W/(m*K) — CONVERGED")
            else:
                print(f"      kappa_zz = {kappa_zz:.2f} W/(m*K) — "
                      f"NOT converged (rel change = {rel_change:.3f})")

            # Save final structure
            write(os.path.join(run_dir, "final_atoms.traj"), result_atoms)

            # Convergence plot
            plot_kappa_convergence(kappa_data, Fe_z, run_dir,
                                  gb_label_str, run_idx)

            result = {
                "structure_index": structure_index,
                "Fe_z": Fe_z,
                "run_index": run_idx,
                "kappa_zz": kappa_zz,
                "energy_ev": atoms.info.get("energy_ev", np.nan),
                "n_atoms": len(atoms),
                "converged": converged,
            }
            all_run_results.append(result)
            forcing_results[Fe_z].append(result)

    # Linear response plot across all forcing values
    plot_linear_response(forcing_results, out_dir, gb_label_str)

    # Aggregate per forcing value
    aggregate = {}
    for Fe_z in sorted(forcing_results):
        kappas = [r["kappa_zz"] for r in forcing_results[Fe_z]]
        n = len(kappas)
        ddof = 1 if n > 1 else 0
        aggregate[Fe_z] = {
            "Fe_z": Fe_z,
            "kappa_mean": float(np.nanmean(kappas)),
            "kappa_std": float(np.nanstd(kappas, ddof=ddof)),
            "n_runs": n,
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {gb_label_str} — HNEMD summary")
    for Fe_z in sorted(aggregate):
        agg = aggregate[Fe_z]
        print(f"  Fe_z = {Fe_z:.1e}: kappa = {agg['kappa_mean']:.2f} "
              f"+/- {agg['kappa_std']:.2f} W/(m*K) ({agg['n_runs']} runs)")
    print(f"{'='*60}")

    return all_run_results, aggregate


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_gb_type(gb_label_str):
    """Find the lowest-energy structure for a GB type and run HNEMD on it."""
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
        print(f"  WARNING: structure.traj not found for best run_{best_run_index} "
              f"in {gb_dir}, skipping.")
        return

    atoms = read(traj_path)

    print(f"\n{'='*60}")
    print(f"Processing {gb_label_str}  (config: {CONFIG_NAME})")
    print(f"  using run_{best_run_index} (lowest E = {best_energy:.4f} eV)")
    print(f"  forcing values: {FORCING_VALUES}")
    print(f"  n_runs per Fe: {N_RUNS}")
    print(f"{'='*60}")

    out_base = os.path.join(HNEMD_RESULTS_DIR, gb_label_str)
    struct_dir = os.path.join(out_base, f"structure_{best_run_index}")

    all_run_results, aggregate = run_hnemd_on_structure(
        atoms, best_run_index, gb_label_str, struct_dir
    )

    # --- Per-run summary CSV ---
    os.makedirs(out_base, exist_ok=True)

    summary_path = os.path.join(out_base, "summary.csv")
    summary_fields = [
        "structure_index", "Fe_z", "run_index", "kappa_zz",
        "energy_ev", "n_atoms", "converged",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_run_results)
    print(f"\nPer-run summary written to {summary_path}")

    # --- Aggregate CSV (one row per forcing value) ---
    agg_path = os.path.join(out_base, "aggregate.csv")
    agg_fields = ["Fe_z", "kappa_mean", "kappa_std", "n_runs"]
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        w.writeheader()
        for Fe_z in sorted(aggregate):
            w.writerow(aggregate[Fe_z])
    print(f"Aggregate summary written to {agg_path}")

    _print_summary_table(all_run_results, gb_label_str)


def _print_summary_table(rows, label):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  {'struct':>6}  {'Fe_z':>10}  {'run':>4}  "
          f"{'kappa_zz [W/m/K]':>18}  {'conv':>5}")
    print(f"{'─'*70}")
    for r in rows:
        print(f"  {r['structure_index']:>6}  {r['Fe_z']:>10.1e}  "
              f"{r['run_index']:>4}  "
              f"{r['kappa_zz']:>18.2f}  "
              f"{'yes' if r['converged'] else 'NO':>5}")
    print(f"{'─'*70}")


def main():
    if not os.path.exists(NEP_MODEL_FILE):
        raise FileNotFoundError(f"NEP model not found at '{NEP_MODEL_FILE}'.")

    os.makedirs(HNEMD_RESULTS_DIR, exist_ok=True)

    if args.gb:
        process_gb_type(args.gb)
    else:
        gb_dirs = sorted(
            glob.glob(os.path.join(GB_RESULTS_DIR, "sigma*")) +
            glob.glob(os.path.join(GB_RESULTS_DIR, "bulk_si"))
        )
        if not gb_dirs:
            raise RuntimeError(f"No GB result folders found in {GB_RESULTS_DIR}.")
        for gb_dir in gb_dirs:
            process_gb_type(os.path.basename(gb_dir))


if __name__ == "__main__":
    main()
