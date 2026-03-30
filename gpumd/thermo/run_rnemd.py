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
   a. Run N_RUNS independent rNEMD simulations, each with fresh MB velocities:
      i.   Run n_cycles of Müller-Plathe rNEMD: each cycle runs steps_per_cycle
           MD steps, then swaps the hottest atom in the cold slab with the
           coldest atom in the hot slab (via utils/muller_plathe.py).
      iii. Record bin temperatures and swapped velocity magnitudes each cycle.
   b. After all cycles, compute per-run TBR, kappa, and heat flux J.
4. Aggregate results across runs (mean ± std) for uncertainty estimation.
5. Write per-run summary.csv and aggregate.csv per GB type.

TBR derivation
--------------
Heat flux:  J = Σ(m/2)(v_hot² - v_cold²) / (2·A·t)
  - Factor of 2 in denominator: heat flows both directions in periodic box.
  - v_hot, v_cold are the swapped atom speeds from swap_velocities (ASE units).
  - m = Si atomic mass.

Bulk kappa: κ = |J / (dT/dx)|
  - Linear fit to bulk crystal regions between cold/hot bins and GB midpoint.

Kapitza resistance: R_K = ΔT_GB / J
  - ΔT_GB from extrapolating left/right bulk fits to the GB plane.
"""

import os
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.visualize.plot import plot_atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell

# NOTE: suppress warnings from re-initializing calorine (weird quirk that Dawson Smith noticed)
# see note in run_one_cycle()
warnings.filterwarnings("ignore", message=".*is not empty.*", module="calorine")

from calorine.calculators import GPUNEP

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.muller_plathe import swap_velocities, bin_atoms
from utils.rnemd_stats import check_steady_state, aggregate_run_results, format_result_summary
from utils.rnemd_plots import plot_temperature_profile, plot_temperature_profile_animated
from utils.work_coordination import gb_label

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
NBINS            = int(rnemd_cfg["nbins"])
COLD_BIN         = NBINS // 4
HOT_BIN          = 3 * NBINS // 4
STEPS_PER_CYCLE  = int(rnemd_cfg["steps_per_cycle"])
TIMESTEP_FS      = float(rnemd_cfg["timestep_fs"])
N_CYCLES         = int(rnemd_cfg["n_cycles"])
N_RUNS           = int(rnemd_cfg.get("n_runs", 3))
ENSEMBLE         = rnemd_cfg.get("ensemble", "npt_scr").lower()
TEMPERATURE_K    = float(rnemd_cfg["temperature_k"])
if ENSEMBLE == "npt_scr":
    TAU_T            = float(rnemd_cfg["tau_t"])
    PRESSURE_GPA     = float(rnemd_cfg["pressure_gpa"])
    BULK_MODULUS_GPA = float(rnemd_cfg["bulk_modulus_gpa"])
    TAU_P            = float(rnemd_cfg["tau_p"])
assert ENSEMBLE in ["npt_scr", "nve"], f"Unsupported ensemble: {ENSEMBLE}"
DEBUG_STRUCTURE   = bool(rnemd_cfg.get("debug_structure", False))
DEBUG_DIAGNOSTICS = bool(rnemd_cfg.get("debug_diagnostics", True))
INCLUDE_MOVIE     = bool(rnemd_cfg.get("include_movie", False))
INCLUDE_ANIMATION = bool(rnemd_cfg.get("debug_animation", False))
N_WARMUP_CYCLES   = int(rnemd_cfg.get("n_warmup_cycles", 0))

# GB list from YAML (used in main() to restrict processing to configured GBs only)
BULK_SI_LABEL = "bulk_si"
_raw_gbs = config["grain_boundaries"]
NO_GB_MODE = len(_raw_gbs) == 1 and _raw_gbs[0].get("sigma") == -1
GB_LIST = [] if NO_GB_MODE else [
    (tuple(entry["axis"]), int(entry["sigma"]), tuple(entry["plane"]))
    for entry in _raw_gbs
]

# Si atomic mass in amu (used for energy flux calculation)
M_SI_AMU = 28.085

# ---------------------------------------------------------------------------
# Single rNEMD cycle
# ---------------------------------------------------------------------------

def run_one_cycle(atoms, run_dir):
    """
    Run STEPS_PER_CYCLE MD steps via GPUMD, read back velocities, and return
    the updated atoms with correct velocities attached.

    Calorine quirk: velocities are not returned by run_custom_md — they must
    be read from velocity.out.  The division by ~0.098 converts from GPUMD's
    internal velocity units (Å/fs) to ASE's internal units (Å/t_ASE where
    t_ASE ≈ 10.18 fs ≈ sqrt(amu·Å²/eV)).  The exact factor is ase.units.fs.
    """
    if ENSEMBLE == "npt_scr":
        ensemble_params = ['npt_scr', TEMPERATURE_K, TEMPERATURE_K, TAU_T, PRESSURE_GPA, BULK_MODULUS_GPA, TAU_P]
    elif ENSEMBLE == "nve":
        ensemble_params = ['nve']

    md_params = [
        ("dump_position", STEPS_PER_CYCLE),
        ("dump_velocity", STEPS_PER_CYCLE),
        ('dump_exyz', [STEPS_PER_CYCLE, 1]),
        ("time_step", TIMESTEP_FS),
        ("ensemble", ensemble_params),
        ("run", STEPS_PER_CYCLE),
    ]

    # Convert ASE velocities (Å/t_ASE) to GPUMD units (Å/fs).
    # Calorine writes vel to model.xyz without converting, but GPUMD
    # reads vel as Å/fs.  Without this, velocities are ~10x too large.
    atoms.set_velocities(atoms.get_velocities() * units.fs)

    # Remove stale movie.xyz before each cycle: GPUMD appends rather than
    # overwrites, so a leftover file from an interrupted run corrupts reads.
    movie_path = os.path.join(run_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    # NOTE: Must re-create calculator each cycle (calorine limitation)
    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=run_dir,
        atoms=atoms,
    )

    atoms = calc.run_custom_md(md_params, return_last_atoms=True)

    # Read velocities from GPUMD output (last len(atoms) lines)
    vel_path = os.path.join(run_dir, "velocity.out")
    vels = pd.read_csv(vel_path, sep=" ", header=None).iloc[-len(atoms):, :]
    atoms.set_velocities(vels.values / units.fs)  # GPUMD (Å/fs) -> ASE units

    # At the end of run_one_cycle, after reading velocities
    # this prevents us from having output files that get longer and longer!
    files_to_remove = ["velocity.out", "position.out", "dump.xyz"]
    if not INCLUDE_MOVIE:
        files_to_remove.append("movie.xyz")
    for fname in files_to_remove:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    return atoms


# ---------------------------------------------------------------------------
# TBR and kappa calculation
# ---------------------------------------------------------------------------

def compute_tbr_and_kappa(temps_avg, velocities_hc, bin_centers_angstrom,
                           cross_section_angstrom2, total_time_fs):
    """
    Compute Kapitza resistance (TBR) and bulk thermal conductivity from
    the converged average temperature profile and cumulative swap velocities.

    Parameters
    ----------
    temps_avg : ndarray, shape (NBINS,)
        Converged (cumulative-average) temperature in each bin [K].
    velocities_hc : ndarray, shape (N_CYCLES, 2)
        Per-cycle swapped velocity magnitudes [v_hot, v_cold] in ASE units.
    bin_centers_angstrom : ndarray, shape (NBINS,)
        Bin center positions along x [Å].
    cross_section_angstrom2 : float
        Y*Z cross-section area [Å²].
    total_time_fs : float
        Total production simulation time [fs].

    Returns
    -------
    dict with R_K_SI, kappa_SI, J_SI, delta_T, dTdx_K_per_m.
    """
    # --- Heat flux J ---
    # Energy transferred per swap: ΔKE = (m/2)(v_hot² - v_cold²)
    # swap_velocities returns speeds in ASE units; 0.5 * m_amu * v_ase² = KE [eV]
    v_hot = velocities_hc[:, 0]   # ASE velocity units
    v_cold = velocities_hc[:, 1]
    delta_KE_eV = 0.5 * M_SI_AMU * (v_hot**2 - v_cold**2)  # eV per swap
    total_energy_eV = np.sum(delta_KE_eV)
    total_energy_J = total_energy_eV * 1.602176634e-19  # eV -> J

    A_m2 = cross_section_angstrom2 * 1e-20  # Å² -> m²
    t_s = total_time_fs * 1e-15              # fs -> s

    # Factor of 2: heat flows in both directions from hot slab in periodic box
    J = total_energy_J / (2.0 * A_m2 * t_s)  # W/m²

    # --- Linear fits for dT/dx and ΔT at GB ---
    # GB is at the midpoint (bin NBINS//2).  Fit left bulk (cold_bin -> GB)
    # and right bulk (GB -> hot_bin), excluding 1 bin margin near swap bins.
    margin = 1
    # NOTE: this is only valid when UC_A == UC_B, if we want twin boundaries then this doesn't work!
    gb_bin = NBINS // 2
    left_slice = slice(COLD_BIN + margin, gb_bin)
    right_slice = slice(gb_bin, HOT_BIN - margin)

    x_left = bin_centers_angstrom[left_slice]
    T_left = temps_avg[left_slice]
    x_right = bin_centers_angstrom[right_slice]
    T_right = temps_avg[right_slice]

    left_fit = np.polyfit(x_left, T_left, 1)    # [slope, intercept]
    right_fit = np.polyfit(x_right, T_right, 1)

    # Average slope for kappa (both sides should agree for a symmetric system)
    avg_slope = (left_fit[0] + right_fit[0]) / 2.0  # K/Å
    dTdx_SI = avg_slope * 1e10  # K/Å -> K/m

    kappa = abs(J / dTdx_SI) if abs(dTdx_SI) > 0 else np.nan  # W/(m·K)

    # TBR: extrapolate left and right fits to the GB position
    x_gb = bin_centers_angstrom[gb_bin]
    T_left_at_gb = np.polyval(left_fit, x_gb)
    T_right_at_gb = np.polyval(right_fit, x_gb)
    delta_T = abs(T_left_at_gb - T_right_at_gb)

    R_K = delta_T / J if J > 0 else np.nan  # K·m²/W

    return {
        "R_K_SI": R_K,
        "kappa_SI": kappa,
        "J_SI": J,
        "delta_T": delta_T,
        "dTdx_K_per_m": dTdx_SI,
        "left_fit": left_fit,
        "right_fit": right_fit,
    }


# ---------------------------------------------------------------------------
# Per-structure RNEMD runner
# ---------------------------------------------------------------------------

def run_rnemd_on_structure(atoms, structure_index, gb_label_str, out_dir):
    """
    Full rNEMD pipeline for a single relaxed structure with N_RUNS independent
    simulations for uncertainty estimation.

    Steps per run:
      1. Copy atoms, assign fresh Maxwell-Boltzmann velocities.
      2. Bin atoms along z-axis.
      3. Production: N_CYCLES of Müller-Plathe rNEMD with velocity swapping.
      4. Check steady-state convergence.
      5. Compute TBR and kappa.
      6. Save raw data and diagnostic plot.

    Returns (all_run_results, aggregate) where all_run_results is a list of
    per-run dicts and aggregate is the mean ± std summary.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find already-completed runs (have final_atoms.traj).
    # N_RUNS is the target total; only add what is still needed.
    existing_run_indices = sorted([
        int(d[4:]) for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
        and d.startswith("run_")
        and os.path.exists(os.path.join(out_dir, d, "final_atoms.traj"))
    ])
    n_existing = len(existing_run_indices)
    runs_to_add = N_RUNS - n_existing
    next_run_idx = (max(existing_run_indices) + 1) if existing_run_indices else 0

    if runs_to_add <= 0:
        print(f"  Already have {n_existing} completed run(s) (target={N_RUNS}), skipping.")
        return [], {}
    if n_existing > 0:
        print(f"  Have {n_existing} completed run(s), adding {runs_to_add} more "
              f"(run_{next_run_idx} onward) to reach target of {N_RUNS}.")

    # Cell geometry: aimsgb with direction=0 stacks grains along ASE cell axis 2.
    # Axes 0 and 1 are the (repeated) cross-section directions.
    stacking_len = np.linalg.norm(atoms.cell[2])          # Å, GB-normal direction
    cross_section = np.linalg.norm(                        # Å², area perpendicular to stacking
        np.cross(atoms.cell[0], atoms.cell[1])
    )
    print(f"  Structure: {len(atoms)} atoms, "
          f"stacking length = {stacking_len:.1f} Å, "
          f"cross-section = {cross_section:.1f} Å²")

    # Bin edges and centers along the stacking direction (axis 0)
    bins = np.linspace(0, 1, NBINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0 * stacking_len  # Å
    total_time_fs = N_CYCLES * STEPS_PER_CYCLE * TIMESTEP_FS

    all_run_results = []

    for run_idx in range(next_run_idx, next_run_idx + runs_to_add):
        run_dir = os.path.join(out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n  --- rNEMD run {run_idx + 1}/{N_RUNS} ---")

        # Fresh MB velocities for statistical independence
        run_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(run_atoms, temperature_K=TEMPERATURE_K)
        print(f"    Initial T = {run_atoms.get_temperature():.1f} K")

        # Bin atoms along the stacking direction (ASE cell axis 2)
        scaled_z = [a.scaled_position[2] for a in run_atoms]
        binned = bin_atoms(bins, scaled_z)

        # Warmup phase: build up the temperature gradient before recording data.
        if N_WARMUP_CYCLES > 0:
            print(f"    Warmup ({N_WARMUP_CYCLES} cycles)...")
            for _ in tqdm(range(N_WARMUP_CYCLES), desc="      warmup"):
                run_atoms = run_one_cycle(run_atoms, run_dir)
                swap_velocities(run_atoms, binned[COLD_BIN], binned[HOT_BIN])
            print(f"    Warmup done. T = {run_atoms.get_temperature():.1f} K")

        # Save bin visualization
        if DEBUG_STRUCTURE:
            fig, ax = plt.subplots(figsize=(10, 4))
            colorlist = np.empty(len(run_atoms), dtype="object")
            for b_idx, atom_indices in enumerate(binned):
                if b_idx == HOT_BIN:
                    colorlist[atom_indices] = "red"
                elif b_idx == COLD_BIN:
                    colorlist[atom_indices] = "blue"
                else:
                    colorlist[atom_indices] = "grey"
            plot_atoms(run_atoms, ax, colors=colorlist, rotation="10x,10y,0z")
            ax.set_title(f"{gb_label_str} run {run_idx} — bin assignment (blue=cold, red=hot)")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "bin_setup.png"), dpi=100)
            plt.close()

        # Production rNEMD cycles (structure already equilibrated by generate_gbs.py)
        print(f"    Production ({N_CYCLES} cycles)...")
        temps_times = np.zeros((N_CYCLES, NBINS))
        velocities_hc = np.zeros((N_CYCLES, 2))

        # Open CSV for incremental bin temperature logging
        bin_temps_csv_path = os.path.join(run_dir, "bin_temps.csv")
        with open(bin_temps_csv_path, "w", newline="") as f_csv:
            csv.writer(f_csv).writerow(["cycle"] + [f"bin_{i}" for i in range(NBINS)])

        for cycle in (pbar := tqdm(range(N_CYCLES))):
            run_atoms = run_one_cycle(run_atoms, run_dir)

            # Müller-Plathe velocity swap
            v_hot, v_cold = swap_velocities(
                run_atoms, binned[COLD_BIN], binned[HOT_BIN]
            )
            velocities_hc[cycle] = [v_hot, v_cold]

            # Record bin temperatures
            for b_idx, atom_indices in enumerate(binned):
                temps_times[cycle, b_idx] = run_atoms[atom_indices].get_temperature()

            # Append this cycle's bin temps to CSV
            with open(bin_temps_csv_path, "a", newline="") as f_csv:
                csv.writer(f_csv).writerow([cycle] + list(temps_times[cycle]))

            pbar.set_description(f"      cycle {cycle + 1}/{N_CYCLES}, T = {run_atoms.get_temperature():.1f} K")

        # Save raw data
        np.save(os.path.join(run_dir, "temps_times.npy"), temps_times)
        np.save(os.path.join(run_dir, "velocities_hc.npy"), velocities_hc)
        np.save(os.path.join(run_dir, "bin_centers.npy"), bin_centers)
        write(os.path.join(run_dir, "final_atoms.traj"), run_atoms)

        # Steady-state check
        converged, max_dev, _ = check_steady_state(temps_times)
        if not converged:
            print(f"    WARNING: may not have reached steady state "
                  f"(max T deviation = {max_dev:.1f} K between windows)")
        else:
            print(f"    Steady-state check passed (max dev = {max_dev:.1f} K)")

        # Compute TBR and kappa from cumulative average
        cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, N_CYCLES + 1)[:, None]
        temps_avg = cumulative_avg[-1]

        result = compute_tbr_and_kappa(
            temps_avg, velocities_hc, bin_centers,
            cross_section, total_time_fs,
        )
        result.update({
            "structure_index": structure_index,
            "run_index": run_idx,
            "energy_ev": atoms.info.get("energy_ev", np.nan),
            "n_atoms": len(run_atoms),
            "converged": converged,
        })
        all_run_results.append(result)

        print(f"    κ = {result['kappa_SI']:.2f} W/(m·K), "
              f"R_K = {result['R_K_SI']:.3e} K·m²/W, "
              f"J = {result['J_SI']:.3e} W/m²")

        # Diagnostic plot
        if DEBUG_DIAGNOSTICS:
            plot_temperature_profile(
                temps_times, bin_centers, result, run_dir,
                gb_label_str, run_idx, converged, max_dev,
                cold_bin=COLD_BIN, hot_bin=HOT_BIN, nbins=NBINS,
            )
        if INCLUDE_ANIMATION:
            plot_temperature_profile_animated(
                temps_times, bin_centers, run_dir,
                gb_label_str, run_idx,
                cold_bin=COLD_BIN, hot_bin=HOT_BIN, nbins=NBINS,
            )

    # Aggregate across runs
    aggregate = aggregate_run_results(all_run_results)
    print(format_result_summary(aggregate, gb_label_str))

    return all_run_results, aggregate


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
        print(f"  WARNING: structure.traj not found for best run_{best_run_index} "
              f"in {gb_dir}, skipping.")
        return

    atoms = read(traj_path)
    print(f"\n{'='*60}")
    print(f"Processing {gb_label_str}  (config: {CONFIG_NAME})")
    print(f"  using run_{best_run_index} (lowest E = {best_energy:.4f} eV)")
    print(f"  n_runs={N_RUNS}")
    print(f"{'='*60}")

    out_base = os.path.join(RNEMD_RESULTS_DIR, gb_label_str)
    struct_dir = os.path.join(out_base, f"structure_{best_run_index}")
    print(f"\n--- Structure run_{best_run_index} (E={best_energy:.4f} eV) ---")

    all_run_results, _ = run_rnemd_on_structure(
        atoms, best_run_index, gb_label_str, struct_dir
    )

    # --- Per-run summary CSV (append to existing rows if present) ---
    os.makedirs(out_base, exist_ok=True)

    summary_path = os.path.join(out_base, "summary.csv")
    summary_fields = [
        "structure_index", "run_index", "energy_ev",
        "R_K_SI", "kappa_SI", "J_SI", "delta_T", "n_atoms", "converged",
    ]
    existing_rows = []
    if os.path.exists(summary_path):
        existing_rows = pd.read_csv(summary_path).to_dict("records")

    if all_run_results:
        open_mode = "a" if existing_rows else "w"
        with open(summary_path, open_mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
            if not existing_rows:
                w.writeheader()
            w.writerows(all_run_results)
        print(f"\nPer-run summary written to {summary_path}")

    # --- Aggregate CSV (recomputed from all runs: existing + new) ---
    existing_for_agg = [
        {k: float(r[k]) for k in ("kappa_SI", "R_K_SI", "J_SI")}
        for r in existing_rows
    ]
    aggregate = aggregate_run_results(existing_for_agg + all_run_results)

    agg_path = os.path.join(out_base, "aggregate.csv")
    agg_fields = [
        "structure_index", "n_runs",
        "kappa_mean", "kappa_std", "R_K_mean", "R_K_std", "J_mean", "J_std",
    ]
    agg_row = {
        "structure_index": best_run_index,
        **aggregate,
    }
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(agg_row)
    print(f"Aggregate summary written to {agg_path} (n={aggregate['n_runs']} total runs)")

    if all_run_results:
        _print_summary_table(all_run_results, gb_label_str)


def _print_summary_table(rows, label):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  {'struct':>6}  {'run':>4}  {'E [eV]':>10}  "
          f"{'R_K [K·m²/W]':>14}  {'κ [W/m/K]':>10}  {'conv':>5}")
    print(f"{'─'*70}")
    for r in rows:
        print(f"  {r['structure_index']:>6}  {r['run_index']:>4}  "
              f"{r['energy_ev']:>10.4f}  "
              f"{r['R_K_SI']:>14.3e}  "
              f"{r['kappa_SI']:>10.2f}  "
              f"{'yes' if r['converged'] else 'NO':>5}")
    print(f"{'─'*70}")


def main():
    if not os.path.exists(NEP_MODEL_FILE):
        raise FileNotFoundError(f"NEP model not found at '{NEP_MODEL_FILE}'.")

    os.makedirs(RNEMD_RESULTS_DIR, exist_ok=True)

    if args.gb:
        process_gb_type(args.gb)
    elif NO_GB_MODE:
        process_gb_type(BULK_SI_LABEL)
    else:
        if not GB_LIST:
            raise RuntimeError("No grain boundaries defined in config.")
        for (axis, sigma, plane) in GB_LIST:
            process_gb_type(gb_label(axis, sigma, plane))


if __name__ == "__main__":
    main()