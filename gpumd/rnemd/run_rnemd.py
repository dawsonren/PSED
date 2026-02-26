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
      i.   Equilibrate for n_equilibration_cycles using NPT (no swapping).
      ii.  Run n_cycles of Müller-Plathe rNEMD: each cycle runs steps_per_cycle
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
import glob
import warnings
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
N_RUNS           = rnemd_cfg.get("n_runs", 3)
TAU_T            = rnemd_cfg["tau_t"]
PRESSURE_GPA     = rnemd_cfg["pressure_gpa"]
BULK_MODULUS_GPA = rnemd_cfg["bulk_modulus_gpa"]
TAU_P            = rnemd_cfg["tau_p"]

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
    md_params = [
        ("dump_position", STEPS_PER_CYCLE),
        ("dump_velocity", STEPS_PER_CYCLE),
        ('dump_exyz', [STEPS_PER_CYCLE, 1]),
        ("time_step", TIMESTEP_FS),
        ("ensemble", ['npt_scr', TEMPERATURE_K, TEMPERATURE_K, TAU_T, PRESSURE_GPA, BULK_MODULUS_GPA, TAU_P]),
        ("run", STEPS_PER_CYCLE),
    ]

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
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_temperature_profile(temps_times, bin_centers, result, out_dir,
                              label, run_index, converged, max_dev):
    """
    Plot the evolving and converged temperature profile with the linear fits
    used to extract ΔT and kappa.

    What to look for:
      - Converged profile: later cycles (darker colour) should overlap with
        the cumulative average, indicating steady state.
      - Clear linear regions on each side of the GB — curved profiles suggest
        the system hasn't equilibrated or the box is too short.
      - Visible discontinuity at x_GB: if there's no step, TBR is very small
        or the GB was not preserved (check RDF and atom positions).
    """
    n_cycles = len(temps_times)
    cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, n_cycles + 1)[:, None]

    cmap = cm.Oranges
    norm = Normalize(vmin=0, vmax=n_cycles)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.35)
    fig.suptitle(f"{label} — run {run_index}", fontsize=12)

    # Panel 1: Per-cycle temperature profiles
    for i, cycle_temps in enumerate(temps_times):
        axes[0].plot(bin_centers, cycle_temps, marker="o", markersize=2,
                     linewidth=0.8, color=cmap(norm(i)), alpha=0.7)
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Per-cycle temperature profiles (light→dark = early→late)")
    axes[0].axvline(bin_centers[COLD_BIN], color="blue", linestyle="--",
                    linewidth=0.8, label="cold bin")
    axes[0].axvline(bin_centers[HOT_BIN], color="red", linestyle="--",
                    linewidth=0.8, label="hot bin")
    axes[0].legend(fontsize=8)

    # Panel 2: Cumulative average + linear fits
    for i, avg in enumerate(cumulative_avg):
        axes[1].plot(bin_centers, avg, marker="o", markersize=2,
                     linewidth=0.8, color=cmap(norm(i)))

    # Overlay final linear fits
    left_fit = result["left_fit"]
    right_fit = result["right_fit"]
    gb_bin = NBINS // 2
    margin = 1

    x_left = bin_centers[COLD_BIN + margin : gb_bin]
    x_right = bin_centers[gb_bin : HOT_BIN - margin]
    axes[1].plot(x_left, np.polyval(left_fit, x_left), color="blue",
                 linewidth=2, linestyle="--", label="left bulk fit")
    axes[1].plot(x_right, np.polyval(right_fit, x_right), color="red",
                 linewidth=2, linestyle="--", label="right bulk fit")

    # Mark ΔT at GB
    x_gb = bin_centers[gb_bin]
    T_l = np.polyval(left_fit, x_gb)
    T_r = np.polyval(right_fit, x_gb)
    axes[1].annotate(
        f"ΔT = {result['delta_T']:.1f} K",
        xy=(x_gb, (T_l + T_r) / 2), fontsize=9,
        arrowprops=dict(arrowstyle="->"), xytext=(x_gb + 5, (T_l + T_r) / 2 + 20),
    )
    axes[1].axvline(x_gb, color="green", linestyle=":", linewidth=0.8, label="GB plane")
    axes[1].set_ylabel("Cumulative avg T [K]")
    axes[1].set_title(
        f"Converged profile — κ = {result['kappa_SI']:.2f} W/(m·K), "
        f"R_K = {result['R_K_SI']:.3e} K·m²/W"
    )
    axes[1].legend(fontsize=8)

    # Panel 3: Steady-state convergence
    window = max(int(n_cycles * 0.25), 1)
    cycle_indices = np.arange(n_cycles)
    per_cycle_mean_T = np.mean(temps_times, axis=1)  # mean T across bins per cycle
    axes[2].plot(cycle_indices, per_cycle_mean_T, color="tomato", linewidth=0.8)
    axes[2].axhline(np.mean(per_cycle_mean_T[-window:]), color="steelblue",
                    linestyle="--", linewidth=1.5, label=f"last {window} cycle avg")
    conv_str = "CONVERGED" if converged else f"NOT converged (max dev = {max_dev:.1f} K)"
    axes[2].set_title(f"Steady-state check: {conv_str}", fontsize=10)
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("Mean bin T [K]")
    axes[2].legend(fontsize=8)

    plt.savefig(os.path.join(out_dir, "temperature_profile.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-structure RNEMD runner
# ---------------------------------------------------------------------------

def run_rnemd_on_structure(atoms, structure_index, gb_label_str, out_dir):
    """
    Full rNEMD pipeline for a single relaxed structure with N_RUNS independent
    simulations for uncertainty estimation.

    Steps per run:
      1. Copy atoms, assign fresh Maxwell-Boltzmann velocities.
      2. Bin atoms along x-axis.
      3. Equilibrate (N_EQUILIBRATION_CYCLES of NPT, no swapping).
      4. Production: N_CYCLES of Müller-Plathe rNEMD with velocity swapping.
      5. Check steady-state convergence.
      6. Compute TBR and kappa.
      7. Save raw data and diagnostic plot.

    Returns (all_run_results, aggregate) where all_run_results is a list of
    per-run dicts and aggregate is the mean ± std summary.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Cell geometry: aimsgb with direction=0 stacks grains along ASE cell axis 2.
    # Axes 0 and 1 are the (repeated) cross-section directions.
    stacking_len = np.linalg.norm(atoms.cell[2])          # Å, GB-normal direction
    cross_section = np.linalg.norm(                        # Å², area perpendicular to stacking
        np.cross(atoms.cell[0], atoms.cell[1])
    )
    print(f"  Structure: {len(atoms)} atoms, "
          f"stacking length = {stacking_len:.1f} Å, "
          f"cross-section = {cross_section:.1f} Å²")

    # Bin edges and centers along the stacking direction (axis 2)
    bins = np.linspace(0, 1, NBINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0 * stacking_len  # Å
    total_time_fs = N_CYCLES * STEPS_PER_CYCLE * TIMESTEP_FS

    all_run_results = []

    for run_idx in range(N_RUNS):
        run_dir = os.path.join(out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n  --- rNEMD run {run_idx + 1}/{N_RUNS} ---")

        # Fresh MB velocities for statistical independence
        run_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(run_atoms, temperature_K=TEMPERATURE_K)
        print(f"    Initial T = {run_atoms.get_temperature():.1f} K")

        # Bin atoms along the stacking direction (ASE cell axis 2)
        scaled_x = [a.scaled_position[2] for a in run_atoms]
        binned = bin_atoms(bins, scaled_x)

        # Save bin visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        colorlist = np.empty(len(run_atoms), dtype="object")
        for b_idx, atom_indices in enumerate(binned):
            if b_idx == HOT_BIN:
                colorlist[atom_indices] = "red"
            elif b_idx == COLD_BIN:
                colorlist[atom_indices] = "blue"
            else:
                colorlist[atom_indices] = "grey"
        plot_atoms(run_atoms, ax, colors=colorlist)
        ax.set_title(f"{gb_label_str} run {run_idx} — bin assignment (blue=cold, red=hot)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "bin_setup.png"), dpi=100)
        plt.close()

        # Equilibration (NPT, no velocity swapping)
        print(f"    Equilibrating ({N_EQUILIBRATION_CYCLES} cycles)...")
        for eq_cycle in range(N_EQUILIBRATION_CYCLES):
            run_atoms = run_one_cycle(run_atoms, run_dir)
        print(f"    Post-equilibration T = {run_atoms.get_temperature():.1f} K")

        # Production rNEMD cycles
        print(f"    Production ({N_CYCLES} cycles)...")
        temps_times = np.zeros((N_CYCLES, NBINS))
        velocities_hc = np.zeros((N_CYCLES, 2))

        for cycle in range(N_CYCLES):
            run_atoms = run_one_cycle(run_atoms, run_dir)

            # Müller-Plathe velocity swap
            v_hot, v_cold = swap_velocities(
                run_atoms, binned[COLD_BIN], binned[HOT_BIN]
            )
            velocities_hc[cycle] = [v_hot, v_cold]

            # Record bin temperatures
            for b_idx, atom_indices in enumerate(binned):
                temps_times[cycle, b_idx] = run_atoms[atom_indices].get_temperature()

            if (cycle + 1) % 10 == 0:
                print(f"      cycle {cycle + 1}/{N_CYCLES}, "
                      f"T = {run_atoms.get_temperature():.1f} K")

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
        plot_temperature_profile(
            temps_times, bin_centers, result, run_dir,
            gb_label_str, run_idx, converged, max_dev,
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
    # # GPUMD requires upper-triangular cell (a along x, b in xy-plane, cz>0).
    # # cellpar_to_cell preserves fractional coords: axis 2 stays the stacking direction.
    # params = cell_to_cellpar(atoms.cell)
    # atoms.set_cell(cellpar_to_cell(params), scale_atoms=True)
    # atoms.wrap()
    print(f"\n{'='*60}")
    print(f"Processing {gb_label_str}  (config: {CONFIG_NAME})")
    print(f"  using run_{best_run_index} (lowest E = {best_energy:.4f} eV)")
    print(f"  n_runs={N_RUNS}")
    print(f"{'='*60}")

    out_base = os.path.join(RNEMD_RESULTS_DIR, gb_label_str)
    struct_dir = os.path.join(out_base, f"structure_{best_run_index}")
    print(f"\n--- Structure run_{best_run_index} (E={best_energy:.4f} eV) ---")

    all_run_results, aggregate = run_rnemd_on_structure(
        atoms, best_run_index, gb_label_str, struct_dir
    )

    # --- Per-run summary CSV ---
    os.makedirs(out_base, exist_ok=True)

    summary_path = os.path.join(out_base, "summary.csv")
    summary_fields = [
        "structure_index", "run_index", "energy_ev",
        "R_K_SI", "kappa_SI", "J_SI", "delta_T", "n_atoms", "converged",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_run_results)
    print(f"\nPer-run summary written to {summary_path}")

    # --- Aggregate CSV ---
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
    print(f"Aggregate summary written to {agg_path}")

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
    else:
        gb_dirs = sorted(glob.glob(os.path.join(GB_RESULTS_DIR, "sigma*")))
        if not gb_dirs:
            raise RuntimeError(f"No GB result folders found in {GB_RESULTS_DIR}.")
        for gb_dir in gb_dirs:
            process_gb_type(os.path.basename(gb_dir))


if __name__ == "__main__":
    main()