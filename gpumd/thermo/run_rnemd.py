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
import shutil
import argparse
import subprocess
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
from ase.io.extxyz import XYZError
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.visualize.plot import plot_atoms
warnings.filterwarnings("ignore", message=".*is not empty.*", module="calorine")
from calorine.calculators import GPUNEP

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.muller_plathe import swap_velocities, bin_atoms
from utils.rnemd_stats import check_steady_state, aggregate_run_results, format_result_summary
from utils.rnemd_plots import plot_temperature_profile, plot_energy_diagnostics, plot_temperature_profile_animated
from utils.work_coordination import (
    gb_label, check_rnemd_status, resolve_results_base,
    try_claim, refresh_claim, release_claim, CLAIM_STALE_HOURS,
)

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
USE_CALORINE   = bool(config.get("use_calorine", False))

RESULTS_BASE      = resolve_results_base(config, GPUMD_ROOT)
GB_RESULTS_DIR    = str(RESULTS_BASE / CONFIG_NAME / "gb_generation")
RNEMD_RESULTS_DIR = str(RESULTS_BASE / CONFIG_NAME / "rnemd")

rnemd_cfg = config["rnemd"]
NBINS            = int(rnemd_cfg["nbins"])
COLD_BIN         = NBINS // 4
HOT_BIN          = 3 * NBINS // 4
STEPS_PER_CYCLE  = int(rnemd_cfg["steps_per_cycle"])
TIMESTEP_FS      = float(rnemd_cfg["timestep_fs"])
N_CYCLES         = int(rnemd_cfg["n_cycles"])
N_RUNS           = int(rnemd_cfg.get("n_runs", 3))
ENSEMBLE         = rnemd_cfg.get("ensemble", "npt_scr").lower()
# Treat "nvt" as shorthand for "nvt_nhc"
if ENSEMBLE == "nvt":
    ENSEMBLE = "nvt_nhc"
TEMPERATURE_K    = float(rnemd_cfg["temperature_k"])
if ENSEMBLE == "npt_scr":
    TAU_T            = float(rnemd_cfg["tau_t"])
    PRESSURE_GPA     = float(rnemd_cfg["pressure_gpa"])
    BULK_MODULUS_GPA = float(rnemd_cfg["bulk_modulus_gpa"])
    TAU_P            = float(rnemd_cfg["tau_p"])
elif ENSEMBLE == "nvt_nhc":
    TAU_T            = float(rnemd_cfg.get("tau_t", 100.0))
assert ENSEMBLE in ["npt_scr", "nve", "nvt_nhc"], f"Unsupported ensemble: {ENSEMBLE}"
DEBUG_STRUCTURE   = bool(rnemd_cfg.get("debug_structure", False))
DEBUG_DIAGNOSTICS = bool(rnemd_cfg.get("debug_diagnostics", True))
INCLUDE_MOVIE     = bool(rnemd_cfg.get("include_movie", False))
INCLUDE_ANIMATION = bool(rnemd_cfg.get("debug_animation", False))
N_WARMUP_CYCLES   = int(rnemd_cfg.get("n_warmup_cycles", 0))

# GB list from YAML (used in main() to restrict processing to configured GBs only)
BULK_SI_LABEL = "bulk_si"
_raw_gbs = config["grain_boundaries"]
GB_LIST = [
    (tuple(entry.get("axis", [])), int(entry["sigma"]), tuple(entry.get("plane", [])))
    for entry in _raw_gbs
]

# Si atomic mass in amu (used for energy flux calculation)
M_SI_AMU = 28.085

# ---------------------------------------------------------------------------
# Single rNEMD cycle
# ---------------------------------------------------------------------------

def _write_run_in(run_dir):
    """Write run.in for direct GPUMD execution (non-calorine path)."""
    rel_potential = os.path.relpath(NEP_MODEL_FILE, run_dir)
    lines = [
        f"potential {rel_potential}",
        f"time_step {TIMESTEP_FS}",
    ]
    if ENSEMBLE == "nve":
        lines.append("ensemble nve")
    elif ENSEMBLE == "nvt_nhc":
        lines.append(f"ensemble nvt_nhc {TEMPERATURE_K} {TEMPERATURE_K} {TAU_T}")
    elif ENSEMBLE == "npt_scr":
        lines.append(
            f"ensemble npt_scr {TEMPERATURE_K} {TEMPERATURE_K} "
            f"{TAU_T} {PRESSURE_GPA} {BULK_MODULUS_GPA} {TAU_P}"
        )
    lines += [
        f"dump_velocity {STEPS_PER_CYCLE}",
        f"dump_position {STEPS_PER_CYCLE}",
        f"dump_thermo {STEPS_PER_CYCLE}",
        f"run {STEPS_PER_CYCLE}",
    ]
    with open(os.path.join(run_dir, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_one_cycle_gpumd(atoms, run_dir):
    """
    Run STEPS_PER_CYCLE MD steps by invoking the GPUMD executable directly.
    Used when use_calorine is False in the YAML config.

    Velocity unit convention (same as the calorine path):
      ASE internal:  Å/t_ASE  (t_ASE = sqrt(amu·Å²/eV) ≈ 10.18 fs)
      GPUMD expects: Å/fs
      Conversion:    v_gpumd = v_ase * units.fs
    """
    # Write model.xyz with velocities in GPUMD units (Å/fs)
    tmp = atoms.copy()
    tmp.set_velocities(atoms.get_velocities() * units.fs)
    write(os.path.join(run_dir, "model.xyz"), tmp, format="extxyz")
    _write_run_in(run_dir)

    # Remove stale per-cycle output (GPUMD appends rather than overwrites)
    for fname in ("velocity.out", "movie.xyz", "thermo.out"):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # Run GPUMD
    with open(os.path.join(run_dir, "stdout"), "w") as stdout_f:
        result = subprocess.run(
            [GPUMD_EXEC], cwd=run_dir,
            stdout=stdout_f, stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPUMD failed (rc={result.returncode}):\n"
            + result.stderr.decode()
        )

    # Read final positions from movie.xyz (last frame, positions in Å)
    updated = read(os.path.join(run_dir, "movie.xyz"), index=-1, format="extxyz")

    # Read final velocities from velocity.out (last N lines, units: Å/fs)
    vel_path = os.path.join(run_dir, "velocity.out")
    vels = pd.read_csv(vel_path, sep=r"\s+", header=None).iloc[-len(atoms):, :3]
    updated.set_velocities(vels.values / units.fs)  # Å/fs → ASE units

    # Read kinetic and potential energy from thermo.out (columns 1=K, 2=U in eV)
    thermo_data = np.loadtxt(os.path.join(run_dir, "thermo.out"))
    if thermo_data.ndim == 1:
        ke = float(thermo_data[1])
        pe = float(thermo_data[2])
    else:
        ke = float(thermo_data[-1, 1])
        pe = float(thermo_data[-1, 2])

    # Clean up per-cycle files to prevent ever-growing outputs
    files_to_remove = ["velocity.out", "thermo.out"]
    if not INCLUDE_MOVIE:
        files_to_remove.append("movie.xyz")
    for fname in files_to_remove:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    return updated, ke, pe


def _gpumd_output_error(run_dir, path, detail):
    """Build a clear, actionable error for a corrupt/truncated GPUMD output file.

    In production the usual culprits are (a) the filesystem quota being exceeded
    mid-write — GPUMD does not check write return codes, so it reports success
    while leaving a truncated/headerless file — and (b) two workers writing into
    the same run directory after a stale-claim steal.  We surface both, plus the
    free space on the output volume, so a multi-hour run fails with a diagnosis
    instead of a cryptic downstream ASE/broadcast error.
    """
    try:
        free_gb = shutil.disk_usage(run_dir).free / 1e9
        space = f"{free_gb:.2f} GB free on output volume"
    except OSError:
        space = "free space unknown"
    return (
        f"Corrupt GPUMD output in {run_dir}:\n"
        f"    {path}: {detail}\n"
        f"  GPUMD reported success but the file is truncated/garbled. Likely causes:\n"
        f"    1. Disk quota exceeded mid-write ({space}; check your quota).\n"
        f"    2. Another worker writing into the same run dir (stale-claim steal).\n"
        f"  Completed runs are preserved; fix storage/claims and re-run."
    )


def run_one_cycle(atoms, run_dir):
    """
    Run STEPS_PER_CYCLE MD steps via GPUMD, read back velocities, and return
    the updated atoms with correct velocities attached alongside the potential
    energy for that cycle.

    Returns
    -------
    (atoms, ke, pe) where ke and pe are the kinetic and potential energy in eV
    read from GPUMD's thermo.out (columns K and U respectively).

    When use_calorine is False, dispatches to run_one_cycle_gpumd() which
    calls the GPUMD executable directly.  Otherwise, uses calorine.

    Calorine quirk: velocities are not returned by run_custom_md — they must
    be read from velocity.out.  The division by ~0.098 converts from GPUMD's
    internal velocity units (Å/fs) to ASE's internal units (Å/t_ASE where
    t_ASE ≈ 10.18 fs ≈ sqrt(amu·Å²/eV)).  The exact factor is ase.units.fs.
    """
    if not USE_CALORINE:
        return run_one_cycle_gpumd(atoms, run_dir)

    if ENSEMBLE == "npt_scr":
        ensemble_params = ['npt_scr', TEMPERATURE_K, TEMPERATURE_K, TAU_T, PRESSURE_GPA, BULK_MODULUS_GPA, TAU_P]
    elif ENSEMBLE == "nvt_nhc":
        ensemble_params = ['nvt_nhc', TEMPERATURE_K, TEMPERATURE_K, TAU_T]
    elif ENSEMBLE == "nve":
        ensemble_params = ['nve']

    md_params = [
        ("dump_position", STEPS_PER_CYCLE),
        ("dump_velocity", STEPS_PER_CYCLE),
        ("dump_thermo", STEPS_PER_CYCLE),
        ("time_step", TIMESTEP_FS),
        ("ensemble", ensemble_params),
        ("run", STEPS_PER_CYCLE),
    ]

    # Convert ASE velocities (Å/t_ASE) to GPUMD units (Å/fs).
    # Calorine writes vel to model.xyz without converting, but GPUMD
    # reads vel as Å/fs.  Without this, velocities are ~10x too large.
    atoms.set_velocities(atoms.get_velocities() * units.fs)

    # Remove stale dump files before each cycle: GPUMD appends rather than
    # overwrites, so leftovers from an interrupted/crashed run (resumed in the
    # same dir) would be appended to and corrupt the reads below.
    for fname in ("movie.xyz", "velocity.out", "position.out", "thermo.out"):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # NOTE: Must re-create calculator each cycle (calorine limitation)
    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=run_dir,
        atoms=atoms,
    )

    # Validate GPUMD output before trusting it.  A truncated movie.xyz makes
    # calorine's internal read raise StopIteration/XYZError/ValueError; catch
    # those and re-raise with a clear cause instead of a cryptic ASE traceback.
    n_expected = len(atoms)
    movie_path = os.path.join(run_dir, "movie.xyz")
    try:
        atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    except (StopIteration, XYZError, ValueError) as exc:
        raise RuntimeError(_gpumd_output_error(run_dir, movie_path, repr(exc))) from exc

    if len(atoms) != n_expected:
        raise RuntimeError(_gpumd_output_error(
            run_dir, movie_path,
            f"last frame has {len(atoms)} atoms, expected {n_expected}"))

    # Read velocities from GPUMD output (last len(atoms) lines)
    vel_path = os.path.join(run_dir, "velocity.out")
    vels = pd.read_csv(vel_path, sep=" ", header=None).iloc[-n_expected:, :]
    if len(vels) != n_expected:
        raise RuntimeError(_gpumd_output_error(
            run_dir, vel_path,
            f"{len(vels)} usable rows, expected {n_expected} (truncated write)"))
    atoms.set_velocities(vels.values / units.fs)  # GPUMD (Å/fs) -> ASE units

    # Read kinetic and potential energy from thermo.out (columns 1=K, 2=U in eV)
    thermo_data = np.loadtxt(os.path.join(run_dir, "thermo.out"))
    if thermo_data.ndim == 1:
        ke = float(thermo_data[1])
        pe = float(thermo_data[2])
    else:
        ke = float(thermo_data[-1, 1])
        pe = float(thermo_data[-1, 2])

    # At the end of run_one_cycle, after reading velocities
    # this prevents us from having output files that get longer and longer!
    files_to_remove = ["velocity.out", "position.out", "thermo.out"]
    if not INCLUDE_MOVIE:
        files_to_remove.append("movie.xyz")
    for fname in files_to_remove:
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    return atoms, ke, pe


# ---------------------------------------------------------------------------
# TBR and kappa calculation
# ---------------------------------------------------------------------------

def compute_tbr_and_kappa(temps_avg, velocities_hc, bin_centers_angstrom,
                           cross_section_angstrom2, total_time_fs, is_bulk=False):
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
    # Due to periodic boundary conditions, there are duplicate bulk segments
    # wrapping around each end of the box (start→cold and hot→end).  These
    # are averaged with the primary fits to reduce noise.
    margin = 1
    # NOTE: this is only valid when UC_A == UC_B, if we want twin boundaries then this doesn't work!
    gb_bin = NBINS // 2

    # Primary segments (inner bulk regions)
    left_slice  = slice(COLD_BIN + margin, gb_bin - margin)   # cold → GB  (slope > 0)
    right_slice = slice(gb_bin + margin, HOT_BIN - margin)    # GB   → hot (slope > 0)
    # Periodic duplicate segments (wrap-around bulk regions)
    cold_dup_slice = slice(margin, COLD_BIN - margin)          # start → cold (slope < 0)
    hot_dup_slice  = slice(HOT_BIN + margin, NBINS - margin)   # hot → end   (slope < 0)

    x_cold_dup  = bin_centers_angstrom[cold_dup_slice]
    T_cold_dup  = temps_avg[cold_dup_slice]
    x_hot_dup   = bin_centers_angstrom[hot_dup_slice]
    T_hot_dup   = temps_avg[hot_dup_slice]
    cold_dup_fit = np.polyfit(x_cold_dup, T_cold_dup, 1)  # slope < 0 (start→cold wrap)
    hot_dup_fit  = np.polyfit(x_hot_dup,  T_hot_dup,  1)  # slope < 0 (hot→end wrap)

    bin_width = bin_centers_angstrom[1] - bin_centers_angstrom[0]
    box_length_angstrom = bin_centers_angstrom[-1] + bin_width / 2.0

    if is_bulk:
        # No grain boundary: fit a single slope across the full cold→hot region
        # instead of splitting at gb_bin. The falling region (hot→cold via periodic
        # wrap) uses the average magnitude of the two separate dup fits.
        rising_slice = slice(COLD_BIN + margin, HOT_BIN - margin)
        x_rising = bin_centers_angstrom[rising_slice]
        T_rising = temps_avg[rising_slice]
        rising_fit = np.polyfit(x_rising, T_rising, 1)  # slope > 0

        rising_slope  = rising_fit[0]                                    # K/Å, > 0
        falling_slope = (-hot_dup_fit[0] + (-cold_dup_fit[0])) / 2.0   # K/Å, > 0

        dTdx_rising_SI  = rising_slope  * 1e10  # K/m
        dTdx_falling_SI = falling_slope * 1e10  # K/m
        dTdx_SI         = (rising_slope + falling_slope) / 2.0 * 1e10

        kappa_rising  = abs(J / dTdx_rising_SI)  if dTdx_rising_SI  > 0 else np.nan
        kappa_falling = abs(J / dTdx_falling_SI) if dTdx_falling_SI > 0 else np.nan
        kappa         = np.nanmean([kappa_rising, kappa_falling])

        return {
            "R_K_SI": np.nan,
            "kappa_SI": kappa,
            "kappa_cold_SI": kappa_rising,   # rising (cold→hot) slope
            "kappa_hot_SI": kappa_falling,   # falling (hot→cold) slope
            "J_SI": J,
            "delta_T": np.nan,
            "dTdx_K_per_m": dTdx_SI,
            "left_fit": rising_fit,   # same fit used for both halves of rising region
            "right_fit": rising_fit,
            "cold_dup_fit": cold_dup_fit,
            "hot_dup_fit": hot_dup_fit,
        }

    x_left      = bin_centers_angstrom[left_slice]
    T_left      = temps_avg[left_slice]
    x_right     = bin_centers_angstrom[right_slice]
    T_right     = temps_avg[right_slice]

    left_fit     = np.polyfit(x_left,     T_left,     1)  # slope > 0 (cold→GB)
    right_fit    = np.polyfit(x_right,    T_right,    1)  # slope > 0 (GB→hot)

    # Per-grain average gradient magnitude:
    #   cold grain: average of left_fit[0] (> 0) and −cold_dup_fit[0] (> 0)
    #   hot  grain: average of right_fit[0] (> 0) and −hot_dup_fit[0] (> 0)
    cold_slope = (left_fit[0] + (-cold_dup_fit[0])) / 2.0  # K/Å
    hot_slope  = (right_fit[0] + (-hot_dup_fit[0])) / 2.0  # K/Å

    # Grand average slope and per-grain kappas
    avg_slope    = (cold_slope + hot_slope) / 2.0   # K/Å
    dTdx_SI      = avg_slope    * 1e10              # K/m
    cold_dTdx_SI = cold_slope   * 1e10
    hot_dTdx_SI  = hot_slope    * 1e10

    kappa      = abs(J / dTdx_SI)      if abs(dTdx_SI)      > 0 else np.nan  # W/(m·K)
    kappa_cold = abs(J / cold_dTdx_SI) if abs(cold_dTdx_SI) > 0 else np.nan
    kappa_hot  = abs(J / hot_dTdx_SI)  if abs(hot_dTdx_SI)  > 0 else np.nan

    # TBR: extrapolate left and right fits to the GB position
    x_gb = bin_centers_angstrom[gb_bin]
    T_left_at_gb = np.polyval(left_fit, x_gb)
    T_right_at_gb = np.polyval(right_fit, x_gb)
    delta_T = abs(T_left_at_gb - T_right_at_gb)

    R_K = delta_T / J if J > 0 else np.nan  # K·m²/W

    # Duplicate TBR: extrapolate cold_dup and hot_dup fits to the periodic boundary
    # (x=0 and x=box_length are the same physical point — the duplicate GB).
    T_cold_dup_at_dup_gb = np.polyval(cold_dup_fit, 0.0)
    T_hot_dup_at_dup_gb  = np.polyval(hot_dup_fit, box_length_angstrom)
    delta_T_dup = abs(T_cold_dup_at_dup_gb - T_hot_dup_at_dup_gb)
    R_K_dup = delta_T_dup / J if J > 0 else np.nan  # K·m²/W

    # Average primary and duplicate TBR estimates for a less noisy final value
    R_K_avg = np.nanmean([R_K, R_K_dup])

    return {
        "R_K_SI": R_K_avg,
        "kappa_SI": kappa,          # grand average — name kept for CSV/analysis compatibility
        "kappa_cold_SI": kappa_cold,
        "kappa_hot_SI": kappa_hot,
        "J_SI": J,
        "delta_T": delta_T,
        "dTdx_K_per_m": dTdx_SI,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "cold_dup_fit": cold_dup_fit,
        "hot_dup_fit": hot_dup_fit,
    }


# ---------------------------------------------------------------------------
# Per-structure RNEMD runner
# ---------------------------------------------------------------------------

def run_rnemd_on_structure(atoms, structure_index, gb_label_str, out_dir, heartbeat=None):
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
                if heartbeat:
                    heartbeat()
                run_atoms, _, _ = run_one_cycle(run_atoms, run_dir)
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
        ke_per_cycle = np.full(N_CYCLES, np.nan)
        pe_per_cycle = np.full(N_CYCLES, np.nan)

        # Open CSV for incremental bin temperature logging
        bin_temps_csv_path = os.path.join(run_dir, "bin_temps.csv")
        with open(bin_temps_csv_path, "w", newline="") as f_csv:
            csv.writer(f_csv).writerow(["cycle"] + [f"bin_{i}" for i in range(NBINS)])

        for cycle in (pbar := tqdm(range(N_CYCLES))):
            if heartbeat:
                heartbeat()
            run_atoms, ke_per_cycle[cycle], pe_per_cycle[cycle] = run_one_cycle(run_atoms, run_dir)

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
        np.save(os.path.join(run_dir, "ke_per_cycle.npy"), ke_per_cycle)
        np.save(os.path.join(run_dir, "pe_per_cycle.npy"), pe_per_cycle)
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
            is_bulk=(gb_label_str == BULK_SI_LABEL),
        )
        result.update({
            "structure_index": structure_index,
            "run_index": run_idx,
            "energy_ev": atoms.info.get("energy_ev", np.nan),
            "n_atoms": len(run_atoms),
            "converged": converged,
        })
        all_run_results.append(result)

        print(f"    κ = {result['kappa_SI']:.2f} W/(m·K) "
              f"[cold: {result['kappa_cold_SI']:.2f}, hot: {result['kappa_hot_SI']:.2f}], "
              f"R_K = {result['R_K_SI']:.3e} K·m²/W, "
              f"J = {result['J_SI']:.3e} W/m²")

        # Diagnostic plots
        if DEBUG_DIAGNOSTICS:
            plot_temperature_profile(
                temps_times, bin_centers, result, run_dir,
                gb_label_str, run_idx,
                cold_bin=COLD_BIN, hot_bin=HOT_BIN, nbins=NBINS,
            )
            plot_energy_diagnostics(
                temps_times, ke_per_cycle, pe_per_cycle, result["n_atoms"], run_dir,
                gb_label_str, run_idx, converged, max_dev,
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

def process_gb_type(gb_label_str, claim_path=None):
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

    # Heartbeat keeps this GB's claim fresh while it runs (~22 h), so another
    # worker won't see it as stale and start writing into the same run dir.
    heartbeat = (lambda: refresh_claim(claim_path)) if claim_path else None

    all_run_results, _ = run_rnemd_on_structure(
        atoms, best_run_index, gb_label_str, struct_dir, heartbeat=heartbeat
    )

    # --- Per-run summary CSV (append to existing rows if present) ---
    os.makedirs(out_base, exist_ok=True)

    summary_path = os.path.join(out_base, "summary.csv")
    summary_fields = [
        "structure_index", "run_index", "energy_ev",
        "R_K_SI", "kappa_SI", "kappa_cold_SI", "kappa_hot_SI",
        "J_SI", "delta_T", "n_atoms", "converged",
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
        raise FileNotFoundError(f"Potential not found at '{NEP_MODEL_FILE}'.")

    os.makedirs(RNEMD_RESULTS_DIR, exist_ok=True)

    if args.gb:
        process_gb_type(args.gb)
        return

    if not GB_LIST:
        raise RuntimeError("No grain boundaries defined in config.")

    rnemd_status = check_rnemd_status(args.config)
    all_done = all(info["runs_remaining"] == 0 for info in rnemd_status.values())
    if all_done:
        print(f"All {len(GB_LIST)} entries already completed ({N_RUNS}/{N_RUNS} runs each). Nothing to do.")
        return

    for (axis, sigma, plane) in GB_LIST:
        label = BULK_SI_LABEL if sigma == -1 else gb_label(axis, sigma, plane)
        info = rnemd_status.get(label, {"status": "not_started", "runs_remaining": N_RUNS})

        if info["status"] == "completed":
            print(f"\nSkipping {label}: already completed.")
            continue

        claim_path = os.path.join(RNEMD_RESULTS_DIR, label, ".claimed")
        if not try_claim(claim_path, stale_hours=CLAIM_STALE_HOURS):
            print(f"\nSkipping {label}: claimed by another worker.")
            continue

        try:
            process_gb_type(label, claim_path)
        finally:
            release_claim(claim_path)


if __name__ == "__main__":
    main()