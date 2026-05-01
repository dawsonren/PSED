"""
Use aimsgb to generate grain boundaries, then use GPUMD to relax
such structures into realistic crystalline configurations.

Usage:
    python gb_generation/generate_gbs.py --config configs/small_box.yaml
    python gb_generation/generate_gbs.py --config configs/large_box.yaml

Pipeline:
1. Load GB specifications and run parameters from a unified YAML config.
2. Build GB structure with aimsgb (GrainBoundary + Grain.stack_grains).
   The x/y/z lengths repeated to achieve the correct BOX_SIZE.
3. Anneal with GPUMD: cooling ramp from npt.t_start to npt.t_end over
   npt.total_time_ps, using npt_scr thermostat.
4. Equilibrate with GPUMD: NVT run from nvt.t_start to nvt.t_end over
   nvt.total_time_ps, using nvt_nhc thermostat.
5. Repeat steps 3-4 n_runs times with different random initial velocities.
   All final structures are saved as .traj files.
6. A summary.csv records energies per run.

File outputs:
    results/<config_name>/gb_generation/
      sigma{n}_{miller}_{axis}/
        run_0/
          npt/                  <- NPT GPUMD working directory (all stages in one run.in)
            run.in
            movie.xyz
            thermo.out
          nvt/                  <- NVT GPUMD working directory (all stages in one run.in)
            run.in
            movie.xyz
            thermo.out
          structure.traj        <- final structure after NVT
        run_1/
          ...
        summary.csv     <- run_index, energy_ev per row
"""

import os
import csv
import argparse
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv

load_dotenv()

from aimsgb import GrainBoundary, Grain
from ase.io import read, write
from ase.visualize.plot import plot_atoms
from calorine.calculators import GPUNEP

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.work_coordination import (
    gb_label, check_gb_generation_status,
    try_claim, release_claim, CLAIM_STALE_HOURS,
)
from utils.gb_energy import bulk_energy_per_atom

# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate and relax grain boundary structures with GPUMD"
)
parser.add_argument(
    "--config", type=str, required=True,
    help="Path to unified YAML config file (e.g. ../configs/small_box.yaml)"
)
args = parser.parse_args()

# Resolve paths: gpumd root is the parent of gb_generation/
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

RESULTS_DIR    = str(GPUMD_ROOT / "results" / CONFIG_NAME / "gb_generation")

gb_cfg = config["gb_generation"]
# minimum length of supercell in x/y/z axes in angstroms
BOX_SIZE    = np.array([float(gb_cfg["x_nm"]) * 10, float(gb_cfg["y_nm"]) * 10, float(gb_cfg["z_nm"]) * 10])
N_RUNS      = int(gb_cfg["n_runs"])
TIMESTEP_FS = float(gb_cfg["timestep_fs"])
DEBUG       = bool(gb_cfg.get("debug", False))

def _as_list(v):
    return v if isinstance(v, list) else [v]

_dump_interval_cfg = int(gb_cfg["dump_interval"]) if DEBUG else None

def _make_stages(t_starts, t_ends, times_ps):
    stages = []
    for ts, te, tp in zip(t_starts, t_ends, times_ps):
        n = int(tp * 1000.0 / TIMESTEP_FS)
        stages.append({
            "t_start": ts, "t_end": te, "total_time_ps": tp,
            "n_steps": n,
            "thermo_interval": max(int(n / 100), 1),
            "dump_interval": _dump_interval_cfg if DEBUG else n - 1,
        })
    return stages

# NPT parameters
npt_cfg          = gb_cfg["npt"]
TAU_T            = float(npt_cfg["tau_t"])
PRESSURE_GPA     = float(npt_cfg["pressure_gpa"])
BULK_MODULUS_GPA = float(npt_cfg["bulk_modulus_gpa"])
TAU_P            = float(npt_cfg["tau_p"])
NPT_STAGES       = _make_stages(
    [float(x) for x in _as_list(npt_cfg["t_start"])],
    [float(x) for x in _as_list(npt_cfg["t_end"])],
    [float(x) for x in _as_list(npt_cfg["total_time_ps"])],
)

# NVT parameters
nvt_cfg   = gb_cfg["nvt"]
NVT_TAU_T = float(nvt_cfg["tau_t"])
NVT_STAGES = _make_stages(
    [float(x) for x in _as_list(nvt_cfg["t_start"])],
    [float(x) for x in _as_list(nvt_cfg["t_end"])],
    [float(x) for x in _as_list(nvt_cfg["total_time_ps"])],
)

_raw_gbs = config["grain_boundaries"]
# sigma: -1 signals a bulk reference entry; axis/plane are ignored for those.
GB_LIST = [
    (tuple(entry.get("axis", [])), int(entry["sigma"]), tuple(entry.get("plane", [])))
    for entry in _raw_gbs
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gb_label(axis, sigma, plane):
    """Produce a filesystem-safe label, e.g. sigma5_2-10_001"""
    a = "".join(str(x) for x in axis)
    p = "".join(str(x) for x in plane)
    return f"{a}_sigma{sigma}_{p}"


def build_gb_atoms(s_input, axis, sigma, plane):
    # --- Probe build: uc_a=uc_b=1 to get base dimensions ---
    gb_probe = GrainBoundary(axis, sigma, plane, s_input, uc_a=1, uc_b=1)
    structure_probe = Grain.stack_grains(
        gb_probe.grain_a, gb_probe.grain_b,
        direction=gb_probe.direction, to_primitive=False
    )
    atoms_probe = structure_probe.to_ase_atoms()
    probe_lengths = atoms_probe.cell.lengths()
    d = gb_probe.direction

    # --- Determine the axis permutation ---
    if d == 0:
        perm = [1, 2, 0]
    elif d == 1:
        perm = [0, 2, 1]
    else:
        perm = [0, 1, 2]

    # --- Compute multipliers ---
    # After permutation: new_x = old[perm[0]], new_y = old[perm[1]], new_z = old[d]
    # UC controls z (stacking), repeat controls x and y (in-plane)
    UC = max(int(np.ceil(BOX_SIZE[2] / probe_lengths[d])), 1)
    scale_x = max(int(np.ceil(BOX_SIZE[0] / probe_lengths[perm[0]])), 1)
    scale_y = max(int(np.ceil(BOX_SIZE[1] / probe_lengths[perm[1]])), 1)

    # --- Real build with correct UC ---
    gb = GrainBoundary(axis, sigma, plane, s_input, uc_a=UC, uc_b=UC)
    structure = Grain.stack_grains(
        gb.grain_a, gb.grain_b,
        direction=gb.direction,
        to_primitive=False
    )
    atoms = structure.to_ase_atoms()

    # --- Permute so stacking direction → z ---
    if d != 2:
        new_cell = atoms.cell[perm][:, perm]
        new_positions = atoms.positions[:, perm]
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.set_positions(new_positions)

    # --- In-plane tiling ---
    atoms = atoms.repeat((scale_x, scale_y, 1))

    # --- Clean up any tiny off-diagonal elements so it's cubic ---
    cell = atoms.cell[:]
    np.fill_diagonal(cell, np.diag(cell))  # keep diagonal
    off_diag_mask = ~np.eye(3, dtype=bool)
    cell[off_diag_mask] = 0.0
    atoms.set_cell(cell, scale_atoms=False)

    atoms.pbc = True
    atoms.wrap()

    return atoms, (scale_x, scale_y, UC)



def cool_with_gpumd(atoms, npt_dir, stages):
    """
    Run all NPT npt_scr stages as a single chained GPUMD run.

    All stages are written into one run.in with a single velocity initialisation
    at the start. Subsequent ensemble blocks continue from the previous state,
    avoiding the spurious re-initialisation to 300 K that occurs when each stage
    is a separate process.

    TAU_T (in timesteps) sets the coupling timescale (recommended
    to be 100 x timestep in GPUMD). Too small causes unphysical velocity kicks;
    too large and the temperature lags the ramp target.
    """
    os.makedirs(npt_dir, exist_ok=True)

    # Remove stale movie.xyz so calorine reads the correct run's output
    movie_path = os.path.join(npt_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=npt_dir,
        atoms=atoms,
    )
    atoms = atoms.copy()
    atoms.calc = calc

    # velocity and time_step appear once; each stage appends its own
    # ensemble/dump_thermo/dump_position/run block
    md_params = [
        ("velocity",  stages[0]["t_start"]),
        ("time_step", TIMESTEP_FS),
    ]
    for stage in stages:
        md_params += [
            ("ensemble",     ["npt_scr", stage["t_start"], stage["t_end"],
                              TAU_T, PRESSURE_GPA, BULK_MODULUS_GPA, TAU_P]),
            ("dump_thermo",  stage["thermo_interval"]),
            ("dump_position", stage["dump_interval"]),
            ("run",          stage["n_steps"]),
        ]

    cooled_atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    # Read potential energy from the last row of thermo.out (column 2 = U [eV])
    thermo_data = np.loadtxt(os.path.join(npt_dir, "thermo.out"))
    if thermo_data.ndim == 1:
        energy_ev = float(thermo_data[2])
    else:
        energy_ev = float(thermo_data[-1, 2])

    return cooled_atoms, energy_ev


def cool_with_gpumd_direct(atoms, npt_dir, stages):
    """
    Run all NPT npt_scr stages as a single chained GPUMD run (direct executable).
    Used when the potential is an empirical potential (SW, Tersoff, tersoff_mini, ...)
    that calorine/GPUNEP does not support.

    All stages are written into one run.in with a single velocity initialisation
    at the start. Subsequent ensemble blocks continue from the previous state,
    avoiding the spurious re-initialisation to 300 K that occurs when each stage
    is a separate process.

    Returns
    -------
    cooled_atoms : ase.Atoms
    energy_ev    : float   potential energy of the final frame [eV]
    """
    os.makedirs(npt_dir, exist_ok=True)

    # Remove stale output files
    for fname in ("thermo.out", "movie.xyz"):
        fpath = os.path.join(npt_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    if not atoms.has('mass'):
        atoms.new_array('mass', atoms.get_masses())

    # Write structure (velocities will be initialised by the 'velocity' keyword in run.in)
    write(os.path.join(npt_dir, "model.xyz"), atoms, format="extxyz")

    rel_potential = os.path.relpath(NEP_MODEL_FILE, npt_dir)
    # velocity and time_step appear once; each stage appends its own block
    lines = [
        f"potential {rel_potential}",
        f"velocity {stages[0]['t_start']}",
        f"time_step {TIMESTEP_FS}",
    ]
    for stage in stages:
        lines += [
            f"ensemble npt_scr {stage['t_start']} {stage['t_end']} "
            f"{TAU_T} {PRESSURE_GPA} {BULK_MODULUS_GPA} {TAU_P}",
            f"dump_thermo {stage['thermo_interval']}",
            f"dump_position {stage['dump_interval']}",
            f"run {stage['n_steps']}",
        ]
    with open(os.path.join(npt_dir, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(npt_dir, "stdout"), "w") as stdout_f:
        result = subprocess.run(
            [GPUMD_EXEC], cwd=npt_dir,
            stdout=stdout_f, stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPUMD NPT failed (rc={result.returncode}):\n"
            + result.stderr.decode()
        )

    # Read final structure from movie.xyz (last frame; cell is preserved in extxyz)
    cooled_atoms = read(os.path.join(npt_dir, "movie.xyz"), index=-1, format="extxyz")
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    # Read potential energy from the last row of thermo.out (column 2 = U [eV])
    thermo_data = np.loadtxt(os.path.join(npt_dir, "thermo.out"))
    if thermo_data.ndim == 1:
        energy_ev = float(thermo_data[2])
    else:
        energy_ev = float(thermo_data[-1, 2])

    return cooled_atoms, energy_ev


def nvt_with_gpumd(atoms, nvt_dir, stages):
    """
    Run all NVT nvt_nhc stages as a single chained GPUMD run.

    All stages are written into one run.in with a single velocity initialisation
    at the start. Subsequent ensemble blocks continue from the previous state.

    Uses the Nosé-Hoover chain thermostat (nvt_nhc). NVT_TAU_T (in timesteps)
    sets the coupling timescale; the same guidance as TAU_T applies.
    """
    os.makedirs(nvt_dir, exist_ok=True)

    movie_path = os.path.join(nvt_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=nvt_dir,
        atoms=atoms,
    )
    atoms = atoms.copy()
    atoms.calc = calc

    md_params = [
        ("velocity",  stages[0]["t_start"]),
        ("time_step", TIMESTEP_FS),
    ]
    for stage in stages:
        md_params += [
            ("ensemble",     ["nvt_nhc", stage["t_start"], stage["t_end"], NVT_TAU_T]),
            ("dump_thermo",  stage["thermo_interval"]),
            ("dump_position", stage["dump_interval"]),
            ("run",          stage["n_steps"]),
        ]

    nvt_atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    nvt_atoms.pbc = atoms.pbc
    nvt_atoms.wrap()

    thermo_data = np.loadtxt(os.path.join(nvt_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return nvt_atoms, energy_ev


def nvt_with_gpumd_direct(atoms, nvt_dir, stages):
    """
    Run all NVT nvt_nhc stages as a single chained GPUMD run (direct executable).
    Used when the potential is an empirical potential (SW, Tersoff, tersoff_mini, ...)
    that calorine/GPUNEP does not support.

    All stages are written into one run.in with a single velocity initialisation
    at the start. Subsequent ensemble blocks continue from the previous state.

    Returns
    -------
    nvt_atoms : ase.Atoms
    energy_ev : float   potential energy of the final frame [eV]
    """
    os.makedirs(nvt_dir, exist_ok=True)

    for fname in ("thermo.out", "movie.xyz"):
        fpath = os.path.join(nvt_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    if not atoms.has('mass'):
        atoms.new_array('mass', atoms.get_masses())

    write(os.path.join(nvt_dir, "model.xyz"), atoms, format="extxyz")

    rel_potential = os.path.relpath(NEP_MODEL_FILE, nvt_dir)
    lines = [
        f"potential {rel_potential}",
        f"velocity {stages[0]['t_start']}",
        f"time_step {TIMESTEP_FS}",
    ]
    for stage in stages:
        lines += [
            f"ensemble nvt_nhc {stage['t_start']} {stage['t_end']} {NVT_TAU_T}",
            f"dump_thermo {stage['thermo_interval']}",
            f"dump_position {stage['dump_interval']}",
            f"run {stage['n_steps']}",
        ]
    with open(os.path.join(nvt_dir, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(nvt_dir, "stdout"), "w") as stdout_f:
        result = subprocess.run(
            [GPUMD_EXEC], cwd=nvt_dir,
            stdout=stdout_f, stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPUMD NVT failed (rc={result.returncode}):\n"
            + result.stderr.decode()
        )

    nvt_atoms = read(os.path.join(nvt_dir, "movie.xyz"), index=-1, format="extxyz")
    nvt_atoms.pbc = atoms.pbc
    nvt_atoms.wrap()

    thermo_data = np.loadtxt(os.path.join(nvt_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return nvt_atoms, energy_ev


def _build_piecewise_target(stages, n_actual_rows):
    """
    Build time_ps and piecewise-linear target_T arrays for a chained thermo.out.

    Each stage contributes floor(n_steps / thermo_interval) rows; the last stage
    absorbs any rounding surplus so the arrays always match n_actual_rows.
    """
    n_expected = [s["n_steps"] // s["thermo_interval"] for s in stages]

    # Allocate actual rows to stages: last stage absorbs any rounding surplus
    rows_per_stage = []
    remaining = n_actual_rows
    for i, ne in enumerate(n_expected[:-1]):
        total_expected = sum(n_expected)
        alloc = round(n_actual_rows * ne / total_expected)
        rows_per_stage.append(alloc)
        remaining -= alloc
    rows_per_stage.append(remaining)

    time_ps_parts = []
    target_T_parts = []
    cumulative_time = 0.0
    for stage, n_rows in zip(stages, rows_per_stage):
        if n_rows <= 0:
            continue
        time_ps_parts.append(
            np.linspace(cumulative_time, cumulative_time + stage["total_time_ps"], n_rows)
        )
        target_T_parts.append(np.linspace(stage["t_start"], stage["t_end"], n_rows))
        cumulative_time += stage["total_time_ps"]

    return np.concatenate(time_ps_parts), np.concatenate(target_T_parts)


def plot_temperature_trace(run_dir, label, run_index, stages):
    """
    Plot actual vs target temperature from thermo.out to validate TAU_T.

    Handles a chained run.in with multiple stages: the target temperature is
    piecewise linear, with one segment per stage. Vertical dashed lines mark
    stage boundaries.

    What to look for:
      - GOOD: actual temperature tracks the ramp smoothly with small fluctuations
      - BAD (TAU_T too small): rapid high-frequency oscillations around the target —
        the thermostat is overcorrecting every few steps, artificially disrupting dynamics
      - BAD (TAU_T too large): actual temperature lags far behind the ramp target —
        the thermostat barely intervenes and the system drifts freely

    thermo.out columns (GPUMD format):
        T  K  U  Pxx Pyy Pzz Pyz Pxz Pxy  ax ay az  bx by bz  cx cy cz
    Only the first column (T) is needed here.
    """
    thermo_path = os.path.join(run_dir, "thermo.out")
    if not os.path.exists(thermo_path):
        print(f"    Warning: thermo.out not found in {run_dir}, skipping T plot.")
        return

    import pandas as pd
    thermo = pd.read_csv(
        thermo_path,
        sep=r"\s+",
        header=None,
        names=["T", "K", "U", "Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy",
               "ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"],
    )

    time_ps, target_T = _build_piecewise_target(stages, len(thermo))

    total_time = sum(s["total_time_ps"] for s in stages)
    stage_summary = " → ".join(f"{s['t_start']}K" for s in stages) + f" → {stages[-1]['t_end']}K"

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.suptitle(
        f"{label} — run {run_index} NPT temperature trace\n"
        f"{stage_summary} over {total_time:.0f} ps  |  "
        f"TAU_T={TAU_T:.0f} steps ({TAU_T * TIMESTEP_FS / 1000:.1f} ps coupling)",
        fontsize=10,
    )

    # Top panel: actual vs target temperature
    axes[0].plot(time_ps, thermo["T"], color="tomato", linewidth=0.8, label="Actual T")
    axes[0].plot(time_ps, target_T, color="steelblue", linewidth=1.5,
                 linestyle="--", label="Target ramp")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Actual vs target — should track smoothly with no large oscillations or lag",
                      fontsize=8)

    # Bottom panel: residual (actual - target) — makes coupling quality obvious
    residual = thermo["T"].values - target_T
    axes[1].plot(time_ps, residual, color="darkorange", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("T_actual - T_target [K]")
    axes[1].set_xlabel("Time [ps]")
    axes[1].set_title(
        "Residual — oscillations → TAU_T too small; persistent drift → TAU_T too large",
        fontsize=8,
    )

    # Mark stage boundaries with vertical lines
    boundary_time = 0.0
    for stage in stages[:-1]:
        boundary_time += stage["total_time_ps"]
        for ax in axes:
            ax.axvline(boundary_time, color="gray", linewidth=0.8, linestyle=":")

    plt.tight_layout()
    out_path = os.path.join(run_dir, "temperature_trace.png")
    plt.savefig(out_path)
    plt.close()
    print(f"    Temperature trace saved to {out_path}")

    # Print a quick numeric summary so you can assess without opening the plot
    rms_residual = np.sqrt(np.mean(residual**2))
    max_residual = np.max(np.abs(residual))
    print(f"    TAU_T validation: RMS residual={rms_residual:.1f} K, "
          f"max |residual|={max_residual:.1f} K "
          f"(RMS < ~50 K is generally acceptable for annealing)")

def plot_nvt_diagnostics(nvt_dir, label, run_index, stages):
    """
    Diagnostic plots for the NVT nvt_nhc equilibration step.

    Handles a chained run.in with multiple stages: the target temperature is
    piecewise linear and stage boundaries are marked with vertical lines.

    Three panels:
      1. Temperature trace: actual T vs target — validates thermostat coupling.
         Oscillations → NVT_TAU_T too small; persistent drift → NVT_TAU_T too large.
      2. Potential energy U: should plateau after equilibration.
         A downward drift means the system is still relaxing; increase NVT_TOTAL_TIME_PS.
      3. Pressure components Pxx/Pyy/Pzz and mean pressure:
         Large anisotropy or a large non-zero mean suggests residual stress in the GB.

    thermo.out columns (GPUMD format, triclinic):
        T  K  U  Pxx Pyy Pzz Pyz Pxz Pxy  ax ay az  bx by bz  cx cy cz
    """
    thermo_path = os.path.join(nvt_dir, "thermo.out")
    if not os.path.exists(thermo_path):
        print(f"    Warning: thermo.out not found in {nvt_dir}, skipping NVT diagnostics.")
        return

    import pandas as pd
    thermo = pd.read_csv(
        thermo_path,
        sep=r"\s+",
        header=None,
        names=["T", "K", "U", "Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy",
               "ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"],
    )

    time_ps, target_T = _build_piecewise_target(stages, len(thermo))

    total_time = sum(s["total_time_ps"] for s in stages)
    stage_summary = " → ".join(f"{s['t_start']}K" for s in stages) + f" → {stages[-1]['t_end']}K"

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    plt.suptitle(
        f"{label} — run {run_index} NVT diagnostics\n"
        f"nvt_nhc: {stage_summary} over {total_time:.0f} ps  |  "
        f"NVT_TAU_T={NVT_TAU_T:.0f} steps ({NVT_TAU_T * TIMESTEP_FS / 1000:.1f} ps coupling)",
        fontsize=10,
    )

    # Panel 1: Temperature trace
    axes[0].plot(time_ps, thermo["T"], color="tomato", linewidth=0.8, label="Actual T")
    axes[0].plot(time_ps, target_T, color="steelblue", linewidth=1.5,
                 linestyle="--", label="Target T")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        "Oscillations → NVT_TAU_T too small; persistent drift → NVT_TAU_T too large",
        fontsize=8,
    )

    # Panel 2: Potential energy
    axes[1].plot(time_ps, thermo["U"], color="mediumseagreen", linewidth=0.8)
    axes[1].set_ylabel("Potential energy [eV]")
    axes[1].set_title("Should plateau; downward drift → increase NVT_TOTAL_TIME_PS", fontsize=8)

    # Panel 3: Pressure components
    mean_p = (thermo["Pxx"] + thermo["Pyy"] + thermo["Pzz"]) / 3.0
    axes[2].plot(time_ps, thermo["Pxx"], color="lightcoral",      linewidth=0.6, alpha=0.7, label="Pxx")
    axes[2].plot(time_ps, thermo["Pyy"], color="cornflowerblue",  linewidth=0.6, alpha=0.7, label="Pyy")
    axes[2].plot(time_ps, thermo["Pzz"], color="mediumorchid",    linewidth=0.6, alpha=0.7, label="Pzz")
    axes[2].plot(time_ps, mean_p,        color="black",           linewidth=1.2,             label="Mean P")
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("Pressure [GPa]")
    axes[2].set_xlabel("Time [ps]")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Large anisotropy or non-zero mean → residual GB stress", fontsize=8)

    # Mark stage boundaries with vertical lines
    boundary_time = 0.0
    for stage in stages[:-1]:
        boundary_time += stage["total_time_ps"]
        for ax in axes:
            ax.axvline(boundary_time, color="gray", linewidth=0.8, linestyle=":")

    plt.tight_layout()
    out_path = os.path.join(nvt_dir, "nvt_diagnostics.png")
    plt.savefig(out_path)
    plt.close()
    print(f"    NVT diagnostics saved to {out_path}")

    # Quick numeric summary
    mean_T    = thermo["T"].mean()
    std_T     = thermo["T"].std()
    drift_U   = float(thermo["U"].iloc[-1] - thermo["U"].iloc[0])
    mean_p_val = mean_p.mean()
    print(f"    NVT summary: mean T={mean_T:.1f} K (±{std_T:.1f} K), "
          f"ΔU={drift_U:.3f} eV, mean P={mean_p_val:.3f} GPa")


# ---------------------------------------------------------------------------
# Bulk Si (no-GB) helper
# ---------------------------------------------------------------------------

BULK_SI_LABEL = "bulk_si"


def build_bulk_atoms(s_input):
    """
    Build a bulk Si supercell by repeating the unit cell to get to
    the minimum BOX_SIZE.
    """
    atoms = s_input.to_ase_atoms()
    lengths = atoms.cell.lengths()  # ~[5.431, 5.431, 5.431] for conventional Si
    nx = max(int(np.ceil(BOX_SIZE[0] / lengths[0])), 1)
    ny = max(int(np.ceil(BOX_SIZE[1] / lengths[1])), 1)
    nz = max(int(np.ceil(BOX_SIZE[2] / lengths[2])), 1)
    scaling = (nx, ny, nz)
    atoms = atoms.repeat((nx, ny, nz))

    # --- Clean up any tiny off-diagonal elements so it's cubic ---
    cell = atoms.cell[:]
    np.fill_diagonal(cell, np.diag(cell))  # keep diagonal
    off_diag_mask = ~np.eye(3, dtype=bool)
    cell[off_diag_mask] = 0.0
    atoms.set_cell(cell, scale_atoms=False)

    atoms.pbc = True
    atoms.wrap()

    return atoms, scaling


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_EV_ANG2_TO_J_M2 = 16.0218


def process_gb(axis, sigma, plane, s_input, start_run=0, e_bulk_per_atom=None):
    # sigma == -1 signals a bulk-only run (no grain boundary)
    no_gb = (sigma == -1)

    if no_gb:
        label = BULK_SI_LABEL
    else:
        label = gb_label(axis, sigma, plane)

    if start_run >= N_RUNS:
        print(f"\nSkipping {label}: already completed ({N_RUNS}/{N_RUNS} runs done)")
        return

    out_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    summary_file = os.path.join(out_dir, "summary.csv")

    print(f"\n{'='*60}")
    print(f"Processing: {label}  (config: {CONFIG_NAME})")
    if no_gb:
        print(f"  Bulk Si — no grain boundary")
    else:
        print(f"  axis={axis}, sigma={sigma}, plane={plane}")
    if start_run > 0:
        print(f"  Resuming from run {start_run} ({start_run}/{N_RUNS} already done)")
    print(f"  n_runs={N_RUNS - start_run} remaining")
    npt_profile = " → ".join(f"{s['t_start']}K" for s in NPT_STAGES) + f" → {NPT_STAGES[-1]['t_end']}K"
    nvt_profile = " → ".join(f"{s['t_start']}K" for s in NVT_STAGES) + f" → {NVT_STAGES[-1]['t_end']}K"
    print(f"  NPT: {npt_profile} over {sum(s['total_time_ps'] for s in NPT_STAGES):.0f} ps (chained)")
    print(f"  NVT: {nvt_profile} over {sum(s['total_time_ps'] for s in NVT_STAGES):.0f} ps (chained)")
    print(f"{'='*60}")

    # Build initial structure
    if no_gb:
        gb_atoms, scaling = build_bulk_atoms(s_input)
        print(f"  Built bulk Si: {len(gb_atoms)} atoms after {scaling[0]}x{scaling[1]}x{scaling[2]} (XxYxZ) repeat\n"
              f"  (cell: {gb_atoms.cell[0,0]:.1f} x {gb_atoms.cell[1,1]:.1f} x {gb_atoms.cell[2,2]:.1f} Å)\n"
              f"  (goal: {BOX_SIZE[0]} x {BOX_SIZE[1]} x {BOX_SIZE[2]} Å)")
    else:
        # Build initial GB structure and repeat along X/Y for cross-section convergence.
        # This must happen before annealing so GPUMD sees a thick enough cell in all
        # periodic directions (NEP requires thickness >= 2 * cutoff = 10 Å).
        gb_atoms, scaling = build_gb_atoms(s_input, axis, sigma, plane)
        gb_atoms.wrap()
        print(f"  Built GB: {len(gb_atoms)} atoms after {scaling[0]}x{scaling[1]}x{scaling[2]} (XxYxZ) repeat\n"
              f"  (cell: {gb_atoms.cell[0,0]:.1f} x {gb_atoms.cell[1,1]:.1f} x {gb_atoms.cell[2,2]:.1f} Å)\n"
              f"  (goal: {BOX_SIZE[0]} x {BOX_SIZE[1]} x {BOX_SIZE[2]} Å)")

    if DEBUG:
        # Save initial structure for reference
        write(os.path.join(out_dir, "initial.traj"), gb_atoms)

    write_gamma = not no_gb and e_bulk_per_atom is not None

    # Open summary CSV (write fresh if starting from scratch, append if resuming)
    open_mode = "a" if start_run > 0 else "w"
    with open(summary_file, open_mode, newline="") as csvf:
        writer = csv.writer(csvf)
        if start_run == 0:
            header = ["run_index", "energy_ev"]
            if write_gamma:
                header.append("gamma_j_m2")
            writer.writerow(header)

        for i in range(start_run, N_RUNS):
            print(f"\n  Run {i+1}/{N_RUNS}...")

            run_dir = os.path.join(out_dir, f"run_{i}")
            npt_dir = os.path.join(run_dir, "npt")
            nvt_dir = os.path.join(run_dir, "nvt")
            start_atoms = gb_atoms.copy()

            # NPT: all stages chained into a single GPUMD run
            if USE_CALORINE:
                cooled_atoms, npt_energy = cool_with_gpumd(start_atoms, npt_dir, NPT_STAGES)
            else:
                cooled_atoms, npt_energy = cool_with_gpumd_direct(start_atoms, npt_dir, NPT_STAGES)
            print(f"    NPT done ({npt_profile} over {sum(s['total_time_ps'] for s in NPT_STAGES):.0f} ps). Energy = {npt_energy:.6f} eV")
            if DEBUG:
                plot_temperature_trace(npt_dir, label, i, NPT_STAGES)

            # NVT: all stages chained into a single GPUMD run
            if USE_CALORINE:
                nvt_atoms, energy = nvt_with_gpumd(cooled_atoms, nvt_dir, NVT_STAGES)
            else:
                nvt_atoms, energy = nvt_with_gpumd_direct(cooled_atoms, nvt_dir, NVT_STAGES)
            print(f"    NVT done ({nvt_profile} over {sum(s['total_time_ps'] for s in NVT_STAGES):.0f} ps). Energy = {energy:.6f} eV")
            if DEBUG:
                plot_nvt_diagnostics(nvt_dir, label, i, NVT_STAGES)

            # Attach metadata to atoms.info so downstream scripts can read it back
            if not no_gb:
                nvt_atoms.info["axis"]   = list(axis)
                nvt_atoms.info["sigma"]  = sigma
                nvt_atoms.info["plane"] = list(plane)
            nvt_atoms.info["run_index"]  = i
            nvt_atoms.info["energy_ev"]  = energy
            nvt_atoms.info["gb_label"]   = label

            row = [i, energy]
            if write_gamma:
                cell = nvt_atoms.cell[:]
                area = np.linalg.norm(cell[0]) * np.linalg.norm(cell[1])
                gamma_ev = (energy - len(nvt_atoms) * e_bulk_per_atom) / (2.0 * area)
                gamma_jm2 = gamma_ev * _EV_ANG2_TO_J_M2
                nvt_atoms.info["gamma_j_m2"] = gamma_jm2
                row.append(gamma_jm2)
                print(f"    GB energy: {gamma_jm2:.4f} J/m²")

            # Write structure to per-run traj file
            write(os.path.join(run_dir, "structure.traj"), nvt_atoms)
            writer.writerow(row)

    # Print energy summary across runs
    all_structures = [
        read(os.path.join(out_dir, f"run_{i}", "structure.traj"))
        for i in range(N_RUNS)
    ]
    energies = [s.info["energy_ev"] for s in all_structures]
    print(f"\n  Energy summary for {label}:")
    for i, e in enumerate(energies):
        marker = " <-- lowest" if e == min(energies) else ""
        print(f"    run {i}: {e:.6f} eV{marker}")
    print(f"  All {len(all_structures)} structures saved to run_*/structure.traj")


def main():
    if not os.path.exists(NEP_MODEL_FILE):
        raise FileNotFoundError(
            f"Potential not found at '{NEP_MODEL_FILE}'. "
            "Check nep_model path in config."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    gb_status = check_gb_generation_status(args.config)
    all_done = all(info["runs_remaining"] == 0 for info in gb_status.values())
    if all_done:
        print(f"All {len(GB_LIST)} entries already completed ({N_RUNS}/{N_RUNS} runs each). Nothing to do.")
        return

    print(f"Fetching Si structure from Materials Project (mp-149)...")
    s_input = Grain.from_mp_id("mp-149")

    # Bulk entries first so e_bulk is available when GB entries run.
    bulk_entries = [(ax, sg, pl) for (ax, sg, pl) in GB_LIST if sg == -1]
    gb_entries   = [(ax, sg, pl) for (ax, sg, pl) in GB_LIST if sg != -1]

    for (axis, sigma, plane) in bulk_entries:
        info = gb_status.get(BULK_SI_LABEL, {"status": "not_started", "runs_remaining": N_RUNS})
        if info["status"] == "completed":
            print(f"\nSkipping {BULK_SI_LABEL}: already completed.")
            continue
        claim_path = os.path.join(RESULTS_DIR, BULK_SI_LABEL, ".claimed")
        if not try_claim(claim_path, stale_hours=CLAIM_STALE_HOURS):
            print(f"\nSkipping {BULK_SI_LABEL}: claimed by another worker.")
            continue
        try:
            start_run = N_RUNS - info["runs_remaining"]
            process_gb(axis, sigma, plane, s_input, start_run=start_run)
        finally:
            release_claim(claim_path)

    # Load bulk reference (None if no bulk entry was specified)
    bulk_dir = os.path.join(RESULTS_DIR, BULK_SI_LABEL)
    e_bulk = None
    if os.path.isdir(bulk_dir):
        try:
            e_bulk, _ = bulk_energy_per_atom(bulk_dir)
            print(f"\nBulk reference: {e_bulk:.6f} eV/atom — will compute GB energies.")
        except FileNotFoundError:
            print("\nWarning: bulk_si results not found; summary.csv will omit gamma_j_m2.")

    for (axis, sigma, plane) in gb_entries:
        label = gb_label(axis, sigma, plane)
        info = gb_status.get(label, {"status": "not_started", "runs_remaining": N_RUNS})
        if info["status"] == "completed":
            print(f"\nSkipping {label}: already completed.")
            continue
        claim_path = os.path.join(RESULTS_DIR, label, ".claimed")
        if not try_claim(claim_path, stale_hours=CLAIM_STALE_HOURS):
            print(f"\nSkipping {label}: claimed by another worker.")
            continue
        try:
            start_run = N_RUNS - info["runs_remaining"]
            process_gb(axis, sigma, plane, s_input, start_run=start_run, e_bulk_per_atom=e_bulk)
        finally:
            release_claim(claim_path)

    print("\nAll structures processed.")


if __name__ == "__main__":
    main()
