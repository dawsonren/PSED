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
3. npt_start: NPT heating ramp from cold to production temperature.
4. nvt_anneal: NVT annealing with bulk atoms frozen via GPUMD fix 0;
               only atoms within ±gb_margin_nm of the GB plane (z = Lz/2) move.
               Temperature ramps: production → annealing → production.
               (Skipped for bulk-only entries with sigma = -1.)
5. nvt: Brief NVT equilibration at production temperature.
6. npt: Brief NPT equilibration at production temperature.
7. If optimize=True: FIRE energy minimization to 0 K; records energy_0k_ev
   and (for GB entries) gamma_0k_j_m2 using the 0K bulk reference.
8. Repeat steps 3-7 n_runs times with different random initial velocities.
9. A summary.csv records energies per run.

File outputs:
    results/<config_name>/gb_generation/
      sigma{n}_{miller}_{axis}/
        run_0/
          npt_start/           <- NPT heating ramp GPUMD directory
          nvt_anneal/          <- NVT anneal with frozen bulk (GB entries only)
          nvt/                 <- brief NVT equilibration
          npt/                 <- brief NPT equilibration
          fire/                <- FIRE optimization (if optimize=True)
          structure.traj       <- final NPT-equilibrated structure
          structure_0k.traj    <- FIRE-optimized structure (if optimize=True)
        run_1/
          ...
        summary.csv  <- run_index, energy_ev, [gamma_j_m2], [energy_0k_ev], [gamma_0k_j_m2]
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
BOX_SIZE    = np.array([float(gb_cfg["x_nm"]) * 10, float(gb_cfg["y_nm"]) * 10, float(gb_cfg["z_nm"]) * 10])
N_RUNS      = int(gb_cfg["n_runs"])
TIMESTEP_FS = float(gb_cfg["timestep_fs"])
DEBUG       = bool(gb_cfg.get("debug", False))
OPTIMIZE    = bool(gb_cfg.get("optimize", False))

def _as_list(v):
    return v if isinstance(v, list) else [v]

_dump_interval_cfg = int(gb_cfg["dump_interval"]) if DEBUG else None

def _make_stages(t_starts, t_ends, times_ps):
    stages = []
    for ts, te, tp in zip(t_starts, t_ends, times_ps):
        if tp == 0:
            continue
        n = int(tp * 1000.0 / TIMESTEP_FS)
        stages.append({
            "t_start": ts, "t_end": te, "total_time_ps": tp,
            "n_steps": n,
            "thermo_interval": max(int(n / 100), 1),
            "dump_interval": min(_dump_interval_cfg, n - 1) if DEBUG else n - 1,
        })
    return stages

# NPT start: heating ramp from cold to production temperature
npt_start_cfg    = gb_cfg["npt_start"]
NPT_START_STAGES = _make_stages(
    [float(x) for x in _as_list(npt_start_cfg["t_start"])],
    [float(x) for x in _as_list(npt_start_cfg["t_end"])],
    [float(x) for x in _as_list(npt_start_cfg["total_time_ps"])],
)
_NPT_START_PARAMS = {
    "tau_t":            float(npt_start_cfg["tau_t"]),
    "pressure_gpa":     float(npt_start_cfg["pressure_gpa"]),
    "bulk_modulus_gpa": float(npt_start_cfg["bulk_modulus_gpa"]),
    "tau_p":            float(npt_start_cfg["tau_p"]),
}

# NVT anneal: bulk atoms frozen (group 0 via fix 0), GB-region atoms mobile (group 1)
nvt_anneal_cfg    = gb_cfg["nvt_anneal"]
NVT_ANNEAL_STAGES = _make_stages(
    [float(x) for x in _as_list(nvt_anneal_cfg["t_start"])],
    [float(x) for x in _as_list(nvt_anneal_cfg["t_end"])],
    [float(x) for x in _as_list(nvt_anneal_cfg["total_time_ps"])],
)
NVT_ANNEAL_TAU_T = float(nvt_anneal_cfg["tau_t"])
GB_MARGIN_ANG    = float(nvt_anneal_cfg["gb_margin_nm"]) * 10.0

# Brief NVT equilibration
nvt_cfg    = gb_cfg["nvt"]
NVT_TAU_T  = float(nvt_cfg["tau_t"])
NVT_STAGES = _make_stages(
    [float(x) for x in _as_list(nvt_cfg["t_start"])],
    [float(x) for x in _as_list(nvt_cfg["t_end"])],
    [float(x) for x in _as_list(nvt_cfg["total_time_ps"])],
)

# Brief NPT equilibration
npt_cfg    = gb_cfg["npt"]
NPT_STAGES = _make_stages(
    [float(x) for x in _as_list(npt_cfg["t_start"])],
    [float(x) for x in _as_list(npt_cfg["t_end"])],
    [float(x) for x in _as_list(npt_cfg["total_time_ps"])],
)
_NPT_PARAMS = {
    "tau_t":            float(npt_cfg["tau_t"]),
    "pressure_gpa":     float(npt_cfg["pressure_gpa"]),
    "bulk_modulus_gpa": float(npt_cfg["bulk_modulus_gpa"]),
    "tau_p":            float(npt_cfg["tau_p"]),
}

_raw_gbs = config["grain_boundaries"]
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
    np.fill_diagonal(cell, np.diag(cell))
    off_diag_mask = ~np.eye(3, dtype=bool)
    cell[off_diag_mask] = 0.0
    atoms.set_cell(cell, scale_atoms=False)

    atoms.pbc = True
    atoms.wrap()

    return atoms, (scale_x, scale_y, UC)


def _assign_gb_groups(atoms):
    """
    Assign GPUMD group labels for the nvt_anneal freeze step.

    Group 0 (frozen via fix 0): bulk atoms farther than GB_MARGIN_ANG from the GB.
    Group 1 (mobile):           atoms within ±GB_MARGIN_ANG of the GB at z = Lz/2.

    Uses PBC-aware distance so the half-periodic images are treated correctly.
    """
    lz = atoms.cell[2, 2]
    z_gb = lz / 2.0
    dz = np.abs(atoms.positions[:, 2] - z_gb)
    dz = np.minimum(dz, lz - dz)  # PBC along z
    groups = np.where(dz <= GB_MARGIN_ANG, 1, 0).astype(np.int32)
    n_mobile = int(np.sum(groups == 1))
    n_frozen = int(np.sum(groups == 0))
    print(f"    GB groups: {n_mobile} mobile atoms (within {GB_MARGIN_ANG/10:.1f} nm of GB), "
          f"{n_frozen} frozen bulk atoms")
    return groups


# ---------------------------------------------------------------------------
# GPUMD runners (calorine path)
# ---------------------------------------------------------------------------

def cool_with_gpumd(atoms, npt_dir, stages, npt_params):
    """
    Run NPT npt_scr stages as a single chained GPUMD run (calorine path).
    """
    os.makedirs(npt_dir, exist_ok=True)

    movie_path = os.path.join(npt_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    tau_t            = npt_params["tau_t"]
    pressure_gpa     = npt_params["pressure_gpa"]
    bulk_modulus_gpa = npt_params["bulk_modulus_gpa"]
    tau_p            = npt_params["tau_p"]

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=npt_dir,
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
            ("ensemble",     ["npt_scr", stage["t_start"], stage["t_end"],
                              tau_t, pressure_gpa, bulk_modulus_gpa, tau_p]),
            ("dump_thermo",  stage["thermo_interval"]),
            ("dump_position", stage["dump_interval"]),
            ("run",          stage["n_steps"]),
        ]

    cooled_atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    thermo_data = np.loadtxt(os.path.join(npt_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return cooled_atoms, energy_ev


def nvt_anneal_with_gpumd(atoms, anneal_dir, stages):
    """
    Run NVT nvt_nhc anneal stages with bulk atoms frozen via fix 0 (calorine path).

    Atoms within GB_MARGIN_ANG of z = Lz/2 are assigned group 1 (mobile).
    All other atoms are group 0 and frozen by GPUMD's fix 0 each run block.
    Group labels are written into the model.xyz Properties as group:I:1.
    """
    os.makedirs(anneal_dir, exist_ok=True)

    movie_path = os.path.join(anneal_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    atoms = atoms.copy()
    atoms.set_array("group", _assign_gb_groups(atoms))

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=anneal_dir,
        atoms=atoms,
    )
    atoms.calc = calc

    md_params = [
        ("velocity",  stages[0]["t_start"]),
        ("time_step", TIMESTEP_FS),
    ]
    for stage in stages:
        md_params += [
            ("fix",          0),   # freeze group-0 (bulk) atoms for this run block
            ("ensemble",     ["nvt_nhc", stage["t_start"], stage["t_end"], NVT_ANNEAL_TAU_T]),
            ("dump_thermo",  stage["thermo_interval"]),
            ("dump_position", stage["dump_interval"]),
            ("run",          stage["n_steps"]),
        ]

    anneal_atoms = calc.run_custom_md(md_params, return_last_atoms=True)
    anneal_atoms.pbc = atoms.pbc
    anneal_atoms.wrap()
    anneal_atoms.arrays.pop("group", None)  # don't carry group labels into subsequent steps

    thermo_data = np.loadtxt(os.path.join(anneal_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return anneal_atoms, energy_ev


def nvt_with_gpumd(atoms, nvt_dir, stages, nvt_tau_t):
    """
    Run NVT nvt_nhc stages as a single chained GPUMD run (calorine path).
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
            ("ensemble",     ["nvt_nhc", stage["t_start"], stage["t_end"], nvt_tau_t]),
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


# ---------------------------------------------------------------------------
# GPUMD runners (direct executable path)
# ---------------------------------------------------------------------------

def cool_with_gpumd_direct(atoms, npt_dir, stages, npt_params):
    """
    Run NPT npt_scr stages as a single chained GPUMD run (direct executable).
    """
    os.makedirs(npt_dir, exist_ok=True)

    for fname in ("thermo.out", "movie.xyz"):
        fpath = os.path.join(npt_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    if not atoms.has("mass"):
        atoms.new_array("mass", atoms.get_masses())

    write(os.path.join(npt_dir, "model.xyz"), atoms, format="extxyz")

    tau_t            = npt_params["tau_t"]
    pressure_gpa     = npt_params["pressure_gpa"]
    bulk_modulus_gpa = npt_params["bulk_modulus_gpa"]
    tau_p            = npt_params["tau_p"]

    rel_potential = os.path.relpath(NEP_MODEL_FILE, npt_dir)
    lines = [
        f"potential {rel_potential}",
        f"velocity {stages[0]['t_start']}",
        f"time_step {TIMESTEP_FS}",
    ]
    for stage in stages:
        lines += [
            f"ensemble npt_scr {stage['t_start']} {stage['t_end']} "
            f"{tau_t} {pressure_gpa} {bulk_modulus_gpa} {tau_p}",
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

    cooled_atoms = read(os.path.join(npt_dir, "movie.xyz"), index=-1, format="extxyz")
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    thermo_data = np.loadtxt(os.path.join(npt_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return cooled_atoms, energy_ev


def nvt_anneal_with_gpumd_direct(atoms, anneal_dir, stages):
    """
    Run NVT nvt_nhc anneal stages with bulk atoms frozen via fix 0 (direct executable).

    Group labels are written into model.xyz as the 'group' integer property so GPUMD
    can assign atoms to grouping method 0. fix 0 before each run block freezes group-0
    (bulk) atoms; group-1 (GB-region) atoms remain mobile.
    """
    os.makedirs(anneal_dir, exist_ok=True)

    for fname in ("thermo.out", "movie.xyz"):
        fpath = os.path.join(anneal_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    atoms = atoms.copy()
    atoms.set_array("group", _assign_gb_groups(atoms))

    if not atoms.has("mass"):
        atoms.new_array("mass", atoms.get_masses())

    write(os.path.join(anneal_dir, "model.xyz"), atoms, format="extxyz")

    rel_potential = os.path.relpath(NEP_MODEL_FILE, anneal_dir)
    lines = [
        f"potential {rel_potential}",
        f"velocity {stages[0]['t_start']}",
        f"time_step {TIMESTEP_FS}",
    ]
    for stage in stages:
        lines += [
            f"fix 0",   # freeze group-0 (bulk) atoms for this run block
            f"ensemble nvt_nhc {stage['t_start']} {stage['t_end']} {NVT_ANNEAL_TAU_T}",
            f"dump_thermo {stage['thermo_interval']}",
            f"dump_position {stage['dump_interval']}",
            f"run {stage['n_steps']}",
        ]
    with open(os.path.join(anneal_dir, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(anneal_dir, "stdout"), "w") as stdout_f:
        result = subprocess.run(
            [GPUMD_EXEC], cwd=anneal_dir,
            stdout=stdout_f, stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPUMD NVT anneal failed (rc={result.returncode}):\n"
            + result.stderr.decode()
        )

    anneal_atoms = read(os.path.join(anneal_dir, "movie.xyz"), index=-1, format="extxyz")
    anneal_atoms.pbc = atoms.pbc
    anneal_atoms.wrap()

    thermo_data = np.loadtxt(os.path.join(anneal_dir, "thermo.out"))
    energy_ev = float(thermo_data[2] if thermo_data.ndim == 1 else thermo_data[-1, 2])

    return anneal_atoms, energy_ev


def nvt_with_gpumd_direct(atoms, nvt_dir, stages, nvt_tau_t):
    """
    Run NVT nvt_nhc stages as a single chained GPUMD run (direct executable).
    """
    os.makedirs(nvt_dir, exist_ok=True)

    for fname in ("thermo.out", "movie.xyz"):
        fpath = os.path.join(nvt_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    if not atoms.has("mass"):
        atoms.new_array("mass", atoms.get_masses())

    write(os.path.join(nvt_dir, "model.xyz"), atoms, format="extxyz")

    rel_potential = os.path.relpath(NEP_MODEL_FILE, nvt_dir)
    lines = [
        f"potential {rel_potential}",
        f"velocity {stages[0]['t_start']}",
        f"time_step {TIMESTEP_FS}",
    ]
    for stage in stages:
        lines += [
            f"ensemble nvt_nhc {stage['t_start']} {stage['t_end']} {nvt_tau_t}",
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


# ---------------------------------------------------------------------------
# FIRE optimization (0 K)
# ---------------------------------------------------------------------------

def optimize_with_fire(atoms, opt_dir):
    """
    Run FIRE energy minimization to a local 0 K energy minimum using the NEP potential.

    Uses GPUNEP as the ASE calculator backend (requires use_calorine=True).
    Returns (optimized_atoms, energy_0k_ev).
    """
    from ase.optimize import FIRE

    os.makedirs(opt_dir, exist_ok=True)

    atoms = atoms.copy()
    atoms.arrays.pop("group", None)  # strip group labels from the anneal step

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=opt_dir,
        atoms=atoms,
    )
    atoms.calc = calc

    traj_path = os.path.join(opt_dir, "fire.traj")
    log_path  = os.path.join(opt_dir, "fire.log")

    dyn = FIRE(atoms, trajectory=traj_path, logfile=log_path, maxstep=0.1)
    dyn.run(fmax=0.05)

    energy_0k = float(atoms.get_potential_energy())
    print(f"    FIRE converged: E_0K = {energy_0k:.6f} eV  ({dyn.get_number_of_steps()} steps)")
    return atoms, energy_0k


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _build_piecewise_target(stages, n_actual_rows):
    """
    Build time_ps and piecewise-linear target_T arrays for a chained thermo.out.

    Each stage contributes floor(n_steps / thermo_interval) rows; the last stage
    absorbs any rounding surplus so the arrays always match n_actual_rows.
    """
    n_expected = [s["n_steps"] // s["thermo_interval"] for s in stages]

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


_STAGE_COLORS = {
    "npt_start":  "cornflowerblue",
    "nvt_anneal": "gold",
    "nvt":        "mediumseagreen",
    "npt":        "tomato",
}


def plot_run_diagnostics(run_dir, out_dir, label, run_index, no_gb):
    """
    Single combined diagnostic plot: temperature and mean pressure across all MD stages.

    Reads thermo.out from npt_start/, nvt_anneal/ (GB only), nvt/, and npt/,
    stitches them into a continuous time axis, and saves the figure to
    out_dir/run_{run_index}_diagnostics.png (next to summary.csv).

    thermo.out columns (GPUMD format):
        T  K  U  Pxx Pyy Pzz Pyz Pxz Pxy  ax ay az  bx by bz  cx cy cz
    """
    stages_info = []
    if NPT_START_STAGES:
        stages_info.append((os.path.join(run_dir, "npt_start"), NPT_START_STAGES, "npt_start"))
    if not no_gb and NVT_ANNEAL_STAGES:
        stages_info.append((os.path.join(run_dir, "nvt_anneal"), NVT_ANNEAL_STAGES, "nvt_anneal"))
    if NVT_STAGES:
        stages_info.append((os.path.join(run_dir, "nvt"), NVT_STAGES, "nvt"))
    if NPT_STAGES:
        stages_info.append((os.path.join(run_dir, "npt"), NPT_STAGES, "npt"))

    all_time, all_T, all_P = [], [], []
    stage_spans = []  # (t_start, t_end, name) for background shading
    cumulative = 0.0

    for stage_dir, stages, stage_name in stages_info:
        stage_duration = sum(s["total_time_ps"] for s in stages)
        path = os.path.join(stage_dir, "thermo.out")
        if not os.path.exists(path):
            print(f"    Warning: thermo.out not found in {stage_dir}; skipping in diagnostics.")
            cumulative += stage_duration
            continue
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        T = data[:, 0]
        P = (data[:, 3] + data[:, 4] + data[:, 5]) / 3.0
        time_ps, _ = _build_piecewise_target(stages, len(data))
        stage_spans.append((cumulative, cumulative + stage_duration, stage_name))
        all_time.append(time_ps + cumulative)
        all_T.append(T)
        all_P.append(P)
        cumulative += stage_duration

    if not all_time:
        print(f"    Warning: no thermo data found for run {run_index}; skipping diagnostics.")
        return

    time  = np.concatenate(all_time)
    T_arr = np.concatenate(all_T)
    P_arr = np.concatenate(all_P)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{label} — run {run_index} diagnostics", fontsize=11)

    for t0, t1, name in stage_spans:
        color = _STAGE_COLORS.get(name, "lightgray")
        for ax in axes:
            ax.axvspan(t0, t1, alpha=0.15, color=color, label=name)

    axes[0].plot(time, T_arr, color="black", linewidth=0.8)
    axes[0].set_ylabel("Temperature [K]")

    axes[1].plot(time, P_arr, color="black", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Mean pressure [GPa]")
    axes[1].set_xlabel("Time [ps]")

    # Stage legend from the shaded regions (deduplicated)
    handles, lbls = axes[0].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    axes[0].legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"run_{run_index}_diagnostics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Diagnostics saved to {out_path}")


# ---------------------------------------------------------------------------
# Bulk Si (no-GB) helper
# ---------------------------------------------------------------------------

BULK_SI_LABEL = "bulk_si"


def build_bulk_atoms(s_input):
    atoms = s_input.to_ase_atoms()
    lengths = atoms.cell.lengths()
    nx = max(int(np.ceil(BOX_SIZE[0] / lengths[0])), 1)
    ny = max(int(np.ceil(BOX_SIZE[1] / lengths[1])), 1)
    nz = max(int(np.ceil(BOX_SIZE[2] / lengths[2])), 1)
    scaling = (nx, ny, nz)
    atoms = atoms.repeat((nx, ny, nz))

    cell = atoms.cell[:]
    np.fill_diagonal(cell, np.diag(cell))
    off_diag_mask = ~np.eye(3, dtype=bool)
    cell[off_diag_mask] = 0.0
    atoms.set_cell(cell, scale_atoms=False)

    atoms.pbc = True
    atoms.wrap()

    return atoms, scaling


def _bulk_energy_0k_per_atom(bulk_results_dir):
    """
    Mean 0K bulk energy per atom from FIRE-optimized bulk runs.

    Reads energy_0k_ev from structure.traj info dicts; returns None if no 0K
    energies were recorded (i.e., optimize=False was used for the bulk run).
    """
    bulk_dir = Path(bulk_results_dir)
    run_trajs = sorted(bulk_dir.glob("run_*/structure.traj"))
    if not run_trajs:
        return None
    e_per_atom = []
    for traj in run_trajs:
        atoms = read(str(traj))
        if "energy_0k_ev" in atoms.info:
            e_per_atom.append(atoms.info["energy_0k_ev"] / len(atoms))
    if not e_per_atom:
        return None
    return float(np.mean(e_per_atom))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_EV_ANG2_TO_J_M2 = 16.0218


def process_gb(axis, sigma, plane, s_input, start_run=0,
               e_bulk_per_atom=None, e_bulk_0k_per_atom=None):
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

    npt_start_profile = (" → ".join(f"{s['t_start']}K" for s in NPT_START_STAGES)
                         + f" → {NPT_START_STAGES[-1]['t_end']}K") if NPT_START_STAGES else ""
    anneal_profile    = (" → ".join(f"{s['t_start']}K" for s in NVT_ANNEAL_STAGES)
                         + f" → {NVT_ANNEAL_STAGES[-1]['t_end']}K") if NVT_ANNEAL_STAGES else ""
    nvt_profile       = (" → ".join(f"{s['t_start']}K" for s in NVT_STAGES)
                         + f" → {NVT_STAGES[-1]['t_end']}K") if NVT_STAGES else ""
    npt_profile       = (" → ".join(f"{s['t_start']}K" for s in NPT_STAGES)
                         + f" → {NPT_STAGES[-1]['t_end']}K") if NPT_STAGES else ""

    if NPT_START_STAGES:
        print(f"  npt_start : {npt_start_profile} over "
              f"{sum(s['total_time_ps'] for s in NPT_START_STAGES):.0f} ps")
    if not no_gb and NVT_ANNEAL_STAGES:
        print(f"  nvt_anneal: {anneal_profile} over "
              f"{sum(s['total_time_ps'] for s in NVT_ANNEAL_STAGES):.0f} ps  (fix 0 on bulk atoms)")
    if NVT_STAGES:
        print(f"  nvt       : {nvt_profile} over "
              f"{sum(s['total_time_ps'] for s in NVT_STAGES):.0f} ps")
    if NPT_STAGES:
        print(f"  npt       : {npt_profile} over "
              f"{sum(s['total_time_ps'] for s in NPT_STAGES):.0f} ps")
    if OPTIMIZE:
        print(f"  fire      : FIRE fmax=0.05 eV/Å → 0 K energy")
    print(f"{'='*60}")

    if no_gb:
        gb_atoms, scaling = build_bulk_atoms(s_input)
        print(f"  Built bulk Si: {len(gb_atoms)} atoms after "
              f"{scaling[0]}x{scaling[1]}x{scaling[2]} (XxYxZ) repeat\n"
              f"  (cell: {gb_atoms.cell[0,0]:.1f} x {gb_atoms.cell[1,1]:.1f} x {gb_atoms.cell[2,2]:.1f} Å)\n"
              f"  (goal: {BOX_SIZE[0]} x {BOX_SIZE[1]} x {BOX_SIZE[2]} Å)")
    else:
        gb_atoms, scaling = build_gb_atoms(s_input, axis, sigma, plane)
        gb_atoms.wrap()
        print(f"  Built GB: {len(gb_atoms)} atoms after "
              f"{scaling[0]}x{scaling[1]}x{scaling[2]} (XxYxZ) repeat\n"
              f"  (cell: {gb_atoms.cell[0,0]:.1f} x {gb_atoms.cell[1,1]:.1f} x {gb_atoms.cell[2,2]:.1f} Å)\n"
              f"  (goal: {BOX_SIZE[0]} x {BOX_SIZE[1]} x {BOX_SIZE[2]} Å)")

    if DEBUG:
        write(os.path.join(out_dir, "initial.traj"), gb_atoms)

    write_gamma    = not no_gb and e_bulk_per_atom is not None
    write_gamma_0k = OPTIMIZE and not no_gb and e_bulk_0k_per_atom is not None

    open_mode = "a" if start_run > 0 else "w"
    with open(summary_file, open_mode, newline="") as csvf:
        writer = csv.writer(csvf)
        if start_run == 0:
            header = ["run_index", "energy_ev"]
            if write_gamma:
                header.append("gamma_j_m2")
            if OPTIMIZE:
                header.append("energy_0k_ev")
            if write_gamma_0k:
                header.append("gamma_0k_j_m2")
            writer.writerow(header)

        for i in range(start_run, N_RUNS):
            print(f"\n  Run {i+1}/{N_RUNS}...")

            run_dir       = os.path.join(out_dir, f"run_{i}")
            npt_start_dir = os.path.join(run_dir, "npt_start")
            anneal_dir    = os.path.join(run_dir, "nvt_anneal")
            nvt_dir       = os.path.join(run_dir, "nvt")
            npt_dir       = os.path.join(run_dir, "npt")
            fire_dir      = os.path.join(run_dir, "fire")
            start_atoms   = gb_atoms.copy()

            energy = None

            # Step 1: NPT heating ramp from cold to production temperature
            if NPT_START_STAGES:
                if USE_CALORINE:
                    warm_atoms, energy = cool_with_gpumd(
                        start_atoms, npt_start_dir, NPT_START_STAGES, _NPT_START_PARAMS)
                else:
                    warm_atoms, energy = cool_with_gpumd_direct(
                        start_atoms, npt_start_dir, NPT_START_STAGES, _NPT_START_PARAMS)
                print(f"    npt_start done ({npt_start_profile} over "
                      f"{sum(s['total_time_ps'] for s in NPT_START_STAGES):.0f} ps). "
                      f"Energy = {energy:.6f} eV")
            else:
                warm_atoms = start_atoms

            # Step 2: NVT anneal with frozen bulk (GB entries only)
            if not no_gb and NVT_ANNEAL_STAGES:
                if USE_CALORINE:
                    anneal_atoms, energy = nvt_anneal_with_gpumd(
                        warm_atoms, anneal_dir, NVT_ANNEAL_STAGES)
                else:
                    anneal_atoms, energy = nvt_anneal_with_gpumd_direct(
                        warm_atoms, anneal_dir, NVT_ANNEAL_STAGES)
                print(f"    nvt_anneal done ({anneal_profile} over "
                      f"{sum(s['total_time_ps'] for s in NVT_ANNEAL_STAGES):.0f} ps). "
                      f"Energy = {energy:.6f} eV")
            else:
                anneal_atoms = warm_atoms

            # Step 3: Brief NVT equilibration (all atoms free)
            if NVT_STAGES:
                if USE_CALORINE:
                    nvt_atoms, energy = nvt_with_gpumd(
                        anneal_atoms, nvt_dir, NVT_STAGES, NVT_TAU_T)
                else:
                    nvt_atoms, energy = nvt_with_gpumd_direct(
                        anneal_atoms, nvt_dir, NVT_STAGES, NVT_TAU_T)
                print(f"    nvt done ({nvt_profile} over "
                      f"{sum(s['total_time_ps'] for s in NVT_STAGES):.0f} ps). "
                      f"Energy = {energy:.6f} eV")
            else:
                nvt_atoms = anneal_atoms

            # Step 4: Brief NPT equilibration
            if NPT_STAGES:
                if USE_CALORINE:
                    final_atoms, energy = cool_with_gpumd(
                        nvt_atoms, npt_dir, NPT_STAGES, _NPT_PARAMS)
                else:
                    final_atoms, energy = cool_with_gpumd_direct(
                        nvt_atoms, npt_dir, NPT_STAGES, _NPT_PARAMS)
                print(f"    npt done ({npt_profile} over "
                      f"{sum(s['total_time_ps'] for s in NPT_STAGES):.0f} ps). "
                      f"Energy = {energy:.6f} eV")
            else:
                final_atoms = nvt_atoms

            # Step 5: FIRE optimization to 0 K (optional)
            energy_0k = None
            if OPTIMIZE:
                if not USE_CALORINE:
                    print("    Warning: FIRE optimization requires use_calorine=True; skipping.")
                else:
                    opt_atoms, energy_0k = optimize_with_fire(final_atoms, fire_dir)
                    final_atoms.info["energy_0k_ev"] = energy_0k
                    write(os.path.join(run_dir, "structure_0k.traj"), opt_atoms)

            # Attach metadata
            if not no_gb:
                final_atoms.info["axis"]   = list(axis)
                final_atoms.info["sigma"]  = sigma
                final_atoms.info["plane"]  = list(plane)
            final_atoms.info["run_index"] = i
            final_atoms.info["energy_ev"] = energy
            final_atoms.info["gb_label"]  = label

            row = [i, energy]

            if write_gamma:
                cell    = final_atoms.cell[:]
                area    = np.linalg.norm(cell[0]) * np.linalg.norm(cell[1])
                gamma_ev  = (energy - len(final_atoms) * e_bulk_per_atom) / (2.0 * area)
                gamma_jm2 = gamma_ev * _EV_ANG2_TO_J_M2
                final_atoms.info["gamma_j_m2"] = gamma_jm2
                row.append(gamma_jm2)
                print(f"    GB energy (thermal): {gamma_jm2:.4f} J/m²")

            if OPTIMIZE:
                row.append(energy_0k if energy_0k is not None else "")

            if write_gamma_0k and energy_0k is not None:
                cell         = final_atoms.cell[:]
                area         = np.linalg.norm(cell[0]) * np.linalg.norm(cell[1])
                gamma_0k_ev  = (energy_0k - len(final_atoms) * e_bulk_0k_per_atom) / (2.0 * area)
                gamma_0k_jm2 = gamma_0k_ev * _EV_ANG2_TO_J_M2
                final_atoms.info["gamma_0k_j_m2"] = gamma_0k_jm2
                row.append(gamma_0k_jm2)
                print(f"    GB energy (0 K):     {gamma_0k_jm2:.4f} J/m²")

            write(os.path.join(run_dir, "structure.traj"), final_atoms)
            writer.writerow(row)

            if DEBUG:
                plot_run_diagnostics(run_dir, out_dir, label, i, no_gb)

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

    # Load thermal bulk reference
    bulk_dir = os.path.join(RESULTS_DIR, BULK_SI_LABEL)
    e_bulk = None
    if os.path.isdir(bulk_dir):
        try:
            e_bulk, _ = bulk_energy_per_atom(bulk_dir)
            print(f"\nBulk reference (thermal): {e_bulk:.6f} eV/atom")
        except FileNotFoundError:
            print("\nWarning: bulk_si results not found; summary.csv will omit gamma_j_m2.")

    # Load 0K bulk reference (only available if bulk was run with optimize=True)
    e_bulk_0k = None
    if OPTIMIZE and os.path.isdir(bulk_dir):
        e_bulk_0k = _bulk_energy_0k_per_atom(bulk_dir)
        if e_bulk_0k is not None:
            print(f"Bulk reference (0 K):     {e_bulk_0k:.6f} eV/atom")
        else:
            print("Warning: no 0K bulk energies found; gamma_0k_j_m2 will be omitted from summary.")

    for (axis, sigma, plane) in gb_entries:
        label = gb_label(axis, sigma, plane)
        info  = gb_status.get(label, {"status": "not_started", "runs_remaining": N_RUNS})
        if info["status"] == "completed":
            print(f"\nSkipping {label}: already completed.")
            continue
        claim_path = os.path.join(RESULTS_DIR, label, ".claimed")
        if not try_claim(claim_path, stale_hours=CLAIM_STALE_HOURS):
            print(f"\nSkipping {label}: claimed by another worker.")
            continue
        try:
            start_run = N_RUNS - info["runs_remaining"]
            process_gb(axis, sigma, plane, s_input, start_run=start_run,
                       e_bulk_per_atom=e_bulk, e_bulk_0k_per_atom=e_bulk_0k)
        finally:
            release_claim(claim_path)

    print("\nAll structures processed.")


if __name__ == "__main__":
    main()
