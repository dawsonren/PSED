"""
Use aimsgb to generate grain boundaries, then use GPUMD to relax
such structures into realistic crystalline configurations.

Usage:
    python gb_generation/generate_gbs.py --config configs/small_box.yaml
    python gb_generation/generate_gbs.py --config configs/large_box.yaml

Pipeline:
1. Load GB specifications and run parameters from a unified YAML config.
2. Build GB structure with aimsgb (GrainBoundary + Grain.stack_grains).
   The x/y/z lengths repeated to achieve the correct BOX_SIZE
3. Relax the structure using LBFGS.
4. Anneal with GPUMD: cooling ramp from t_start to t_end over
   total_time_ps, using npt_scr thermostat.
5. Repeat step 4 n_runs times with different random initial velocities.
   All relaxed structures are saved as .traj files.
6. A summary.csv records energies per run.

File outputs:
    results/<config_name>/gb_generation/
      sigma{n}_{miller}_{axis}/
        run_0/          <- GPUMD working directory
          movie.xyz
          thermo.out
        run_1/
          ...
        summary.csv     <- run_index, energy_ev per row
"""

import os
import csv
import argparse
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
from ase.optimize import LBFGS

from calorine.calculators import GPUNEP, CPUNEP

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.work_coordination import gb_label, check_gb_generation_status

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
RESULTS_DIR    = str(GPUMD_ROOT / "results" / CONFIG_NAME / "gb_generation")

gb_cfg = config["gb_generation"]
# minimum length of supercell in x/y/z axes in angstroms
BOX_SIZE         = np.array([float(gb_cfg["x_nm"]) * 10, float(gb_cfg["y_nm"]) * 10, float(gb_cfg["z_nm"]) * 10])
N_RUNS           = int(gb_cfg["n_runs"])
F_MAX            = float(gb_cfg.get("f_max", 0.1))
STEPS            = int(gb_cfg.get("steps", 500))
T_START          = float(gb_cfg["t_start"])
T_END            = float(gb_cfg["t_end"])
TOTAL_TIME_PS    = float(gb_cfg["total_time_ps"])
TIMESTEP_FS      = float(gb_cfg["timestep_fs"])
TAU_T            = float(gb_cfg["tau_t"])
PRESSURE_GPA     = float(gb_cfg["pressure_gpa"])
BULK_MODULUS_GPA = float(gb_cfg["bulk_modulus_gpa"])
TAU_P            = float(gb_cfg["tau_p"])
DEBUG            = bool(gb_cfg.get("debug", False))

# Derived
N_STEPS         = int(TOTAL_TIME_PS * 1000.0 / TIMESTEP_FS)
DUMP_INTERVAL   = int(gb_cfg["dump_interval"]) if DEBUG else N_STEPS - 1  # only dump at the end if not debugging
THERMO_INTERVAL = max(int(N_STEPS / 100), 1)  # aim for ~100 thermo points per run

_raw_gbs = config["grain_boundaries"]
# sigma: -1 in the config signals a bulk-only run (no grain boundary)
NO_GB_MODE = len(_raw_gbs) == 1 and _raw_gbs[0]["sigma"] == -1
GB_LIST = [] if NO_GB_MODE else [
    (tuple(entry["axis"]), int(entry["sigma"]), tuple(entry["plane"]))
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


def relax_with_lbfgs(atoms, fmax=F_MAX, steps=STEPS):
    """
    Relax atomic positions with LBFGS using CPUNEP.

    Removes the worst of the high-energy overlaps that aimsgb introduces at
    the grain boundary interface before handing off to the GPUMD annealing
    step.  This keeps the MD from immediately exploding due to unphysical
    forces.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to relax (will NOT be modified; a copy is returned).
    fmax  : float
        Force convergence threshold in eV/Å.
    steps : int
        Maximum number of LBFGS steps.

    Returns
    -------
    relaxed : ase.Atoms  (calculator detached)
    """
    relaxed = atoms.copy()
    relaxed.calc = CPUNEP(NEP_MODEL_FILE)
    opt = LBFGS(relaxed, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    relaxed.calc = None  # detach so the GPUMD calc can be attached cleanly
    return relaxed


def cool_with_gpumd(atoms, run_dir):
    """
    Run a GPUMD npt_scr cooling ramp: T_START -> T_END over TOTAL_TIME_PS.

    TAU_T (in timesteps) sets the coupling timescale (recommended
    to be 100 x timestep in GPUMD). Too small causes unphysical velocity kicks;
    too large and the temperature lags the ramp target.
    """
    os.makedirs(run_dir, exist_ok=True)

    # Remove stale movie.xyz so calorine reads the correct run's output
    movie_path = os.path.join(run_dir, "movie.xyz")
    if os.path.exists(movie_path):
        os.remove(movie_path)

    calc = GPUNEP(
        NEP_MODEL_FILE,
        command=GPUMD_EXEC,
        gpu_identifier_index=0,
        directory=run_dir,
        atoms=atoms,
    )
    atoms = atoms.copy()
    atoms.calc = calc

    md_params = [
        ("velocity",  T_START),
        ("time_step", TIMESTEP_FS),
        ("ensemble",  ["npt_scr", T_START, T_END, TAU_T, PRESSURE_GPA, BULK_MODULUS_GPA, TAU_P]),
        ("dump_position", DUMP_INTERVAL),
        ("run", N_STEPS),
    ]

    if DEBUG:
        md_params.insert(3, ("dump_thermo", THERMO_INTERVAL))

    cooled_atoms = calc.run_custom_md(
        md_params,
        return_last_atoms=True
    )
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    return cooled_atoms


def plot_temperature_trace(run_dir, label, run_index):
    """
    Plot actual vs target temperature from thermo.out to validate TAU_T.

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

    n_thermo_steps = len(thermo)
    # Time axis: each thermo row corresponds to THERMO_INTERVAL MD steps
    time_ps = np.arange(n_thermo_steps) * THERMO_INTERVAL * TIMESTEP_FS / 1000.0

    # Reconstruct the linear cooling ramp as the target
    target_T = np.linspace(T_START, T_END, n_thermo_steps)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.suptitle(
        f"{label} — run {run_index} temperature trace\n"
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

def process_gb(axis, sigma, plane, s_input, start_run=0):
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
    print(f"  n_runs={N_RUNS - start_run} remaining, T: {T_START}K -> {T_END}K over {TOTAL_TIME_PS}ps")
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

    # Open summary CSV (write fresh if starting from scratch, append if resuming)
    open_mode = "a" if start_run > 0 else "w"
    with open(summary_file, open_mode, newline="") as csvf:
        writer = csv.writer(csvf)
        if start_run == 0:
            writer.writerow(["run_index", "energy_ev"])

        for i in range(start_run, N_RUNS):
            print(f"\n  Run {i+1}/{N_RUNS}...")

            run_dir = os.path.join(out_dir, f"run_{i}")
            start_atoms = gb_atoms.copy()

            # Step 4 — LBFGS pre-relaxation
            # Insert between structure build (step 3) and GPUMD annealing (step 5).
            # This quenches the high-energy atoms at the GB interface so the MD
            # timestep doesn't blow up on the first few steps.
            print(f"    Relaxing with LBFGS...")
            start_atoms = relax_with_lbfgs(start_atoms)

            # Anneal with GPUMD
            cooled_atoms = cool_with_gpumd(start_atoms, run_dir=run_dir)
            calc = CPUNEP(NEP_MODEL_FILE)
            cooled_atoms.calc = calc
            energy = cooled_atoms.get_potential_energy()
            print(f"    Cooling done. Energy = {energy:.6f} eV")
            if DEBUG:
                plot_temperature_trace(run_dir, label, i)

            # Attach metadata to atoms.info so downstream scripts can read it back
            if not no_gb:
                cooled_atoms.info["axis"]   = list(axis)
                cooled_atoms.info["sigma"]  = sigma
                cooled_atoms.info["plane"] = list(plane)
            cooled_atoms.info["run_index"]  = i
            cooled_atoms.info["energy_ev"]  = energy
            cooled_atoms.info["gb_label"]   = label

            # Write structure to per-run traj file
            write(os.path.join(run_dir, "structure.traj"), cooled_atoms)
            writer.writerow([i, energy])

            if DEBUG:
                # Save per-run visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_atoms(cooled_atoms, ax, rotation="10x,10y,0z")
                ax.set_title(f"{label} run {i} — E={energy:.4f} eV")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "relaxed.png"))
                plt.close()

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
            f"NEP model not found at '{NEP_MODEL_FILE}'. "
            "Check nep_model path in config."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fetch Si structure from Materials Project (requires API key in .env)
    print(f"Fetching Si structure from Materials Project (mp-149)...")
    s_input = Grain.from_mp_id("mp-149")

    if NO_GB_MODE:
        process_gb(None, -1, None, s_input)
    else:
        gb_status = check_gb_generation_status(args.config)
        for (axis, sigma, plane) in GB_LIST:
            label = gb_label(axis, sigma, plane)
            info = gb_status.get(label, {"status": "not_started", "runs_remaining": N_RUNS})
            start_run = N_RUNS - info["runs_remaining"]
            process_gb(axis, sigma, plane, s_input, start_run=start_run)

    print("\nAll structures processed.")


if __name__ == "__main__":
    main()
