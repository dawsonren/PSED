"""
Use aimsgb to generate grain boundaries, then use GPUMD to relax
such structures into realistic crystalline configurations.

Usage:
    python generate_gbs.py --config ../configs/small_box.yaml
    python generate_gbs.py --config ../configs/large_box.yaml

Pipeline:
1. Load GB specifications and run parameters from a unified YAML config.
2. Build GB structure with aimsgb (GrainBoundary + Grain.stack_grains).
   The x-length is controlled by uc_a + uc_b, y/z scale is controlled
   by SCALE_REPEAT parameter
3. Anneal with GPUMD: cooling ramp from t_start down to t_end over
   total_time_ps, using nvt_bdp thermostat.
4. Final BFGS relaxation (CPU NEP, BFGS) to remove residual forces.
5. Repeat steps 3-4 n_runs times with different random initial velocities.
   ALL relaxed structures are saved (not just the lowest-energy one) because
   each metastable structure is a genuine local minimum and valid ML training
   data — different runs produce distinct atomic arrangements at the interface
   consistent with the same crystallographic misorientation.
6. A summary.csv records energies per run.

File outputs:
    results/<config_name>/gb_generation/
      sigma{n}_{miller}_{axis}/
        run_0/          <- GPUMD working directory
          movie.xyz
          thermo.out
          velocity.out
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
UC_A           = gb_cfg["uc_a"]
UC_B           = gb_cfg["uc_b"]
N_RUNS         = gb_cfg["n_runs"]
SCALE_REPEAT   = int(gb_cfg["scale_repeat"])
T_START        = float(gb_cfg["t_start"])
T_END          = float(gb_cfg["t_end"])
TOTAL_TIME_PS  = float(gb_cfg["total_time_ps"])
TIMESTEP_FS    = float(gb_cfg["timestep_fs"])
RELAX_FMAX     = float(gb_cfg["relax_fmax"])
TAU_T         = float(gb_cfg["tau_t"])

# Derived
N_STEPS        = int(TOTAL_TIME_PS * 1000.0 / TIMESTEP_FS)
THERMO_INTERVAL = 5000    # write thermo every 5 ps — enough to verify T tracking
DUMP_INTERVAL   = 50000   # write positions every 50 ps — file size control

GB_LIST = [
    (entry["sigma"], tuple(entry["miller"]), tuple(entry["axis"]))
    for entry in config["grain_boundaries"]
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gb_label(sigma, miller, axis):
    """Produce a filesystem-safe label, e.g. sigma5_2-10_001"""
    m = "".join(str(x) for x in miller).replace("-", "-")
    a = "".join(str(x) for x in axis)
    return f"sigma{sigma}_{m}_{a}"


def build_gb_atoms(s_input, sigma, miller, axis):
    """
    Build a periodic bicrystal using aimsgb and return an ASE Atoms object.
    The GB plane is perpendicular to the x-axis (direction=0 by default in
    aimsgb), which aligns with the heat transport direction for RNEMD.

    The x-length is determined by uc_a + uc_b. No cell repetition is done
    here; Y/Z repetition for cross-section convergence happens in run_rnemd.py.
    """
    gb = GrainBoundary(axis, sigma, miller, s_input, uc_a=UC_A, uc_b=UC_B)
    structure = Grain.stack_grains(gb.grain_a, gb.grain_b, direction=gb.direction)

    atoms = structure.to_ase_atoms()
    atoms.pbc = True
    atoms.wrap()

    return atoms, gb


def cool_with_gpumd(atoms, run_dir):
    """
    Run a GPUMD nvt_bdp cooling ramp: T_START -> T_END over TOTAL_TIME_PS.

    nvt_bdp is the Bussi-Donadio-Parrinello stochastic velocity-rescaling
    thermostat. TAU_T (in timesteps) sets the coupling timescale (recommended
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

    md_parameters = [
        ("velocity",  float(T_START)),
        ("time_step", float(TIMESTEP_FS)),
        ("ensemble",  ["npt_scr", float(T_START), float(T_END), float(TAU_T)]),
        ("dump_thermo",   int(THERMO_INTERVAL)),
        ("dump_position", int(DUMP_INTERVAL)),
        ("run", int(N_STEPS)),
    ]

    cooled_atoms = calc.run_custom_md(
        md_parameters,
        return_last_atoms=True
    )
    cooled_atoms.pbc = atoms.pbc
    cooled_atoms.wrap()

    return cooled_atoms


def final_bfgs_relaxation(atoms, logfile="relax.log"):
    """
    Relax atomic positions at constant cell shape using BFGS + CPUNEP.
    """
    atoms.calc = CPUNEP(NEP_MODEL_FILE)

    optimizer = LBFGS(atoms, logfile=logfile)
    optimizer.run(fmax=RELAX_FMAX)

    energy = atoms.get_potential_energy()
    return atoms, energy


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
    axes[1].set_ylabel("T_actual − T_target [K]")
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
# Main loop
# ---------------------------------------------------------------------------

def process_gb(sigma, miller, axis, s_input):
    label   = gb_label(sigma, miller, axis)
    out_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    summary_file = os.path.join(out_dir, "summary.csv")

    print(f"\n{'='*60}")
    print(f"Processing: {label}  (config: {CONFIG_NAME})")
    print(f"  sigma={sigma}, miller={miller}, axis={axis}")
    print(f"  uc_a={UC_A}, uc_b={UC_B}, scale_repeat={SCALE_REPEAT}")
    print(f"  n_runs={N_RUNS}, T: {T_START}K -> {T_END}K over {TOTAL_TIME_PS}ps")
    print(f"{'='*60}")

    # Build initial GB structure and repeat along Y/Z for cross-section convergence.
    # This must happen before annealing so GPUMD sees a thick enough cell in all
    # periodic directions (NEP requires thickness >= 2 * cutoff = 10 Å).
    gb_atoms, gb = build_gb_atoms(s_input, sigma, miller, axis)
    gb_atoms = gb_atoms.repeat((SCALE_REPEAT, SCALE_REPEAT, 1))
    gb_atoms.wrap()
    print(f"  Built GB: {len(gb_atoms)} atoms after {SCALE_REPEAT}x Y/Z repeat "
          f"(cell: {gb_atoms.cell[0,0]:.1f} x {gb_atoms.cell[1,1]:.1f} x {gb_atoms.cell[2,2]:.1f} Å)")

    # Save initial structure for reference
    write(os.path.join(out_dir, "initial.traj"), gb_atoms)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_atoms(gb_atoms, ax)
    ax.set_title(f"{label} — initial")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "initial.png"))
    plt.close()

    # Open summary CSV
    with open(summary_file, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["run_index", "energy_ev"])

        for i in range(N_RUNS):
            print(f"\n  Run {i+1}/{N_RUNS}...")

            run_dir = os.path.join(out_dir, f"run_{i}")
            start_atoms = gb_atoms.copy()

            # Step 1: Anneal with GPUMD
            cooled_atoms = cool_with_gpumd(start_atoms, run_dir=run_dir)
            print(f"    Cooling done.")
            plot_temperature_trace(run_dir, label, i)

            # Step 2: Final BFGS relaxation at constant cell (CPU NEP)
            relax_log = os.path.join(run_dir, "relax.log")
            relaxed_atoms, energy = final_bfgs_relaxation(cooled_atoms, logfile=relax_log)
            print(f"    Relaxed. Energy = {energy:.6f} eV")

            # Attach metadata to atoms.info so downstream scripts can read it back
            relaxed_atoms.info["sigma"]      = sigma
            relaxed_atoms.info["miller"]     = list(miller)
            relaxed_atoms.info["axis"]       = list(axis)
            relaxed_atoms.info["run_index"]  = i
            relaxed_atoms.info["energy_ev"]  = energy
            relaxed_atoms.info["gb_label"]   = label

            # Write structure to per-run traj file
            write(os.path.join(run_dir, "structure.traj"), relaxed_atoms)
            writer.writerow([i, energy])

            # Save per-run visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_atoms(relaxed_atoms, ax)
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

    for (sigma, miller, axis) in GB_LIST:
        process_gb(sigma, miller, axis, s_input)

    print("\nAll grain boundaries processed.")


if __name__ == "__main__":
    main()
