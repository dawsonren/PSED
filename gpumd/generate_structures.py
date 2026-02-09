"""
Use aimsgb to generate grain boundaries, then use 
GPUMD to relax such structures into realistic configurations.

Pipeline:
1. Build GB structure with aimsgb.
2. Convert to ASE Atoms and (implicitly) write to GPUMD format via GPUNEP.
3. Use Si NEP potential (Si_GAP_nep.txt) with a cooling ramp
   4000 K -> 25 K over 1275 ps, dt = 1 fs using GPUMD (nvt_bdp).
4. After cooling, perform a final energy relaxation with L-BFGS (via
   calorine.tools.relax_structure, minimizer='lbfgs').
5. Repeat steps 3-4 ten times with different initial velocities, pick the
   lowest-energy final structure.
6. Save the best relaxed structure to file in ASE format for later RNEMD.
"""

import os
from dotenv import load_dotenv

from aimsgb import GrainBoundary, Grain
from ase.io import write

from calorine.calculators import GPUNEP, CPUNEP
from calorine.tools import relax_structure

import matplotlib.pyplot as plt

# If you want visualization during debugging:
from ase.visualize.plot import plot_atoms

load_dotenv()

# ----------------------------------------------------------------------
# User-configurable parameters
# ----------------------------------------------------------------------

MP_ID = "mp-149"          # Silicon
UC_A = 2
UC_B = 2

# NEP potential assumed in current working directory
NEP_MODEL_FILE = "Si_GAP_nep.txt"

# Cooling schedule
T_START = 4000.0          # K
T_END = 25.0              # K
TOTAL_TIME_PS = 1275.0    # ps
TIMESTEP_FS = 1.0         # fs

# Number of independent MD+relax runs
N_RUNS = 10

# GPUMD / GPUNEP parameters
THERMO_INTERVAL = 5000    # how often to write thermo output (steps)
DUMP_INTERVAL = 50000     # how often to write positions (steps)
T_COUP = 1000.0           # thermostat coupling parameter for nvt_bdp (in timesteps)

# Final relaxation parameters
RELAX_FMAX = 0.01         # eV/Å
RELAX_STEPS = 2000
RELAX_CONSTANT_CELL = True  # keep GB cell fixed during final relaxation


# ----------------------------------------------------------------------
# Grain boundary specification
# ----------------------------------------------------------------------
# In Lortaraprasert and Shiomi 2022, they use:
# 1. Sigma 5 (2 -1 0) [0 0 1]
# 2. Sigma 9 (-1 -2 1) [1 1 0]
# 3. Sigma 9 (2 -2 1) [2 1 2]
# 4. Sigma 3 (1 -1 2) [1 1 0]
# 5. Sigma 5 (3 1 0) [3 1 0]
# 6. Sigma 13 (1 0 0) [1 0 0]
GB_LIST = [
    (5, (2, -1, 0), (0, 0, 1)),
    # (9, (-2, 2, 1), (1, 1, 0)),
    # (9, (-2, 2, 1), (2, 1, 2)),
    # (3, (1, -1, 2), (1, 1, 0)),
    # (5, (3, 1, 0), (3, 1, 0)),
    # (13, (1, 0, 0), (1, 0, 0))
]


def build_gb_atoms(s_input, sigma, miller, axis, uc_a=UC_A, uc_b=UC_B):
    """Construct a grain boundary and return an ASE Atoms object."""
    gb = GrainBoundary(axis, sigma, miller, s_input, uc_a=uc_a, uc_b=uc_b)
    structure = Grain.stack_grains(gb.grain_a, gb.grain_b, direction=gb.direction)

    atoms = structure.to_ase_atoms()
    atoms.pbc = True
    atoms.wrap()

    # Optional: quick visual sanity check
    fig, ax = plt.subplots()
    plot_atoms(atoms, ax)
    plt.savefig("initial_gb.png")

    return atoms, gb


def cool_with_gpumd(atoms, sigma, run_index, nep_model=NEP_MODEL_FILE):
    """
    Run a GPUMD MD simulation via GPUNEP with a cooling ramp:
    4000 K -> 25 K over TOTAL_TIME_PS with timestep TIMESTEP_FS.

    Uses the nvt_bdp (Bussi-Donadio-Parrinello) thermostat ensemble in GPUMD.
    Returns the last snapshot as an ASE Atoms object.
    """
    # Directory for this specific MD run
    run_dir = f"gpumd_sigma{sigma}_run{run_index}"
    os.makedirs(run_dir, exist_ok=True)

    # Attach GPU NEP calculator that drives GPUMD
    calc = GPUNEP(nep_model, atoms=atoms, directory=run_dir)
    atoms.calc = calc

    # Total number of MD steps
    n_steps = int(TOTAL_TIME_PS * 1000.0 / TIMESTEP_FS)

    # Parameters for run.in (potential keyword is handled automatically by GPUNEP)
    # We use:
    #   - velocity 4000  (initial velocities at 4000 K)
    #   - ensemble nvt_bdp 4000 25 T_COUP  (Bussi thermostat with ramp)
    #   - run n_steps
    #   - time_step 1     (fs)
    parameters = [
        ("dump_thermo", THERMO_INTERVAL),
        ("dump_position", DUMP_INTERVAL),
        ("time_step", TIMESTEP_FS),
        ("velocity", T_START),
        ("ensemble", ["nvt_bdp", T_START, T_END, T_COUP]),
        ("run", n_steps),
    ]

    # This writes xyz.in/run.in etc. in GPUMD format and runs gpumd.
    # If return_last_atoms=True, calorine parses the last snapshot back into ASE.
    final_atoms = calc.run_custom_md(parameters, return_last_atoms=True)

    # Ensure PBC and wrapping are preserved on the returned structure
    final_atoms.pbc = True
    final_atoms.wrap()

    return final_atoms


def final_lbfgs_relaxation(atoms, nep_model=NEP_MODEL_FILE):
    """
    Take a cooled structure, attach CPUNEP, and relax with L-BFGS using
    calorine.tools.relax_structure(minimizer='lbfgs').
    """
    atoms = atoms.copy()
    atoms.calc = CPUNEP(nep_model)

    relax_structure(
        atoms,
        fmax=RELAX_FMAX,
        steps=RELAX_STEPS,
        minimizer="lbfgs",
        constant_cell=RELAX_CONSTANT_CELL,
        # you can pass e.g. logfile="relax.log" here if you want
    )

    energy = atoms.get_potential_energy()
    return atoms, energy


def anneal_and_relax_many_times(gb_atoms, sigma, n_runs=N_RUNS):
    """
    For a given GB Atoms object, run:
      - GPUMD cooling (4000K -> 25K) via GPUNEP
      - final LBFGS relaxation via CPUNEP
    n_runs times, each with different initial velocities (handled internally
    by GPUMD), and return the lowest-energy final structure.
    """
    best_atoms = None
    best_energy = None
    energies = []

    for i in range(n_runs):
        print(f"[sigma={sigma}] Starting MD+relax run {i + 1}/{n_runs}")

        # Start each run from the same initial GB structure
        start_atoms = gb_atoms.copy()

        # Cooling MD with GPUMD
        cooled_atoms = cool_with_gpumd(start_atoms, sigma=sigma, run_index=i)

        # Final LBFGS relaxation (CPU NEP, no GPU needed here)
        relaxed_atoms, energy = final_lbfgs_relaxation(cooled_atoms)

        print(f"[sigma={sigma}] Run {i + 1}: final energy = {energy:.6f} eV")

        energies.append(energy)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_atoms = relaxed_atoms.copy()

    print(f"[sigma={sigma}] Best energy over {n_runs} runs: {best_energy:.6f} eV")
    return best_atoms, best_energy, energies


def main():
    # Sanity check: NEP model must exist
    if not os.path.exists(NEP_MODEL_FILE):
        raise FileNotFoundError(
            f"NEP model file '{NEP_MODEL_FILE}' not found in current directory. "
            "Place your Si NEP potential here or change NEP_MODEL_FILE."
        )

    # Build starting grain(s)
    s_input = Grain.from_mp_id(MP_ID)

    for (sigma, miller, axis) in GB_LIST:
        print(f"Building grain boundary: sigma={sigma}, miller={miller}, axis={axis}")
        gb_atoms, gb = build_gb_atoms(s_input, sigma, miller, axis)

        # Save the initial GB structure in GPUMD format for reference
        # (xyz.in-style input). ASE has a built-in gpumd writer. :contentReference[oaicite:3]{index=3}
        initial_xyz_file = f"gb_sigma{sigma}_initial.xyz"
        write(initial_xyz_file, gb_atoms, format="gpumd")
        print(f"Initial GB structure written to {initial_xyz_file} (GPUMD format).")

        # Also optional: save as an ASE trajectory snapshot
        write(f"gb_sigma{sigma}_initial.traj", gb_atoms)
        print(f"Initial GB structure written to gb_sigma{sigma}_initial.traj (ASE).")

        # Run multiple cool+relax cycles and pick the best
        best_atoms, best_energy, energies = anneal_and_relax_many_times(
            gb_atoms, sigma=sigma, n_runs=N_RUNS
        )

        # Save the most stable final structure for later RNEMD
        best_traj_file = f"gb_sigma{sigma}_best_relaxed.traj"
        write(best_traj_file, best_atoms)
        print(
            f"Saved most stable relaxed structure for sigma={sigma} "
            f"to {best_traj_file} (E = {best_energy:.6f} eV)"
        )

        fig, ax = plt.subplots()
        plot_atoms(best_atoms, ax)
        plt.savefig(f"gb_sigma{sigma}_best_relaxed.png")

        # If you want, you could also save a CIF or XYZ of the best structure:
        # write(f"gb_sigma{sigma}_best_relaxed.cif", best_atoms)
        # write(f"gb_sigma{sigma}_best_relaxed.xyz", best_atoms)


if __name__ == "__main__":
    main()