"""
Grain boundary energy calculation.

GB energy (γ_GB) is the excess energy per unit area due to the presence of the
grain boundary, relative to bulk:

    γ_GB = (E_GB − N_GB × e_bulk) / (2 × A)   [eV/Å²]

where:
    E_GB    — total potential energy of the relaxed GB supercell [eV]
    N_GB    — number of atoms in the GB supercell
    e_bulk  — energy per atom in bulk Si [eV/atom]
    A       — GB plane area = Lx × Ly [Å²]  (stacking is along z)
    /2      — the periodic supercell contains two GB planes

To convert to SI units:  1 eV/Å² = 16.0218 J/m²
"""

import csv
from pathlib import Path

import numpy as np
from ase.io import read


J_PER_M2_PER_EV_PER_ANG2 = 16.0218  # conversion factor


def bulk_energy_per_atom(bulk_results_dir: str | Path) -> tuple[float, float]:
    """
    Compute the mean bulk energy per atom from completed bulk_si runs.

    Reads all run_*/structure.traj files from the bulk_si output directory and
    returns (mean_e_bulk, std_e_bulk) in eV/atom.

    Parameters
    ----------
    bulk_results_dir : path to results/<config>/gb_generation/bulk_si/

    Returns
    -------
    mean_e_bulk : float   mean energy per atom [eV/atom]
    std_e_bulk  : float   std dev across runs [eV/atom]
    """
    bulk_dir = Path(bulk_results_dir)
    run_dirs = sorted(bulk_dir.glob("run_*/structure.traj"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_*/structure.traj found in {bulk_dir}")

    e_per_atom = []
    for traj in run_dirs:
        atoms = read(str(traj))
        e_per_atom.append(atoms.info["energy_ev"] / len(atoms))

    arr = np.array(e_per_atom)
    return float(arr.mean()), float(arr.std())


def compute_gb_energy(
    gb_traj: str | Path,
    e_bulk_per_atom: float,
    gb_axis: int = 2,
) -> dict:
    """
    Compute grain boundary energy for a single relaxed GB structure.

    Parameters
    ----------
    gb_traj : path to a structure.traj file produced by generate_gbs.py
    e_bulk_per_atom : float
        Bulk energy per atom [eV/atom] from bulk_energy_per_atom().
    gb_axis : int
        Cell axis perpendicular to the GB plane (default 2 = z, matching
        aimsgb direction=0 convention used in generate_gbs.py).

    Returns
    -------
    dict with keys:
        gamma_ev_ang2  — GB energy in eV/Å²
        gamma_j_m2     — GB energy in J/m²
        E_GB           — total potential energy of the GB cell [eV]
        N_GB           — number of atoms
        e_bulk         — bulk reference energy used [eV/atom]
        area_ang2      — GB plane area [Å²]
        label          — gb_label from atoms.info (if present)
    """
    atoms = read(str(gb_traj))
    E_GB = atoms.info["energy_ev"]
    N_GB = len(atoms)

    cell = atoms.cell[:]
    # GB plane area: product of the two cell lengths not along gb_axis
    in_plane = [i for i in range(3) if i != gb_axis]
    a1 = np.linalg.norm(cell[in_plane[0]])
    a2 = np.linalg.norm(cell[in_plane[1]])
    A = a1 * a2  # valid for orthogonal cells (aimsgb output cleaned in generate_gbs.py)

    gamma_ev = (E_GB - N_GB * e_bulk_per_atom) / (2.0 * A)
    gamma_jm2 = gamma_ev * J_PER_M2_PER_EV_PER_ANG2

    return {
        "gamma_ev_ang2": gamma_ev,
        "gamma_j_m2":    gamma_jm2,
        "E_GB":          E_GB,
        "N_GB":          N_GB,
        "e_bulk":        e_bulk_per_atom,
        "area_ang2":     A,
        "label":         atoms.info.get("gb_label", ""),
    }


def compute_gb_energies_from_config(config_path: str | Path) -> list[dict]:
    """
    Compute GB energies for all completed runs under a given config.

    Requires a completed bulk_si run in the same results directory to serve as
    the bulk reference.  Each run produces one result dict; runs from the same
    GB label can be averaged by the caller.

    Parameters
    ----------
    config_path : path to the YAML config used during generation

    Returns
    -------
    list of result dicts (one per completed run), each with all keys from
    compute_gb_energy() plus 'run_index'.
    """
    import yaml

    config_path = Path(config_path).resolve()
    gpumd_root = config_path.parent.parent
    config_name = config_path.stem

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = gpumd_root / "results" / config_name / "gb_generation"
    bulk_dir    = results_dir / "bulk_si"

    e_bulk, e_bulk_std = bulk_energy_per_atom(bulk_dir)
    print(f"Bulk reference: {e_bulk:.6f} ± {e_bulk_std:.6f} eV/atom")

    from utils.work_coordination import gb_label

    results = []
    for entry in config["grain_boundaries"]:
        if entry["sigma"] == -1:
            continue
        label = gb_label(tuple(entry["axis"]), entry["sigma"], tuple(entry["plane"]))
        gb_dir = results_dir / label

        for traj in sorted(gb_dir.glob("run_*/structure.traj")):
            run_idx = int(traj.parent.name.split("_")[1])
            r = compute_gb_energy(traj, e_bulk)
            r["run_index"] = run_idx
            r["label"]     = label
            results.append(r)
            print(f"  {label} run {run_idx}: "
                  f"γ = {r['gamma_j_m2']:.3f} J/m²  ({r['gamma_ev_ang2']:.4f} eV/Å²)")

    return results
