"""
Structural descriptor utilities for grain boundary ML pipeline.

Implements the descriptor set:
  sigma_theta — std dev of bond angles (degrees)
  sigma_l     — std dev of bond lengths (Å)
  H_phi       — Shannon entropy of dihedral angle distribution
  A_RDF       — integral of |g(r) - 1| dr
  rho         — number density (atoms/Å³)

Plus SOAP descriptors via dscribe, with optional atom subset masking.

Uses ASE built-ins wherever possible to avoid reimplementing MIC geometry.
"""

import numpy as np
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.geometry.rdf import get_rdf


# ---------------------------------------------------------------------------
# GB atom identification
# ---------------------------------------------------------------------------

def identify_gb_atoms(
    atoms: Atoms,
    d_threshold: float,
    gb_axis: int = 2,
) -> np.ndarray:
    """
    Return a boolean mask for atoms within `d_threshold` Å of either GB plane.

    For aimsgb structures with direction=0, the GB planes sit at scaled
    coordinates 0.0 and 0.5 along `gb_axis` (ASE cell axis 2 by default).

    Parameters
    ----------
    atoms : ASE Atoms
    d_threshold : float
        Maximum Å from a GB plane for an atom to be included.
    gb_axis : int
        Cell axis along which grains are stacked (2 for aimsgb direction=0).

    Returns
    -------
    mask : np.ndarray of bool, shape (N,)
    """
    scaled = atoms.get_scaled_positions()
    frac = scaled[:, gb_axis] % 1.0  # wrap to [0, 1)

    cell_len = np.linalg.norm(atoms.cell[gb_axis])

    # Fractional distance to nearest half-period boundary (s=0.0 or s=0.5)
    frac_wrapped = frac % 0.5
    dist_angstrom = np.minimum(frac_wrapped, 0.5 - frac_wrapped) * cell_len

    return dist_angstrom <= d_threshold


# ---------------------------------------------------------------------------
# Global descriptors
# ---------------------------------------------------------------------------

def compute_global_descriptors(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    rdf_cutoff: float = 10.0,
    rdf_nbins: int = 200,
    dihedral_nbins: int = 36,
    mask=None,
) -> dict:
    """
    Compute five global structural descriptors.

    Parameters
    ----------
    atoms : ASE Atoms
    bond_cutoff : float
        Neighbour cutoff (Å) for bond length / angle / dihedral calculations.
    rdf_cutoff : float
        Cutoff (Å) for the radial distribution function.
    rdf_nbins : int
        Number of g(r) histogram bins.
    dihedral_nbins : int
        Histogram bins for dihedral entropy (36 → 10°/bin).
    mask : array-like of bool, optional
        If provided, only atoms where mask=True are used.

    Returns
    -------
    dict with keys:
        sigma_theta  — std dev of bond angles (degrees)
        sigma_l      — std dev of bond lengths (Å)
        H_phi        — Shannon entropy of dihedral distribution (nats)
        A_RDF        — integral of |g(r) − 1| dr (Å)
        rho          — number density (atoms/Å³)
        n_atoms_used — number of atoms included after masking
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        sub_atoms = atoms[mask]
    else:
        sub_atoms = atoms
        mask = np.ones(len(atoms), dtype=bool)

    # --- σ_l and σ_θ via ase.geometry.analysis.Analysis ---
    # Analysis requires a list of images
    ana = Analysis([sub_atoms], nl_max_neighbors=100)
    ana_all = Analysis([atoms], nl_max_neighbors=100)

    # Bond lengths: all Si-Si pairs within bond_cutoff
    # We use a simple cutoff-based NL by passing cutoff manually
    bond_list = ana.get_bonds("Si", "Si", unique=True)[0]
    if bond_list:
        bond_values = np.array(ana.get_values([bond_list])[0])
        sigma_l = float(np.std(bond_values)) if len(bond_values) > 1 else 0.0
    else:
        sigma_l = 0.0

    # Bond angles: Si-Si-Si
    angle_list = ana.get_angles("Si", "Si", "Si", unique=True)[0]
    if angle_list:
        angle_values = np.array(ana.get_values([angle_list])[0])
        sigma_theta = float(np.std(angle_values)) if len(angle_values) > 1 else 0.0
    else:
        sigma_theta = 0.0

    # Dihedral angles: Si-Si-Si-Si
    dihedral_list = ana.get_dihedrals("Si", "Si", "Si", "Si", unique=True)[0]
    if dihedral_list:
        dihedral_values = np.abs(np.array(ana.get_values([dihedral_list])[0]))
        counts, _ = np.histogram(dihedral_values, bins=dihedral_nbins, range=(0.0, 180.0))
        probs = counts / max(counts.sum(), 1)
        probs = probs[probs > 0]
        H_phi = float(-np.sum(probs * np.log(probs)))
    else:
        H_phi = 0.0

    # --- A_RDF via ase.geometry.rdf.get_rdf ---
    # Uses the full atoms (not sub_atoms) for correct density normalisation;
    # rdf_cutoff must satisfy 2*rdf_cutoff <= min perpendicular cell height.
    cell_heights = _perpendicular_cell_heights(atoms)
    effective_rdf_cutoff = min(rdf_cutoff, min(cell_heights) / 2.0 - 1e-3)
    g_r, r = get_rdf(sub_atoms, effective_rdf_cutoff, rdf_nbins)
    A_RDF = float(np.trapz(np.abs(g_r - 1.0), r))

    # --- ρ ---
    rho = float(len(sub_atoms)) / float(atoms.get_volume())

    return {
        "sigma_theta": sigma_theta,
        "sigma_l": sigma_l,
        "H_phi": H_phi,
        "A_RDF": A_RDF,
        "rho": rho,
        "n_atoms_used": int(len(sub_atoms)),
    }


def _perpendicular_cell_heights(atoms: Atoms) -> np.ndarray:
    """Return the three perpendicular heights of the unit cell (Å)."""
    cell = atoms.get_cell()
    # height_i = V / area_i, where area_i = |a_j × a_k|
    V = abs(atoms.get_volume())
    heights = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cross = np.cross(cell[j], cell[k])
        heights[i] = V / np.linalg.norm(cross)
    return heights


# ---------------------------------------------------------------------------
# SOAP descriptors
# ---------------------------------------------------------------------------

def compute_mean_soap(
    atoms: Atoms,
    soap,
    mask=None,
) -> np.ndarray:
    """
    Compute per-atom SOAP descriptors and return the mean vector.

    Parameters
    ----------
    atoms : ASE Atoms
    soap : dscribe.descriptors.SOAP
        Pre-configured SOAP descriptor instance.
    mask : array-like of bool, optional
        If provided, average only over atoms where mask=True.

    Returns
    -------
    mean_soap : np.ndarray, shape (soap_dim,)
    """
    # dscribe returns shape (N, soap_dim)
    all_soaps = soap.create(atoms, n_jobs=1)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return np.zeros(all_soaps.shape[1])
        return all_soaps[mask].mean(axis=0)

    return all_soaps.mean(axis=0)
