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
from ase.neighborlist import neighbor_list
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
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.geometry.rdf import get_rdf


def compute_global_descriptors(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    rdf_cutoff: float = 10.0,
    rdf_nbins: int = 200,
    dihedral_nbins: int = 36,
    mask=None,
) -> dict:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        sub_atoms = atoms[mask]
    else:
        sub_atoms = atoms
        mask = np.ones(len(atoms), dtype=bool)

    # O(N) neighbor list via cell-decomposition.
    # D_arr[n] = minimum-image vector FROM i_arr[n] TO j_arr[n].
    # Full list: every pair (a,b) appears in both directions.
    i_arr, j_arr, d_arr, D_arr = neighbor_list(
        'ijdD', sub_atoms, bond_cutoff
    )

    # σ_l — keep only i < j to count each bond once
    bond_lengths = d_arr[i_arr < j_arr]
    sigma_l = float(np.std(bond_lengths)) if len(bond_lengths) > 1 else 0.0

    # Per-atom neighbor map: neighbors_of[a][b] = vector FROM a TO b
    neighbors_of = _build_neighbor_map(len(sub_atoms), i_arr, j_arr, D_arr)

    # σ_θ
    angles = _compute_all_angles(neighbors_of)
    sigma_theta = float(np.std(angles)) if len(angles) > 1 else 0.0

    # H_φ
    dihedrals = _compute_all_dihedrals(neighbors_of)
    if len(dihedrals) > 0:
        counts, _ = np.histogram(
            np.abs(dihedrals), bins=dihedral_nbins, range=(0.0, 180.0)
        )
        probs = counts / max(counts.sum(), 1)
        probs = probs[probs > 0]
        H_phi = float(-np.sum(probs * np.log(probs)))
    else:
        H_phi = 0.0

    # A_RDF
    cell_heights = _perpendicular_cell_heights(atoms)
    effective_rdf_cutoff = min(rdf_cutoff, min(cell_heights) / 2.0 - 1e-3)
    g_r, r = get_rdf(sub_atoms, effective_rdf_cutoff, rdf_nbins)
    # ensure that r is sorted
    sorted_indices = np.argsort(r)
    g_r = g_r[sorted_indices]
    r = r[sorted_indices]
    A_RDF = float(np.trapezoid(np.abs(g_r - 1.0), r))

    # ρ
    rho = float(len(sub_atoms)) / float(atoms.get_volume())

    return {
        "sigma_theta": sigma_theta,
        "sigma_l": sigma_l,
        "H_phi": H_phi,
        "A_RDF": A_RDF,
        "rho": rho,
        "n_atoms_used": int(len(sub_atoms)),
    }


def _build_neighbor_map(n_atoms, i_arr, j_arr, D_arr):
    """Per-atom neighbor dict from neighbor_list() output.

    Convention: neighbors_of[a][b] = vector FROM a TO b.

    Since neighbor_list() returns a full list (both directions of every
    pair), every direction is set directly—no manual reversal needed.
    """
    neighbors_of = [dict() for _ in range(n_atoms)]
    for idx in range(len(i_arr)):
        neighbors_of[int(i_arr[idx])][int(j_arr[idx])] = D_arr[idx]
    return neighbors_of


def _compute_all_angles(neighbors_of):
    """All unique i-j-k bond angles in degrees.

    For central atom j, each unordered pair of neighbors {i, k} is
    visited once (ii < kk in the inner loop).  The angle is between
    vectors j→i and j→k, read directly from the map.
    """
    all_angles = []
    for j, neigh_j in enumerate(neighbors_of):
        nbrs = list(neigh_j.values())
        n = len(nbrs)
        for ii in range(n):
            for kk in range(ii + 1, n):
                v1, v2 = nbrs[ii], nbrs[kk]
                cos_a = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30
                )
                all_angles.append(
                    np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                )
    return np.array(all_angles) if all_angles else np.array([])


def _compute_all_dihedrals(neighbors_of):
    """All unique i-j-k-l proper dihedral angles in degrees.

    Each bond j-k is visited once (j < k).  For that bond we enumerate
    every neighbor i of j (i ≠ k) and every neighbor l of k (l ≠ j,
    l ≠ i).  The three consecutive bond vectors are:

        b1 = i→j = -(j→i)  =  -neighbors_of[j][i]
        b2 = j→k            =   neighbors_of[j][k]
        b3 = k→l            =   neighbors_of[k][l]

    b1 is the only one that requires negation: we have the vector j→i
    stored in the map but need i→j, so we flip the sign.
    """
    all_dihedrals = []
    for j, neigh_j in enumerate(neighbors_of):
        for k, vec_jk in neigh_j.items():
            if k <= j:
                continue

            i_atoms = [i for i in neigh_j if i != k]
            l_atoms = [l for l in neighbors_of[k] if l != j]
            b2 = vec_jk

            for i in i_atoms:
                b1 = -neigh_j[i]
                for l in l_atoms:
                    if l == i:
                        continue
                    b3 = neighbors_of[k][l]

                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    nn1 = np.linalg.norm(n1)
                    nn2 = np.linalg.norm(n2)
                    if nn1 < 1e-10 or nn2 < 1e-10:
                        continue
                    n1 /= nn1
                    n2 /= nn2
                    b2_hat = b2 / np.linalg.norm(b2)
                    m1 = np.cross(n1, b2_hat)
                    angle = np.degrees(
                        np.arctan2(np.dot(m1, n2), np.dot(n1, n2))
                    )
                    all_dihedrals.append(angle)

    return np.array(all_dihedrals) if all_dihedrals else np.array([])


def _perpendicular_cell_heights(atoms: Atoms) -> np.ndarray:
    """Three perpendicular heights of the unit cell (Å)."""
    cell = atoms.get_cell()
    V = abs(atoms.get_volume())
    heights = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        heights[i] = V / np.linalg.norm(np.cross(cell[j], cell[k]))
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
