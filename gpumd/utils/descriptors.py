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


# ---------------------------------------------------------------------------
# Coordination-defect descriptors
# ---------------------------------------------------------------------------

def coordination_descriptors(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    ideal_coord: int = 4,
    mask=None,
) -> dict:
    """
    Coordination-defect statistics over an atom subset.

    For tetrahedral Si the ideal coordination number is 4; under- and
    over-coordinated atoms mark broken/strained bonding at the grain boundary
    and turn out to be a strong predictor of the thermal boundary resistance.

    Parameters
    ----------
    atoms : ASE Atoms
    bond_cutoff : float
        Neighbor cutoff in Å (same first-shell cutoff used for the other
        descriptors; ~3.0 Å sits between the 1st and 2nd Si-Si shells).
    ideal_coord : int
        Reference coordination number (4 for diamond-cubic Si).
    mask : array-like of bool, optional
        Restrict the statistics to this atom subset (e.g. the GB slab).

    Returns
    -------
    dict with keys: coord_under, coord_over, coord_mean, coord_std
        coord_under / coord_over are the fractions of subset atoms with
        coordination below / above `ideal_coord`.
    """
    n = len(atoms)
    # 'i' returns the source index of every neighbor pair (full list), so a
    # bincount over it is exactly the per-atom coordination number. O(N).
    i_arr = neighbor_list('i', atoms, bond_cutoff)
    coord = np.bincount(i_arr, minlength=n)

    if mask is not None:
        coord = coord[np.asarray(mask, dtype=bool)]

    if len(coord) == 0:
        return {"coord_under": 0.0, "coord_over": 0.0,
                "coord_mean": 0.0, "coord_std": 0.0}

    return {
        "coord_under": float(np.mean(coord < ideal_coord)),
        "coord_over":  float(np.mean(coord > ideal_coord)),
        "coord_mean":  float(np.mean(coord)),
        "coord_std":   float(np.std(coord)),
    }


# ---------------------------------------------------------------------------
# Steinhardt bond-orientational order parameters
# ---------------------------------------------------------------------------

def _sph_harm(l, m, theta_polar, phi_azim):
    """Y_l^m on polar/azimuthal angle arrays, across scipy versions.

    scipy >=1.15 exposes ``sph_harm_y(n, m, theta, phi)`` with theta the polar
    (colatitude) angle and phi the azimuth; older scipy uses the removed
    ``sph_harm(m, n, theta_azim, phi_polar)`` with the argument order flipped.
    """
    from scipy import special
    if hasattr(special, "sph_harm_y"):
        return special.sph_harm_y(l, m, theta_polar, phi_azim)
    return special.sph_harm(m, l, phi_azim, theta_polar)  # legacy arg order


def _steinhardt_qlm(atoms: Atoms, bond_cutoff: float, ls):
    """Per-atom complex q_lm vectors, the shared core of q_l and w_l.

    Returns ``(qlm, valid)`` where ``qlm[l]`` is a complex array of shape
    (N, 2l+1) holding q_lm(i) = (1/N_b) Σ_j Y_l^m(r_ij) for each atom (rows for
    atoms with no neighbour are left zero), and ``valid`` is the length-N boolean
    mask of atoms that have at least one neighbour inside the cutoff.
    """
    n = len(atoms)
    i_arr, _, D_arr = neighbor_list("ijD", atoms, bond_cutoff)
    qlm = {l: np.zeros((n, 2 * l + 1), dtype=complex) for l in ls}
    if i_arr.size == 0:
        return qlm, np.zeros(n, dtype=bool)

    nbr_count = np.bincount(i_arr, minlength=n).astype(float)
    valid = nbr_count > 0

    r = np.linalg.norm(D_arr, axis=1)
    theta_polar = np.arccos(np.clip(D_arr[:, 2] / r, -1.0, 1.0))
    phi_azim = np.arctan2(D_arr[:, 1], D_arr[:, 0]) % (2.0 * np.pi)

    for l in ls:
        for col, m in enumerate(range(-l, l + 1)):
            np.add.at(qlm[l][:, col], i_arr, _sph_harm(l, m, theta_polar, phi_azim))
        qlm[l][valid] /= nbr_count[valid, None]
    return qlm, valid


def steinhardt_per_atom(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    ls=(4, 6),
) -> dict:
    """
    Per-atom local Steinhardt bond-orientational order parameters q_l.

    For each atom i with first-shell neighbours j (within ``bond_cutoff``),

        q_lm(i) = (1/N_b) Σ_j Y_l^m(r_ij),
        q_l(i)  = sqrt( 4π/(2l+1) · Σ_m |q_lm(i)|² ).

    Returns
    -------
    dict mapping each l in ``ls`` to a length-N float array of q_l, with NaN for
    atoms that have no neighbour inside the cutoff. Aggregating these over an
    atom subset (mask) and/or several slab widths is left to the caller; see
    ``steinhardt_descriptors`` for the standard mean/std reduction.
    """
    n = len(atoms)
    qlm, valid = _steinhardt_qlm(atoms, bond_cutoff, ls)
    per_atom = {l: np.full(n, np.nan) for l in ls}
    for l in ls:
        per_atom[l][valid] = np.sqrt(
            4.0 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm[l][valid]) ** 2, axis=1)
        )
    return per_atom


# Cache of nonzero Wigner-3j coefficients (l l l; m1 m2 m3), m1+m2+m3=0, keyed by l.
_W3J_CACHE: dict = {}


def _wigner3j_table(l):
    """List of (i1, i2, i3, coeff) with m-indices shifted to 0..2l, cached per l."""
    if l not in _W3J_CACHE:
        from sympy.physics.wigner import wigner_3j
        tab = []
        for m1 in range(-l, l + 1):
            for m2 in range(-l, l + 1):
                m3 = -(m1 + m2)
                if -l <= m3 <= l:
                    c = float(wigner_3j(l, l, l, m1, m2, m3))
                    if c != 0.0:
                        tab.append((m1 + l, m2 + l, m3 + l, c))
        _W3J_CACHE[l] = tab
    return _W3J_CACHE[l]


def steinhardt_w_descriptors(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    ls=(4, 6),
    mask=None,
) -> dict:
    """
    Normalised third-order Steinhardt invariants ŵ_l, averaged over an atom
    subset.

        w_l(i)  = Σ_{m1+m2+m3=0} (l l l; m1 m2 m3) q_lm1 q_lm2 q_lm3,
        ŵ_l(i)  = Re w_l(i) / ( Σ_m |q_lm(i)|² )^{3/2}.

    ŵ_l is a dimensionless, rotation-invariant fingerprint of the local bonding
    *shape* (perfect diamond-cubic Si: ŵ_4 ≈ -0.159, ŵ_6 ≈ +0.013). It carries
    similar information to q_l for this system; see ml_pipeline.ipynb section 11c.
    Requires ``sympy`` for the Wigner-3j symbols.

    Returns
    -------
    dict with keys ``w{l}_mean`` and ``w{l}_std`` for each l in ``ls``.
    """
    n = len(atoms)
    qlm, valid = _steinhardt_qlm(atoms, bond_cutoff, ls)
    sub = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    out = {}
    for l in ls:
        w = np.zeros(n, dtype=complex)
        q = qlm[l]
        for i1, i2, i3, c in _wigner3j_table(l):
            w += c * q[:, i1] * q[:, i2] * q[:, i3]
        s = np.sum(np.abs(q) ** 2, axis=1)
        good = valid & (s > 1e-12)
        what = np.full(n, np.nan)
        what[good] = np.real(w[good]) / s[good] ** 1.5

        w_sub = what[sub]
        w_sub = w_sub[np.isfinite(w_sub)]
        if w_sub.size:
            out[f"w{l}_mean"] = float(np.mean(w_sub))
            out[f"w{l}_std"] = float(np.std(w_sub))
        else:
            out[f"w{l}_mean"] = 0.0
            out[f"w{l}_std"] = 0.0
    return out


def steinhardt_descriptors(
    atoms: Atoms,
    bond_cutoff: float = 3.0,
    ls=(4, 6),
    mask=None,
) -> dict:
    """
    Local Steinhardt bond-orientational order parameters q_l, averaged over an
    atom subset.

    Perfect diamond-cubic (tetrahedral) Si gives q_4 ≈ 0.509, q_6 ≈ 0.629; the
    disordered/strained bonding at a grain boundary pulls these away from the
    ideal values, so the mean and spread of q_4, q_6 over the GB slab are a
    sharper local-order signal than the under/over-coordination counts.

    Parameters
    ----------
    atoms : ASE Atoms
    bond_cutoff : float
        First-shell neighbour cutoff in Å (same 3.0 Å as the other descriptors).
    ls : iterable of int
        Steinhardt degrees to compute (default q_4 and q_6).
    mask : array-like of bool, optional
        Restrict the mean/std to this atom subset (e.g. the GB slab). q_l is
        still built from *every* neighbour of each masked atom.

    Returns
    -------
    dict with keys ``q{l}_mean`` and ``q{l}_std`` for each l in ``ls``.
    """
    n = len(atoms)
    per_atom = steinhardt_per_atom(atoms, bond_cutoff=bond_cutoff, ls=ls)
    sub = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    out = {}
    for l in ls:
        q_sub = per_atom[l][sub]
        q_sub = q_sub[np.isfinite(q_sub)]
        if q_sub.size:
            out[f"q{l}_mean"] = float(np.mean(q_sub))
            out[f"q{l}_std"] = float(np.std(q_sub))
        else:
            out[f"q{l}_mean"] = 0.0
            out[f"q{l}_std"] = 0.0
    return out
