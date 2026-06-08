"""
Grain boundary macroscopic geometry, built on top of the aimsgb library.

For a CSL grain boundary specified by (axis, sigma, plane) this returns:

    character          — "twist", "tilt", or "mixed"
    tilt_type          — "symmetric" / "asymmetric" for tilt GBs, else "" (empty)
    misorientation_deg — rotation (disorientation) angle about the tilt axis
    inclination_deg    — rotation of the boundary plane about the tilt axis away
                         from the nearest symmetric tilt plane (0 = symmetric);
                         NaN for twist / mixed GBs

What comes from aimsgb vs. what is computed here
------------------------------------------------
aimsgb's ``GBInformation.get_gb_info()[sigma]`` directly provides the
misorientation angle (``Theta``) and the CSL ``Rotation matrix``.  aimsgb
defines the character exactly as the geometry below (GB plane ∥ axis → twist,
⊥ axis → tilt; see the GrainBoundary docstring), but it has no API for the
symmetric/asymmetric distinction or the inclination angle, so those are derived
here from aimsgb's rotation matrix.

Geometry (cubic crystals, where a Miller plane (hkl) has normal [h,k,l]):
    twist : axis × plane == 0          (normal ∥ axis)
    tilt  : axis · plane == 0          (normal ⊥ axis)

Symmetric tilt: a boundary plane with normal n is a mirror plane of the
bicrystal iff reflecting grain A across it reproduces grain B up to a lattice
symmetry, i.e. M_n · R lies in the cubic point group O_h (48 signed permutation
matrices), where M_n is the reflection across the plane and R the CSL
misorientation.  The inclination is the smallest angle between the boundary
normal and any symmetric tilt plane normal of the same CSL.
"""

from functools import reduce
from itertools import product as _iproduct
from math import gcd

import numpy as np


# ---------------------------------------------------------------------------
# Cubic point group O_h: 48 signed permutation matrices (det = ±1)
# ---------------------------------------------------------------------------

def _build_cubic_point_group():
    ops = []
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for perm in perms:
        for signs in _iproduct([1, -1], repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for i in range(3):
                m[i, perm[i]] = signs[i]
            ops.append(m)
    return ops


_OH_OPS = _build_cubic_point_group()
assert len(_OH_OPS) == 48


def _is_cubic_op(M, tol=1e-6):
    return any(np.allclose(M, o, atol=tol) for o in _OH_OPS)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _reflection(normal):
    n = _unit(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _is_symmetric_plane(R, normal):
    M = _reflection(normal)
    return _is_cubic_op(M @ R) or _is_cubic_op(R @ M)


# ---------------------------------------------------------------------------
# aimsgb lookups, cached per (axis, sigma)
# ---------------------------------------------------------------------------

_INFO_CACHE = {}   # (axis, sigma) -> (rotation_matrix, misorientation_deg)
_SYM_CACHE = {}    # (axis, sigma) -> list of symmetric tilt plane normals (unit)


def _gb_info(axis, sigma):
    """Return (rotation_matrix, misorientation_deg) from aimsgb, or (None, nan)."""
    key = (tuple(int(x) for x in axis), int(sigma))
    if key in _INFO_CACHE:
        return _INFO_CACHE[key]

    from aimsgb import GBInformation

    try:
        data = GBInformation(list(key[0]), key[1], specific=True).get_gb_info()[key[1]]
        R = np.array(data["Rotation matrix"], dtype=float)
        # aimsgb may list a misorientation and its symmetry-equivalent
        # complement; the disorientation is the smaller angle.
        theta = float(min(data["Theta"]))
        result = (R, theta)
    except (ValueError, KeyError):
        result = (None, float("nan"))

    _INFO_CACHE[key] = result
    return result


def _symmetric_normals(axis, sigma, R, max_index=8):
    """All symmetric tilt plane normals (unit vectors) for this CSL, cached."""
    key = (tuple(int(x) for x in axis), int(sigma))
    if key in _SYM_CACHE:
        return _SYM_CACHE[key]

    axis = np.asarray(axis, dtype=float)
    normals = []
    rng = range(-max_index, max_index + 1)
    for v in _iproduct(rng, rng, rng):
        if v == (0, 0, 0):
            continue
        if reduce(gcd, [abs(x) for x in v]) != 1:   # primitive Miller indices only
            continue
        if abs(float(np.dot(axis, v))) > 1e-9:       # tilt planes are ⊥ axis
            continue
        if _is_symmetric_plane(R, v):
            normals.append(_unit(v))

    _SYM_CACHE[key] = normals
    return normals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gb_geometry(axis, sigma, plane, tol=1e-9, sym_tol=1e-3):
    """Macroscopic geometry of a CSL grain boundary.

    Parameters
    ----------
    axis, plane : length-3 Miller index sequences
    sigma : int  CSL sigma value
    sym_tol : float  inclination (deg) below which a tilt GB is "symmetric"

    Returns
    -------
    dict with keys:
        character          : "twist" | "tilt" | "mixed"
        tilt_type          : "symmetric" | "asymmetric" | ""
        misorientation_deg : float
        inclination_deg    : float (NaN for twist / mixed)
    """
    axis_arr = np.asarray(axis, dtype=float)
    plane_arr = np.asarray(plane, dtype=float)

    R, misorientation = _gb_info(axis, sigma)

    result = {
        "character": "mixed",
        "tilt_type": "",
        "misorientation_deg": misorientation,
        "inclination_deg": float("nan"),
    }

    if np.allclose(np.cross(axis_arr, plane_arr), 0.0, atol=tol):
        result["character"] = "twist"
        return result

    if abs(float(np.dot(axis_arr, plane_arr))) < tol:
        result["character"] = "tilt"
        if R is not None:
            syms = _symmetric_normals(axis, sigma, R)
            p = _unit(plane_arr)
            inclination = 180.0
            for ns in syms:
                cos = abs(float(np.clip(np.dot(p, ns), -1.0, 1.0)))
                inclination = min(inclination, float(np.degrees(np.arccos(cos))))
            if syms:
                result["inclination_deg"] = inclination
                result["tilt_type"] = (
                    "symmetric" if inclination < sym_tol else "asymmetric"
                )
        return result

    return result  # mixed
