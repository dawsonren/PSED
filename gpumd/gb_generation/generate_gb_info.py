"""
Enumerate unique (axis, sigma, plane) triples for CSL grain boundaries
in cubic systems, suitable for feeding into aimsgb.

Usage:
python generate_gb_info.py --max_sigma 50 --max_axis_index 5 --config ../configs/small_box.yaml

Strategy:
1. Use rotation axes in the fundamental zone (h >= k >= l >= 0, gcd=1)
2. Query GBInformation for each axis up to max_sigma
3. Deduplicate misorientations using the 24 cubic symmetry operations

The 24 proper rotations of the cubic group Oh include:
- 1 identity
- 6 face rotations (90°, 270° about <100>)
- 3 face rotations (180° about <100>)
- 8 vertex rotations (120°, 240° about <111>)
- 6 edge rotations (180° about <110>)
"""

import numpy as np
from math import gcd
from functools import reduce
from itertools import product as iterproduct

from aimsgb import GBInformation

# ============================================================
# 24 proper cubic symmetry operations (rotation matrices)
# ============================================================
def _build_cubic_symmetry_ops():
    """Build the 24 proper rotation matrices of the cubic group."""
    ops = []
    # Generate from 90° rotations about [100], [010], [001]
    # and 120° rotations about [111]
    # Easier: just enumerate all signed permutation matrices with det=+1
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in iterproduct([1, -1], repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for i in range(3):
                mat[i, perm[i]] = signs[i]
            if np.linalg.det(mat) > 0:
                ops.append(mat)
    return ops


CUBIC_OPS = _build_cubic_symmetry_ops()
assert len(CUBIC_OPS) == 24, f"Expected 24 ops, got {len(CUBIC_OPS)}"


# ============================================================
# Axis-angle utilities
# ============================================================
def rotation_matrix_to_axis_angle(R):
    """Extract rotation angle and axis from a 3x3 rotation matrix."""
    R = np.array(R, dtype=float)
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(angle) < 1e-10:
        return np.array([1, 0, 0]), 0.0
    if abs(angle - np.pi) < 1e-10:
        # 180° rotation: axis from eigenvector with eigenvalue +1
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        # Canonical sign: first nonzero component positive
        for i in range(3):
            if abs(axis[i]) > 1e-10:
                if axis[i] < 0:
                    axis = -axis
                break
        return axis, angle
    # General case
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / np.linalg.norm(axis)
    # Canonical sign
    for i in range(3):
        if abs(axis[i]) > 1e-10:
            if axis[i] < 0:
                axis = -axis
                # angle = 2*pi - angle for proper convention
                # but for comparison we can keep angle as-is since
                # we'll compare the full rotation matrix
            break
    return axis, angle


def canonical_misorientation(R):
    """
    Reduce rotation matrix R to its canonical form in the cubic
    fundamental zone by testing all S_i @ R @ S_j^T and picking
    the one with the smallest angle (disorientation).
    
    Returns (min_angle, canonical_R).
    """
    min_angle = np.pi * 2
    canonical_R = R.copy()
    
    for Si in CUBIC_OPS:
        for Sj in CUBIC_OPS:
            Rp = Si @ R @ Sj.T
            angle = np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1))
            if angle < min_angle - 1e-10:
                min_angle = angle
                canonical_R = Rp.copy()
            elif abs(angle - min_angle) < 1e-10:
                # Tie-break: use lexicographic comparison of the matrix elements
                if tuple(Rp.ravel()) < tuple(canonical_R.ravel()):
                    canonical_R = Rp.copy()
    
    return min_angle, canonical_R


def rotations_equivalent(R1, R2, tol=1e-6):
    """Check if two rotation matrices are equivalent under cubic symmetry."""
    _, C1 = canonical_misorientation(R1)
    _, C2 = canonical_misorientation(R2)
    return np.allclose(C1, C2, atol=tol)


# ============================================================
# Axis enumeration in fundamental zone
# ============================================================
def gcd_list(lst):
    return reduce(gcd, lst)


def axes_in_fundamental_zone(max_index=3):
    """
    Generate rotation axes [h, k, l] in the fundamental zone of
    cubic symmetry: h >= k >= l >= 0, gcd(h,k,l) = 1, not all zero.
    
    max_index controls how high the Miller indices go.
    """
    axes = []
    for h in range(max_index + 1):
        for k in range(h + 1):         # k <= h
            for l in range(k + 1):     # l <= k
                if h == 0 and k == 0 and l == 0:
                    continue
                if gcd_list([h, k, l]) != 1:
                    continue
                axes.append([h, k, l])
    return axes

# ============================================================
# Main enumeration
# ============================================================
def enumerate_gbs(max_sigma=50, max_axis_index=3):
    """
    1. Only use axes in the fundamental zone
    """
    axes = axes_in_fundamental_zone(max_axis_index)
    
    # Track unique misorientations by (sigma, canonical_angle)
    seen_misorientations = {}  # key: (sigma, rounded_angle) -> first axis
    
    triples = []
    skipped = 0
    
    for axis in axes:
        info = GBInformation(axis, max_sigma)
        gb_data = info.get_gb_info()

        for sigma, data in gb_data.items():
            sigma = int(sigma)
            rot_matrix = np.array(data['Rotation matrix'])
            min_angle, _ = canonical_misorientation(rot_matrix)
            
            # Check if we've seen this misorientation
            key = (sigma, round(np.degrees(min_angle), 4))
            if key in seen_misorientations:
                skipped += 1
                continue
            seen_misorientations[key] = tuple(axis)
            
            planes = data['GB plane']
            for plane_set in planes:
                for plane in plane_set:
                    plane = [int(x) for x in plane]
                    triples.append({
                        'axis': list(axis),
                        'sigma': sigma,
                        'plane': plane,
                    })
    
    print(f"Found {len(triples)} unique (axis, sigma, plane) triples and skipped {skipped} duplicates")
    return triples


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    import sys
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    parser = argparse.ArgumentParser(
        description="Enumerate CSL grain boundaries and write them into a YAML config."
    )
    parser.add_argument("--max_sigma", type=int, help="Maximum sigma value")
    parser.add_argument("--max_axis_index", type=int, help="Maximum Miller index for rotation axes")
    parser.add_argument("--config", help="Path to a YAML config file (e.g. small_box.yaml)")
    args = parser.parse_args()

    if not args.config.endswith((".yaml", ".yml")):
        print(f"Error: config file must be a YAML file (.yaml or .yml), got: {args.config}", file=sys.stderr)
        sys.exit(1)

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(args.config, "r") as f:
        config = yaml.load(f)

    print(f"Enumerating unique (axis, sigma, plane) triples (max_sigma={args.max_sigma}, max_axis_index={args.max_axis_index})...")
    triples = enumerate_gbs(max_sigma=args.max_sigma, max_axis_index=args.max_axis_index)

    def flow_list(lst):
        s = CommentedSeq(lst)
        s.fa.set_flow_style()
        return s

    # Build a ruamel-compatible sequence so comments elsewhere are preserved
    gb_seq = CommentedSeq()
    for t in triples:
        entry = CommentedMap()
        entry['axis'] = flow_list(t['axis'])
        entry['sigma'] = t['sigma']
        entry['plane'] = flow_list(t['plane'])
        gb_seq.append(entry)

    config["grain_boundaries"] = gb_seq

    with open(args.config, "w") as f:
        yaml.dump(config, f)

    print(f"Wrote {len(triples)} grain boundaries to {args.config}")
