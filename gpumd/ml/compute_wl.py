"""
compute_wl.py — Cache normalized third-order Steinhardt invariants w4, w6.

Companion to compute_bond_order.py. The ŵ_l invariants are a rotation-invariant
fingerprint of the local bonding *shape* (perfect diamond-cubic Si: ŵ_4 ≈ -0.159,
ŵ_6 ≈ +0.013). Individually they are the strongest single linear predictors of the
thermal boundary resistance found so far (e.g. r(gb10_w4_mean, R_K) ≈ +0.66), but
they carry the same local-order information as q_4, q_6 and so do not improve the
nested-CV model beyond the bond-order features — see ml_pipeline.ipynb section 11c.
This cache backs the single-descriptor figure in that section.

Usage:
    python ml/compute_wl.py \
        [--dataset /projects/p33174/PSED_results/full/dataset.csv] \
        [--gbgen   /projects/p33174/PSED_results/full/gb_generation] \
        [--d-threshold 10.0] [--bond-cutoff 3.0] \
        [--out ml/gb10_wl.csv]

Structure selection (lowest energy_ev run per GB) matches the other caches.
Requires sympy (for the Wigner-3j symbols).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GPUMD_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(GPUMD_ROOT))

from ase.io import read
from utils.descriptors import steinhardt_w_descriptors, identify_gb_atoms

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", default="/projects/p33174/PSED_results/full/dataset.csv")
p.add_argument("--gbgen",   default="/projects/p33174/PSED_results/full/gb_generation")
p.add_argument("--d-threshold", type=float, default=10.0,
               help="GB-slab half-width in Å (matches the gb10 subset)")
p.add_argument("--bond-cutoff", type=float, default=3.0)
p.add_argument("--ls", type=int, nargs="+", default=[4, 6])
p.add_argument("--out", default=str(SCRIPT_DIR / "gb10_wl.csv"))
args = p.parse_args()

prefix = f"gb{int(args.d_threshold)}"
df = pd.read_csv(args.dataset)
gbgen = Path(args.gbgen)

rows = []
for label in df["gb_label"]:
    summary = gbgen / label / "summary.csv"
    if not summary.exists():
        print(f"  SKIP {label}: no summary.csv")
        continue
    s = pd.read_csv(summary)
    if s.empty:
        print(f"  SKIP {label}: empty summary.csv")
        continue
    best = int(s.loc[s["energy_ev"].idxmin(), "run_index"])
    traj = gbgen / label / f"run_{best}" / "structure.traj"
    if not traj.exists():
        print(f"  SKIP {label}: missing structure.traj for run_{best}")
        continue

    atoms = read(str(traj))
    mask = identify_gb_atoms(atoms, d_threshold=args.d_threshold)
    if mask.sum() == 0:
        mask = None
    desc = steinhardt_w_descriptors(atoms, bond_cutoff=args.bond_cutoff,
                                    ls=tuple(args.ls), mask=mask)
    row = {"gb_label": label}
    for l in args.ls:
        row[f"{prefix}_w{l}_mean"] = desc[f"w{l}_mean"]
        row[f"{prefix}_w{l}_std"] = desc[f"w{l}_std"]
    rows.append(row)
    wstr = " ".join(f"w{l}={desc[f'w{l}_mean']:+.4f}" for l in args.ls)
    print(f"  {label:24s} N={len(atoms):6d} {wstr}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(args.out, index=False)
print(f"\nWrote {len(out)} rows x {out.shape[1]} cols -> {args.out}")
