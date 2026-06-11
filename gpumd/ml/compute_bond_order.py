"""
compute_bond_order.py — Cache Steinhardt bond-orientational order descriptors.

The coordination-defect features (compute_coordination.py) were the first
descriptor change to beat the disorder scalars for predicting the thermal
boundary resistance (ml_pipeline.ipynb section 11). This script extends that
thread with the natural next step: local Steinhardt order parameters q_4, q_6
over the GB slab. Diamond-cubic Si has sharp ideal values (q_4 ~ 0.509,
q_6 ~ 0.629), so the mean/std of q_4, q_6 measure how far the GB bonding is
pulled from perfect tetrahedral order — a finer probe than coordination counts.

Usage:
    python ml/compute_bond_order.py \
        [--dataset /projects/p33174/PSED_results/full/dataset.csv] \
        [--gbgen   /projects/p33174/PSED_results/full/gb_generation] \
        [--d-threshold 10.0] [--bond-cutoff 3.0] \
        [--out ml/gb10_bondorder.csv]

Structure selection (lowest energy_ev run per GB) matches compute_coordination.py
and format_cleaned_data.py, so the cached features line up with the dataset rows.
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
from utils.descriptors import steinhardt_descriptors, identify_gb_atoms

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", default="/projects/p33174/PSED_results/full/dataset.csv")
p.add_argument("--gbgen",   default="/projects/p33174/PSED_results/full/gb_generation")
p.add_argument("--d-threshold", type=float, default=10.0,
               help="GB-slab half-width in Å (matches the gb10 subset)")
p.add_argument("--bond-cutoff", type=float, default=3.0)
p.add_argument("--ls", type=int, nargs="+", default=[4, 6],
               help="Steinhardt degrees to compute (default: 4 6)")
p.add_argument("--out", default=str(SCRIPT_DIR / "gb10_bondorder.csv"))
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
    desc = steinhardt_descriptors(atoms, bond_cutoff=args.bond_cutoff,
                                  ls=tuple(args.ls), mask=mask)
    row = {"gb_label": label}
    for l in args.ls:
        row[f"{prefix}_q{l}_mean"] = desc[f"q{l}_mean"]
        row[f"{prefix}_q{l}_std"] = desc[f"q{l}_std"]
    rows.append(row)
    qstr = " ".join(f"q{l}={desc[f'q{l}_mean']:.3f}" for l in args.ls)
    print(f"  {label:24s} N={len(atoms):6d} {qstr}")

out = pd.DataFrame(rows)
out.to_csv(args.out, index=False)
print(f"\nWrote {len(out)} rows x {out.shape[1]} cols -> {args.out}")
