"""
compute_coordination.py — Cache coordination-defect descriptors for every GB.

These features (fraction of under/over-coordinated atoms, mean/std coordination
in the GB slab) are not yet part of dataset.csv but are a strong predictor of
the thermal boundary resistance (see ml_pipeline.ipynb section 11). This script
recomputes them from the lowest-energy relaxed structure of each GB and writes a
small cache CSV that the notebook merges by gb_label.

Usage:
    python ml/compute_coordination.py \
        [--dataset /projects/p33174/PSED_results/full/dataset.csv] \
        [--gbgen   /projects/p33174/PSED_results/full/gb_generation] \
        [--d-threshold 10.0] [--bond-cutoff 3.0] \
        [--out ml/gb10_coordination.csv]

The structure selection (lowest energy_ev run from each GB's summary.csv) matches
format_cleaned_data.py, so the cached features line up with the dataset rows.
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
from utils.descriptors import coordination_descriptors, identify_gb_atoms

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", default="/projects/p33174/PSED_results/full/dataset.csv")
p.add_argument("--gbgen",   default="/projects/p33174/PSED_results/full/gb_generation")
p.add_argument("--d-threshold", type=float, default=10.0,
               help="GB-slab half-width in Å (matches the gb10 subset)")
p.add_argument("--bond-cutoff", type=float, default=3.0)
p.add_argument("--out", default=str(SCRIPT_DIR / "gb10_coordination.csv"))
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
    desc = coordination_descriptors(atoms, bond_cutoff=args.bond_cutoff, mask=mask)
    rows.append({
        "gb_label": label,
        f"{prefix}_coord_under": desc["coord_under"],
        f"{prefix}_coord_over":  desc["coord_over"],
        f"{prefix}_coord_mean":  desc["coord_mean"],
        f"{prefix}_coord_std":   desc["coord_std"],
    })
    print(f"  {label:24s} N={len(atoms):6d} under={desc['coord_under']:.3f} "
          f"over={desc['coord_over']:.3f} mean={desc['coord_mean']:.3f}")

out = pd.DataFrame(rows)
out.to_csv(args.out, index=False)
print(f"\nWrote {len(out)} rows x {out.shape[1]} cols -> {args.out}")
