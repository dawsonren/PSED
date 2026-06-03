"""
visualize_ovito.py  -  Render GB cross-section PNGs from structure.traj files.

Scans results/<config>/gb_generation/<gb>/run_*/structure.traj, runs OVITO's
PTM modifier to isolate GB atoms, and writes a cross-section PNG next to each
traj file. The GB is assumed to sit at z = L_z/2.

Usage:
    python visualize_ovito.py <config> [--slab-thickness ANG] [--z-width NM] [--axis {x,y}]

Example:
    python visualize_ovito.py tersoff_test --slab-thickness 5 --z-width 4 --axis y
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

GPUMD_ROOT  = Path(__file__).resolve().parent
RESULTS_DIR = GPUMD_ROOT / "results"
CONFIGS_DIR = GPUMD_ROOT / "configs"


def traj_to_extxyz(traj_path: Path, out_path: Path, frame: int = -1):
    atoms = read(str(traj_path), index=frame)
    write(str(out_path), atoms)
    print(f"    wrote {out_path.name} ({len(atoms)} atoms)")


def analyze_gb(xyz_path: Path):
    """Load bicrystal .extxyz, classify via PTM, delete bulk, return pipeline data."""
    from ovito.io import import_file
    from ovito.modifiers import (
        PolyhedralTemplateMatchingModifier,
        ExpressionSelectionModifier,
        DeleteSelectedModifier,
        ColorCodingModifier,
    )

    pipeline = import_file(str(xyz_path))

    ptm = PolyhedralTemplateMatchingModifier(
        output_orientation=True,
        output_ordering=True,
        output_rmsd=True,
    )
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled           = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled           = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled           = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled           = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled   = False
    pipeline.modifiers.append(ptm)

    # StructureType 4 = cubic diamond (bulk Si) → delete
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="StructureType == 4"))
    pipeline.modifiers.append(DeleteSelectedModifier())

    pipeline.modifiers.append(
        ColorCodingModifier(
            property="RMSD",
            start_value=0.0,
            end_value=0.15,
            gradient=ColorCodingModifier.Rainbow(),
        )
    )

    return pipeline.compute()


def plot_gb_crosssection(data, slab_thickness_ang: float, z_width_nm: float,
                         slice_axis: str, save_path: Path):
    """
    Project a thin slab of GB atoms and color by PTM RMSD.

    slice_axis="y": thin slab in y → plot x vs z (look down y).
    slice_axis="x": thin slab in x → plot y vs z (look down x).
    z_width_nm    : total width in z centered on GB (z = L_z/2) to display.
    """
    import matplotlib.pyplot as plt

    pos  = data.particles["Position"].array
    rmsd = data.particles["RMSD"].array
    cell = data.cell[...]

    slice_idx, plane_idx, plane_label = (
        (1, 0, "x") if slice_axis == "y" else (0, 1, "y")
    )

    slice_mid = cell[slice_idx, slice_idx] / 2
    z_mid     = cell[2, 2] / 2
    z_half    = z_width_nm * 10.0 / 2  # nm → Å

    mask = (
        (np.abs(pos[:, slice_idx] - slice_mid) < slab_thickness_ang / 2)
        & (np.abs(pos[:, 2] - z_mid) < z_half)
    )

    pos_slab  = pos[mask]
    rmsd_slab = rmsd[mask]

    if len(pos_slab) == 0:
        print(f"    [warn] no GB atoms in requested slab; skipping {save_path.name}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        pos_slab[:, 2], pos_slab[:, plane_idx],
        c=rmsd_slab, cmap="plasma",
        s=6, vmin=0, vmax=0.15,
    )
    plt.colorbar(sc, ax=ax, label="PTM RMSD (Å)")
    ax.set_xlabel("z (Å)")
    ax.set_ylabel(f"{plane_label} (Å)")
    ax.set_title(
        f"GB cross-section  |  slab Δ{slice_axis} = {slab_thickness_ang} Å, "
        f"z-window = {z_width_nm} nm"
    )
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close(fig)
    print(f"    saved {save_path.name}")


def process_traj(traj_path: Path, slab_thickness_ang: float, z_width_nm: float,
                 slice_axis: str):
    run_dir  = traj_path.parent
    xyz_path = run_dir / f"{traj_path.stem}_last.xyz"
    png_path = run_dir / f"{traj_path.stem}_gb_crosssection_{slice_axis}.png"

    print(f"  [traj] {traj_path.relative_to(RESULTS_DIR)}")
    traj_to_extxyz(traj_path, xyz_path)
    try:
        data = analyze_gb(xyz_path)
        plot_gb_crosssection(data, slab_thickness_ang, z_width_nm, slice_axis, png_path)
    finally:
        xyz_path.unlink(missing_ok=True)


def iter_traj_files(config_name: str):
    """Yield traj files under results/<config>/{gb_generation,rnemd}/."""
    config_dir = RESULTS_DIR / config_name
    gbgen_dir  = config_dir / "gb_generation"
    rnemd_dir  = config_dir / "rnemd"

    if not gbgen_dir.exists() and not rnemd_dir.exists():
        print(f"error: neither {gbgen_dir} nor {rnemd_dir} exists", file=sys.stderr)
        sys.exit(1)

    found_any = False

    if gbgen_dir.exists():
        for gb_dir in sorted(p for p in gbgen_dir.iterdir() if p.is_dir()):
            initial_path = gb_dir / "initial.traj"
            if initial_path.exists():
                found_any = True
                yield initial_path
            for run_dir in sorted(p for p in gb_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                traj_path = run_dir / "structure.traj"
                if traj_path.exists():
                    found_any = True
                    yield traj_path
                else:
                    print(f"  [skip] no structure.traj in {run_dir.relative_to(RESULTS_DIR)}")

    if rnemd_dir.exists():
        for gb_dir in sorted(p for p in rnemd_dir.iterdir() if p.is_dir()):
            for struct_dir in sorted(p for p in gb_dir.iterdir() if p.is_dir() and p.name.startswith("structure_")):
                for run_dir in sorted(p for p in struct_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
                    traj_path = run_dir / "final_atoms.traj"
                    if traj_path.exists():
                        found_any = True
                        yield traj_path
                    else:
                        print(f"  [skip] no final_atoms.traj in {run_dir.relative_to(RESULTS_DIR)}")

    if not found_any:
        print(f"  (no traj files found under {config_dir})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Render GB cross-section PNGs from structure.traj files under results/<config>/gb_generation/.",
    )
    p.add_argument("config", help="Config name (file under configs/, without .yaml).")
    p.add_argument("--slab-thickness", type=float, default=10.0, dest="slab_thickness",
                   help="Slab thickness in Å along the slice axis (default: 10.0).")
    p.add_argument("--z-width", type=float, default=10.0, dest="z_width",
                   help="Plot window along z in nm, centered on the GB (default: 10.0).")
    p.add_argument("--axis", choices=("x", "y"), default="y",
                   help="Which axis to slice along (default: y).")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = CONFIGS_DIR / f"{args.config}.yaml"
    if not cfg_path.exists():
        print(f"error: config not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    print(f"visualize_ovito: config={args.config}  axis={args.axis}  "
          f"slab={args.slab_thickness} Å  z-window={args.z_width} nm")

    for traj_path in iter_traj_files(args.config):
        try:
            process_traj(traj_path, args.slab_thickness, args.z_width, args.axis)
        except Exception as e:
            print(f"    [error] {traj_path}: {e}")


if __name__ == "__main__":
    main()
