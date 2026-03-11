import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

# ── 1. Convert .traj → .extxyz ─────────────────────────────────────────────

def traj_to_extxyz(traj_path: str, out_path: str, frame: int = -1):
    """
    Convert an ASE .traj file to .extxyz.
    frame=-1 reads the last frame; use slice(None) for all frames.
    """
    atoms = read(traj_path, index=frame)
    write(out_path, atoms)
    print(f"Written: {out_path}  ({len(atoms)} atoms)")
    return out_path


# ── 2. OVITO pipeline ───────────────────────────────────────────────────────

def analyze_gb(xyz_path: str, export_path: str | None = None):
    """
    Load a bicrystal .extxyz, run PTM to classify local structure,
    then isolate GB atoms (anything that isn't cubic-diamond bulk).

    Returns the OVITO pipeline so you can inspect or render further.
    """
    from ovito.io import import_file
    from ovito.modifiers import (
        PolyhedralTemplateMatchingModifier,
        ExpressionSelectionModifier,
        DeleteSelectedModifier,
        ColorCodingModifier,
    )

    pipeline = import_file(xyz_path)

    # ── PTM: classify each atom's local structure ──────────────────────────
    ptm = PolyhedralTemplateMatchingModifier(
        output_orientation=True,   # quaternion per atom → useful for misorientation
        output_ordering=True,
        output_rmsd=True,
    )
    # For silicon we only need cubic diamond; disable others to save time
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled        = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled        = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled        = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled        = False
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled  = True
    ptm.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled    = False
    pipeline.modifiers.append(ptm)

    # ── Select bulk diamond atoms, then delete them ────────────────────────
    # Structure type 0 = "Other" (unclassified / GB atoms we want to keep)
    # Structure type 4 = cubic diamond (bulk Si we want to remove)
    # The PTM modifier stores results in the "Structure Type" particle property.
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression="StructureType == 4")
        # StructureType integer codes:
        #   0 = Other, 1 = FCC, 2 = HCP, 3 = BCC, 4 = Cubic diamond, 5 = Hex diamond
    )
    pipeline.modifiers.append(DeleteSelectedModifier())

    # ── Color remaining atoms by their PTM RMSD (structural disorder) ──────
    pipeline.modifiers.append(
        ColorCodingModifier(
            property="RMSD",
            start_value=0.0,
            end_value=0.15,        # tune to your system; ~0.1–0.2 Å typical
            gradient=ColorCodingModifier.Rainbow(),
        )
    )

    # ── Compute and inspect ────────────────────────────────────────────────
    data = pipeline.compute()
    n_gb = data.particles.count
    print(f"GB + defect atoms remaining: {n_gb}")

    pos = data.particles["Position"].array        # shape (n_gb, 3)
    rmsd = data.particles["RMSD"].array

    print(f"  z range of GB atoms: {pos[:, 2].min():.2f} – {pos[:, 2].max():.2f} Å")
    print(f"  mean PTM RMSD:       {rmsd.mean():.4f} Å")

    # ── Optional: export GB-only atoms back to extxyz ─────────────────────
    if export_path:
        from ovito.io import export_file
        export_file(pipeline, export_path, "xyz", columns=[
            "Particle Identifier", "Position.X", "Position.Y", "Position.Z",
            "Structure Type", "RMSD",
        ])
        print(f"GB atoms exported to: {export_path}")

    return pipeline, data


# ── 3. Quick matplotlib cross-section plot ──────────────────────────────────

def plot_gb_crosssection(data, slab_thickness: float = 5.0, save_path: str | None = None):
    """
    Project a thin slab of GB atoms onto the XZ plane and color by PTM RMSD.
    Good for checking GB structure and planarity.

    slab_thickness : Å, centered on the GB (assumed near z-midpoint of cell)
    """
    import matplotlib.pyplot as plt

    pos  = data.particles["Position"].array
    rmsd = data.particles["RMSD"].array

    # Slice a thin slab in y centered on the cell midpoint
    cell = data.cell[...]           # 3×3 cell matrix
    y_mid = cell[1, 1] / 2
    mask = np.abs(pos[:, 1] - y_mid) < slab_thickness / 2
    pos_slab  = pos[mask]
    rmsd_slab = rmsd[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        pos_slab[:, 0], pos_slab[:, 2],
        c=rmsd_slab, cmap="plasma",
        s=6, vmin=0, vmax=0.15,
    )
    plt.colorbar(sc, ax=ax, label="PTM RMSD (Å)")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("z (Å)")
    ax.set_title(f"GB cross-section  |  slab Δy = {slab_thickness} Å")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    xyz  = traj_to_extxyz("results/small_box/gb_generation/110_sigma17_-22-3/run_0/structure.traj", "bicrystal.xyz", frame=-1)
    pipe, data = analyze_gb(xyz, export_path="gb_atoms.xyz")
    plot_gb_crosssection(data, slab_thickness=5.0, save_path="gb_crosssection.png")