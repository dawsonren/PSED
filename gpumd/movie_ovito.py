"""
movie_ovito.py  -  Render a GPUMD movie.xyz as a video using OVITO.

Usage:
    python movie_ovito.py --file results/nve_test/rnemd/100_sigma13_0-32/structure_0/run_0/movie.xyz

    # Custom output path, resolution, frame rate, GB focus, and cross-section:
    python movie_ovito.py --file results/.../movie.xyz --output out.mp4 \\
        --width 1920 --height 540 --fps 15 --focus 50 --slice-thickness 5

Notes:
    The Z-axis (thermal-gradient / GB-normal direction) is rendered horizontally
    left-to-right. Internally the movie is rendered in portrait orientation and
    then rotated 90 ° clockwise via ffmpeg, so ffmpeg must be on PATH.

    --focus N   : shows only ±N Å around the grain boundary (GB sits at z=L_z/2).
    --slice-thickness T : keeps only atoms within a T-Å-thick slab at y=L_y/2
                         so the GB structure is visible without overlapping bulk.
                         Set to 0 to disable the cross-section (default: 5).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description="Render a GPUMD movie.xyz to a video via OVITO.")
    p.add_argument("--file", required=True,
                   help="Path to movie.xyz, relative to the gpumd directory.")
    p.add_argument("--output",
                   help="Output movie file (mp4/avi/gif). Default: <input_dir>/movie.mp4")
    p.add_argument("--width",  type=int, default=1280, help="Frame width in pixels  (default: 1280)")
    p.add_argument("--height", type=int, default=720,  help="Frame height in pixels (default: 720)")
    p.add_argument("--fps",    type=int, default=10,   help="Frames per second      (default: 10)")
    p.add_argument("--every",  type=int, default=1,
                   help="Render every Nth frame to slim down long trajectories (default: 1 = all)")
    p.add_argument("--focus", type=float, default=None,
                   help="Zoom to ±N Å around the grain boundary (GB at z=L_z/2)")
    p.add_argument("--slice-thickness", type=float, default=5.0, dest="slice_thickness",
                   help="Cross-section slab thickness in Å at y=L_y/2 (default: 5.0). Set 0 to disable.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    gpumd_dir = os.path.dirname(os.path.abspath(__file__))
    xyz_path  = os.path.join(gpumd_dir, args.file)

    if not os.path.isfile(xyz_path):
        sys.exit(f"ERROR: file not found: {xyz_path}")

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(os.path.dirname(xyz_path), "movie.mp4")

    print(f"Input : {xyz_path}")
    print(f"Output: {out_path}")

    # ── OVITO imports (heavy – deferred until paths are validated) ───────────
    try:
        from ovito.io        import import_file
        from ovito.modifiers import SliceModifier
        from ovito.vis       import Viewport
    except ImportError as e:
        print(e)
        sys.exit("ERROR: ovito is not importable. Activate the correct conda/venv environment.")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = import_file(xyz_path)
    pipeline.add_to_scene()

    # Count total frames so we can apply --every
    n_frames = pipeline.source.num_frames
    print(f"Total frames in trajectory: {n_frames}")
    print(f"Frames to render: {len(range(0, n_frames, args.every))}  (every {args.every})")

    # ── Read cell geometry from frame 0 ───────────────────────────────────────
    # cell.matrix is a 3×4 array; columns = [a_vec, b_vec, c_vec, origin]
    data   = pipeline.compute(0)
    m      = data.cell.matrix
    origin = m[:, 3]
    lx, ly, lz = m[0, 0], m[1, 1], m[2, 2]
    x_mid  = origin[0] + lx / 2
    y_mid  = origin[1] + ly / 2
    z_mid  = origin[2] + lz / 2

    # ── Cross-section slice at y_mid ─────────────────────────────────────────
    if args.slice_thickness > 0:
        pipeline.modifiers.append(SliceModifier(
            normal=(0, 1, 0),
            distance=y_mid,
            slab_width=args.slice_thickness,
        ))
        print(f"Cross-section: {args.slice_thickness} Å slab at y={y_mid:.2f} Å")

    # ── Set up viewport ───────────────────────────────────────────────────────
    # We want Z (thermal-gradient axis) to run left-to-right in the final movie.
    # OVITO's world Z is always "up" in any view, so we cannot natively roll the
    # camera. Workaround: render in portrait (dims swapped, Z vertical), then
    # rotate 90 ° clockwise with ffmpeg so Z becomes horizontal.
    render_w = args.height   # portrait width  = desired final height
    render_h = args.width    # portrait height = desired final width

    vp = Viewport(type=Viewport.Type.Ortho)
    vp.camera_dir = (0, -1, 0)   # look along −Y → see X–Z plane

    if args.focus is not None:
        # Center camera on the GB (z_mid) and set FOV to show ±focus in Z.
        # For orthographic, fov is the visible height in world units.
        vp.camera_pos = (x_mid, origin[1] + ly + 1000, z_mid)
        vp.fov        = 2 * args.focus
        print(f"Focus: ±{args.focus} Å around GB at z={z_mid:.2f} Å")
    else:
        vp.zoom_all(size=(render_w, render_h))

    # ── Render portrait, then rotate to landscape (Z horizontal) ─────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(out_path)[1])
    os.close(tmp_fd)

    print("Rendering …")
    vp.render_anim(
        filename=tmp_path,
        size=(render_w, render_h),
        fps=args.fps,
        every_nth=args.every,
    )

    # transpose=1: 90 ° clockwise — what was "up" (high Z) moves to the right,
    # so Z increases left-to-right in the final landscape movie.
    print("Rotating movie so Z-axis runs left-to-right …")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-vf", "transpose=1", out_path],
            check=True, capture_output=True,
        )
        os.unlink(tmp_path)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        os.rename(tmp_path, out_path)
        print(f"Warning: ffmpeg rotation failed ({exc}); portrait video saved instead.")

    print(f"Done. Movie written to: {out_path}")


if __name__ == "__main__":
    main()
