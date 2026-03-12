"""
movie_ovito.py  –  Render a GPUMD movie.xyz as a video using OVITO.

Usage:
    python movie_ovito.py --file results/nve_test/rnemd/100_sigma13_0-32/structure_0/run_0/movie.xyz

    # Custom output path, resolution, and frame rate:
    python movie_ovito.py --file results/.../movie.xyz --output out.mp4 --width 1920 --height 540 --fps 15
"""

import argparse
import os
import sys


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
        from ovito.io  import import_file
        from ovito.vis import Viewport, RenderSettings
    except ImportError:
        sys.exit("ERROR: ovito is not importable. Activate the correct conda/venv environment.")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = import_file(xyz_path)
    pipeline.add_to_scene()

    # Count total frames so we can apply --every
    n_frames = pipeline.source.num_frames
    print(f"Total frames in trajectory: {n_frames}")

    frame_list = list(range(0, n_frames, args.every))
    print(f"Frames to render: {len(frame_list)}  (every {args.every})")

    # ── Set up viewport ───────────────────────────────────────────────────────
    # For RNEMD cells the z-axis is very long; look along Y so we see the
    # full thermal-gradient direction (z) vs the lateral direction (x).
    vp = Viewport()
    vp.type = Viewport.Type.Front   # camera looks along -Y, showing X-Z plane

    # Zoom to fit the whole cell into view
    vp.zoom_all(size=(args.width, args.height))

    # ── Render ────────────────────────────────────────────────────────────────
    rs = RenderSettings(
        filename=out_path,
        size=(args.width, args.height),
        frames_per_second=args.fps,
    )

    print("Rendering …")
    vp.render_anim(rs, frame_list=frame_list)
    print(f"Done. Movie written to: {out_path}")


if __name__ == "__main__":
    main()
