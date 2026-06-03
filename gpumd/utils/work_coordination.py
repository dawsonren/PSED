"""
Provide utilities to coordinate work between different GPUs
for GB generation and kappa calculation (either RNEMD/HNEMD)
"""

import csv
import os
import socket
import time
from pathlib import Path

import yaml

# A claim older than this (with no heartbeat) is treated as stale and
# re-claimable.  Workers refresh their claim's mtime as they run (see
# refresh_claim), so this only needs to exceed the longest gap between
# heartbeats (a single MD stage/cycle), NOT the whole per-GB wall time.
# A full GB takes ~22 h, so the old value of 8 h caused a second worker to
# steal a still-running claim and corrupt the shared run directory.
CLAIM_STALE_HOURS = 6


def gb_label(axis, sigma, plane):
    """Produce a filesystem-safe label, e.g. sigma5_2-10_001"""
    a = "".join(str(x) for x in axis)
    p = "".join(str(x) for x in plane)
    return f"{a}_sigma{sigma}_{p}"


def resolve_results_base(config, gpumd_root):
    """Return the base directory under which all results are written.

    Honors an optional top-level ``results_dir`` key in the config (absolute
    path, with ``~`` and ``$VARS`` expanded) so bulk MD output can live on
    large/unquota'd storage (e.g. /projects or /scratch) instead of the
    quota-limited home directory.  Falls back to ``<gpumd_root>/results`` for
    backward compatibility when the key is absent.

    The path is resolved to its real (symlink-free) location.  This matters
    because callers write GPUMD run.in files with a *relative* potential path
    (os.path.relpath against the run dir).  GPUMD's fopen resolves ".."
    physically, so if the run dir is reached through a symlink (e.g.
    /projects -> /gpfs/projects), a relpath computed against the logical path
    is short one ".." and GPUMD fails with "cannot open ...nep.txt".  Resolving
    here keeps every derived run dir on the same physical basis as the model.
    """
    rd = config.get("results_dir")
    if rd:
        return Path(os.path.expanduser(os.path.expandvars(str(rd)))).resolve()
    return (Path(gpumd_root) / "results").resolve()


def try_claim(claim_path, stale_hours=CLAIM_STALE_HOURS):
    """
    Atomically claim a work slot by creating a sentinel file.

    Returns True if this process acquired the claim, False if another worker
    holds a fresh claim.  A claim is considered stale (and re-claimable) when
    the file is older than stale_hours.  The file records hostname:pid for
    debugging.
    """
    claim_path = Path(claim_path)
    claim_path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_create():
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{socket.gethostname()}:{os.getpid()}\n".encode())
        os.close(fd)

    try:
        _atomic_create()
        return True
    except FileExistsError:
        pass

    # Claim exists — steal it if stale
    try:
        age_hours = (time.time() - claim_path.stat().st_mtime) / 3600
        if age_hours > stale_hours:
            claim_path.unlink(missing_ok=True)
            try:
                _atomic_create()
                return True
            except FileExistsError:
                pass  # Another worker grabbed it first
    except FileNotFoundError:
        pass  # Claim was released between our stat and unlink

    return False


def refresh_claim(claim_path):
    """Update a held claim's mtime so an actively-worked slot is not seen as
    stale by other workers.  Call this periodically (e.g. once per MD cycle)
    while holding the claim.  Silently no-ops if the claim file is gone (it
    was released or stolen); the caller keeps running but its slot may now be
    claimable by another worker."""
    try:
        os.utime(claim_path, None)
    except (FileNotFoundError, OSError):
        pass


def release_claim(claim_path):
    """Remove a claim file once work is done (or on error)."""
    try:
        Path(claim_path).unlink()
    except FileNotFoundError:
        pass  # Already gone — stale claim was stolen, or double-release


def check_gb_generation_status(yaml_path):
    """
    Check completion status of gb_generation runs for all grain boundaries in a config.

    The GPUMD root is inferred as the grandparent of the yaml file (i.e. the parent
    of the configs/ directory). Results are expected at:
        <gpumd_root>/results/<config_stem>/gb_generation/<gb_label>/summary.csv

    Each completed run appends one data row to summary.csv; a GB is "completed" when
    the number of data rows equals n_runs from gb_generation.n_runs in the config.

    Args:
        yaml_path: Path to a unified YAML config file (e.g. configs/small_box.yaml).

    Returns:
        dict mapping gb_label (str) -> {"status": str, "runs_remaining": int}, where
        status is one of:
            "completed"   — summary.csv has at least n_runs data rows (runs_remaining=0)
            "in_progress" — folder exists but summary.csv is absent or incomplete
            "not_started" — no output folder found for this gb_label
    """
    yaml_path = Path(yaml_path).resolve()
    gpumd_root = yaml_path.parent.parent  # .../gpumd/configs/foo.yaml -> .../gpumd/
    config_name = yaml_path.stem

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    n_runs = int(config["gb_generation"]["n_runs"])
    results_dir = resolve_results_base(config, gpumd_root) / config_name / "gb_generation"

    status = {}
    for entry in config["grain_boundaries"]:
        sigma = entry["sigma"]
        if sigma == -1:
            label = "bulk_si"
        else:
            label = gb_label(tuple(entry["axis"]), sigma, tuple(entry["plane"]))

        gb_dir = results_dir / label
        if not gb_dir.exists():
            status[label] = {"status": "not_started", "runs_remaining": n_runs}
            continue

        summary_path = gb_dir / "summary.csv"
        if not summary_path.exists():
            status[label] = {"status": "in_progress", "runs_remaining": n_runs}
            continue

        with open(summary_path, "r") as f:
            n_data_rows = max(sum(1 for _ in csv.reader(f)) - 1, 0)  # subtract header, clamp to 0

        runs_remaining = max(n_runs - n_data_rows, 0)
        status[label] = {
            "status": "completed" if runs_remaining == 0 else "in_progress",
            "runs_remaining": runs_remaining,
        }

    return status


def check_rnemd_status(yaml_path):
    """
    Check completion status of rNEMD runs for all grain boundaries in a config.

    The GPUMD root is inferred as the grandparent of the yaml file.  Results
    are expected at:
        <gpumd_root>/results/<config_stem>/rnemd/<gb_label>/structure_*/run_*/final_atoms.traj

    A GB is "completed" when the number of final_atoms.traj files found across
    all structure_*/run_*/ subdirectories equals n_runs from rnemd.n_runs in
    the config.

    Args:
        yaml_path: Path to a unified YAML config file (e.g. configs/small_box.yaml).

    Returns:
        dict mapping gb_label (str) -> {"status": str, "runs_remaining": int}, where
        status is one of "completed", "in_progress", or "not_started".
    """
    yaml_path = Path(yaml_path).resolve()
    gpumd_root = yaml_path.parent.parent
    config_name = yaml_path.stem

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    n_runs = int(config["rnemd"]["n_runs"])
    results_dir = resolve_results_base(config, gpumd_root) / config_name / "rnemd"

    status = {}
    for entry in config["grain_boundaries"]:
        sigma = entry["sigma"]
        label = "bulk_si" if sigma == -1 else gb_label(
            tuple(entry["axis"]), sigma, tuple(entry["plane"])
        )

        gb_dir = results_dir / label
        if not gb_dir.exists():
            status[label] = {"status": "not_started", "runs_remaining": n_runs}
            continue

        n_completed = 0
        for struct_dir in gb_dir.iterdir():
            if not (struct_dir.is_dir() and struct_dir.name.startswith("structure_")):
                continue
            for run_dir in struct_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    if (run_dir / "final_atoms.traj").exists():
                        n_completed += 1

        runs_remaining = max(n_runs - n_completed, 0)
        status[label] = {
            "status": "completed" if runs_remaining == 0 else (
                "in_progress" if n_completed > 0 else "not_started"
            ),
            "runs_remaining": runs_remaining,
        }

    return status

