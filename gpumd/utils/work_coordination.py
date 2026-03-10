"""
Provide utilities to coordinate work between different GPUs
for GB generation and kappa calculation (either RNEMD/HNEMD)
"""

import csv
from pathlib import Path

import yaml


def gb_label(axis, sigma, plane):
    """Produce a filesystem-safe label, e.g. sigma5_2-10_001"""
    a = "".join(str(x) for x in axis)
    p = "".join(str(x) for x in plane)
    return f"{a}_sigma{sigma}_{p}"


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
    results_dir = gpumd_root / "results" / config_name / "gb_generation"

    status = {}
    for entry in config["grain_boundaries"]:
        axis = tuple(entry["axis"])
        sigma = entry["sigma"]
        plane = tuple(entry["plane"])
        label = gb_label(axis, sigma, plane)

        gb_dir = results_dir / label
        if not gb_dir.exists():
            status[label] = {"status": "not_started", "runs_remaining": n_runs}
            continue

        summary_path = gb_dir / "summary.csv"
        if not summary_path.exists():
            status[label] = {"status": "in_progress", "runs_remaining": n_runs}
            continue

        with open(summary_path, "r") as f:
            n_data_rows = sum(1 for _ in csv.reader(f)) - 1  # subtract header

        runs_remaining = max(n_runs - n_data_rows, 0)
        status[label] = {
            "status": "completed" if runs_remaining == 0 else "in_progress",
            "runs_remaining": runs_remaining,
        }

    return status

