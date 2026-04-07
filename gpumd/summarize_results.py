"""
summarize_results.py

Scans all configs under results/ and prints a table summary covering:
  1. GB generation — status, runs done vs target, n_atoms, cell dims, best energy
  2. rNEMD results — n_runs, convergence, κ and R_K (mean ± std, full range)

Pass --convergence to also regenerate convergence plots for every rNEMD result.

Usage:
    python summarize_results.py
    python summarize_results.py --convergence
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ase.io import read

GPUMD_ROOT  = Path(__file__).resolve().parent
RESULTS_DIR = GPUMD_ROOT / "results"
CONFIGS_DIR = GPUMD_ROOT / "configs"

sys.path.insert(0, str(GPUMD_ROOT))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_name):
    """Return parsed YAML for config_name, or None if the file does not exist."""
    path = CONFIGS_DIR / f"{config_name}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None


# ---------------------------------------------------------------------------
# GB generation summary
# ---------------------------------------------------------------------------

def read_gb_generation(config_dir, config):
    """
    Return one dict per GB directory found under config_dir/gb_generation/.

    Keys: label, status, runs_done, runs_target,
          n_atoms, cell_x, cell_y, cell_z, best_energy
    """
    gbgen_dir = config_dir / "gb_generation"
    if not gbgen_dir.exists():
        return []

    runs_target = int(config["gb_generation"]["n_runs"]) if config else None

    rows = []
    for gb_dir in sorted(gbgen_dir.iterdir()):
        if not gb_dir.is_dir():
            continue

        label        = gb_dir.name
        summary_path = gb_dir / "summary.csv"

        if not summary_path.exists():
            rows.append({
                "label": label, "status": "in_progress",
                "runs_done": 0, "runs_target": runs_target,
                "n_atoms": None, "cell_x": None, "cell_y": None, "cell_z": None,
                "best_energy": None,
            })
            continue

        try:
            df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            rows.append({
                "label": label, "status": "in_progress",
                "runs_done": 0, "runs_target": runs_target,
                "n_atoms": None, "cell_x": None, "cell_y": None, "cell_z": None,
                "best_energy": None,
            })
            continue
        runs_done   = len(df)
        best_idx    = int(df.loc[df["energy_ev"].idxmin(), "run_index"])
        best_energy = float(df["energy_ev"].min())

        if runs_target is None or runs_done >= runs_target:
            status = "completed"
        else:
            status = "in_progress"

        # Try to read structural info from the best run's traj file
        n_atoms = cell_x = cell_y = cell_z = None
        traj_path = gb_dir / f"run_{best_idx}" / "structure.traj"
        if traj_path.exists():
            try:
                atoms  = read(str(traj_path))
                n_atoms = len(atoms)
                diag   = np.diag(atoms.cell[:])
                cell_x, cell_y, cell_z = diag
            except Exception:
                pass

        rows.append({
            "label": label, "status": status,
            "runs_done": runs_done, "runs_target": runs_target,
            "n_atoms": n_atoms,
            "cell_x": cell_x, "cell_y": cell_y, "cell_z": cell_z,
            "best_energy": best_energy,
        })

    return rows


def print_gbgen_table(rows):
    if not rows:
        print("    (no gb_generation results)")
        return

    w = max(len(r["label"]) for r in rows)

    hdr = (f"  {'GB Label':<{w}}  {'Status':<12}  {'Runs':>6}  "
           f"{'n_atoms':>7}  {'Cell (Å)':>28}  {'Best E (eV)':>14}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in rows:
        runs_str   = f"{r['runs_done']}/{r['runs_target'] if r['runs_target'] is not None else '?'}"
        natoms_str = str(r["n_atoms"]) if r["n_atoms"] is not None else "?"
        if r["cell_x"] is not None:
            cell_str = f"{r['cell_x']:.1f} × {r['cell_y']:.1f} × {r['cell_z']:.1f}"
        else:
            cell_str = "?"
        energy_str = f"{r['best_energy']:.2f}" if r["best_energy"] is not None else "?"

        print(f"  {r['label']:<{w}}  {r['status']:<12}  {runs_str:>6}  "
              f"{natoms_str:>7}  {cell_str:>28}  {energy_str:>14}")


# ---------------------------------------------------------------------------
# rNEMD summary
# ---------------------------------------------------------------------------

def read_rnemd(config_dir):
    """
    Return one dict per GB directory found under config_dir/rnemd/.

    Keys: label, n_runs, n_converged,
          kappa_mean, kappa_std, kappa_min, kappa_max,
          rk_mean, rk_std, rk_min, rk_max
    """
    rnemd_dir = config_dir / "rnemd"
    if not rnemd_dir.exists():
        return []

    rows = []
    for gb_dir in sorted(rnemd_dir.iterdir()):
        if not gb_dir.is_dir():
            continue

        label        = gb_dir.name
        summary_path = gb_dir / "summary.csv"

        if not summary_path.exists():
            rows.append({"label": label, "n_runs": 0})
            continue

        df      = pd.read_csv(summary_path)
        n_runs  = len(df)
        n_conv  = int(df["converged"].sum()) if "converged" in df.columns else None

        kappas  = df["kappa_SI"].values
        rks     = df["R_K_SI"].values
        ddof    = 1 if n_runs > 1 else 0

        rows.append({
            "label":      label,
            "n_runs":     n_runs,
            "n_converged": n_conv,
            "kappa_mean": float(np.nanmean(kappas)),
            "kappa_std":  float(np.nanstd(kappas, ddof=ddof)),
            "kappa_min":  float(np.nanmin(kappas)),
            "kappa_max":  float(np.nanmax(kappas)),
            "rk_mean":    float(np.nanmean(rks)),
            "rk_std":     float(np.nanstd(rks, ddof=ddof)),
            "rk_min":     float(np.nanmin(rks)),
            "rk_max":     float(np.nanmax(rks)),
        })

    return rows


def print_rnemd_table(rows):
    if not rows:
        print("    (no rNEMD results)")
        return

    w = max(len(r["label"]) for r in rows)

    # Two header rows: one for grouping, one for column names
    hdr = (f"  {'GB Label':<{w}}  {'runs':>5}  {'conv':>5}  "
           f"{'κ mean':>8}  {'κ std':>8}  {'κ range [W/(m·K)]':>20}  "
           f"{'R_K mean':>10}  {'R_K std':>10}  {'R_K range [K·m²/W]':>22}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in rows:
        if r["n_runs"] == 0:
            print(f"  {r['label']:<{w}}  (no runs)")
            continue

        conv_str    = (f"{r['n_converged']}/{r['n_runs']}"
                       if r.get("n_converged") is not None else "?")
        kappa_range = f"{r['kappa_min']:.2f}–{r['kappa_max']:.2f}"
        rk_range    = f"{r['rk_min']:.3e}–{r['rk_max']:.3e}"

        print(f"  {r['label']:<{w}}  {r['n_runs']:>5}  {conv_str:>5}  "
              f"{r['kappa_mean']:>8.2f}  {r['kappa_std']:>8.2f}  {kappa_range:>20}  "
              f"{r['rk_mean']:>10.3e}  {r['rk_std']:>10.3e}  {rk_range:>22}")


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def run_convergence(config_dir, config_name, config):
    """Invoke process_gb from convergence_analysis.py for all rNEMD GBs."""
    if config is None:
        print(f"    [convergence] No config YAML found for '{config_name}', skipping.")
        return

    rnemd_cfg = config.get("rnemd")
    if rnemd_cfg is None:
        print(f"    [convergence] No 'rnemd' section in config for '{config_name}', skipping.")
        return

    rnemd_dir = config_dir / "rnemd"
    if not rnemd_dir.exists():
        print(f"    [convergence] No rNEMD results found for '{config_name}'.")
        return

    from convergence_analysis import process_gb

    for gb_dir in sorted(rnemd_dir.iterdir()):
        if not gb_dir.is_dir():
            continue
        label = gb_dir.name
        print(f"    {label}...")
        process_gb(label, config_name, GPUMD_ROOT, rnemd_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print a summary of GB generation and rNEMD results for all configs"
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Also run convergence analysis and regenerate plots for all rNEMD results",
    )
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    config_dirs = sorted(
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "gb_generation").exists()
    )

    if not config_dirs:
        print("No results found.")
        return

    sep = "=" * 80

    for config_dir in config_dirs:
        config_name = config_dir.name
        config      = load_config(config_name)

        print()
        print(sep)
        label_suffix = "" if config else "  [no YAML — run counts are inferred from files]"
        print(f"  Config: {config_name}{label_suffix}")
        print(sep)

        # --- GB generation ---
        print()
        print("  GB Generation")
        print("  " + "─" * 76)
        gbgen_rows = read_gb_generation(config_dir, config)
        print_gbgen_table(gbgen_rows)

        # --- rNEMD ---
        rnemd_rows = read_rnemd(config_dir)
        if rnemd_rows:
            print()
            print("  rNEMD Results")
            print("  " + "─" * 76)
            print_rnemd_table(rnemd_rows)

        # --- Convergence analysis ---
        if args.convergence:
            print()
            print("  Convergence Analysis")
            print("  " + "─" * 76)
            run_convergence(config_dir, config_name, config)

    print()


if __name__ == "__main__":
    main()
