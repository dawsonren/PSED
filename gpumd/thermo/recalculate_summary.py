"""
recalculate_summary.py — Rebuild summary.csv and aggregate.csv from raw run data.

Use this after a SLURM timeout cut short a run_rnemd.py job. Individual run
directories (run_0/, run_1/, …) save their .npy data before the process is
killed, but summary.csv is only written once all runs in an invocation
complete — so it can be incomplete or missing.

This script walks every completed run directory, reloads the saved numpy
arrays, recomputes kappa/R_K/J identically to run_rnemd.py, and overwrites
summary.csv + aggregate.csv with the full set of results.

Usage:
    python recalculate_summary.py --config ../configs/full.yaml
    python recalculate_summary.py --config ../configs/full.yaml --gb 110_sigma27_-552
"""

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import yaml

from ase.io import read

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.rnemd_stats import check_steady_state, aggregate_run_results

# ---------------------------------------------------------------------------
# CLI and config
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Rebuild rNEMD summary.csv / aggregate.csv from raw run data"
)
parser.add_argument("--config", required=True, help="Path to unified YAML config")
parser.add_argument("--gb", default=None, help="Restrict to a specific GB label")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
GPUMD_ROOT = SCRIPT_DIR.parent

with open(args.config) as f:
    config = yaml.safe_load(f)

CONFIG_NAME = Path(args.config).stem

rnemd_cfg        = config["rnemd"]
NBINS            = int(rnemd_cfg["nbins"])
COLD_BIN         = NBINS // 4
HOT_BIN          = 3 * NBINS // 4
STEPS_PER_CYCLE  = int(rnemd_cfg["steps_per_cycle"])
TIMESTEP_FS      = float(rnemd_cfg["timestep_fs"])
N_CYCLES         = int(rnemd_cfg["n_cycles"])
BULK_SI_LABEL    = "bulk_si"
M_SI_AMU         = 28.085

RNEMD_RESULTS_DIR = str(GPUMD_ROOT / "results" / CONFIG_NAME / "rnemd")

# ---------------------------------------------------------------------------
# Physics — copied verbatim from run_rnemd.py (uses the globals above)
# ---------------------------------------------------------------------------

def compute_tbr_and_kappa(temps_avg, velocities_hc, bin_centers_angstrom,
                           cross_section_angstrom2, total_time_fs, is_bulk=False):
    v_hot  = velocities_hc[:, 0]
    v_cold = velocities_hc[:, 1]
    delta_KE_eV    = 0.5 * M_SI_AMU * (v_hot**2 - v_cold**2)
    total_energy_J = np.sum(delta_KE_eV) * 1.602176634e-19

    A_m2 = cross_section_angstrom2 * 1e-20
    t_s  = total_time_fs * 1e-15
    J    = total_energy_J / (2.0 * A_m2 * t_s)

    margin         = 1
    gb_bin         = NBINS // 2
    left_slice     = slice(COLD_BIN + margin, gb_bin - margin)
    right_slice    = slice(gb_bin + margin, HOT_BIN - margin)
    cold_dup_slice = slice(margin, COLD_BIN - margin)
    hot_dup_slice  = slice(HOT_BIN + margin, NBINS - margin)

    x_cold_dup   = bin_centers_angstrom[cold_dup_slice]
    T_cold_dup   = temps_avg[cold_dup_slice]
    x_hot_dup    = bin_centers_angstrom[hot_dup_slice]
    T_hot_dup    = temps_avg[hot_dup_slice]
    cold_dup_fit = np.polyfit(x_cold_dup, T_cold_dup, 1)
    hot_dup_fit  = np.polyfit(x_hot_dup,  T_hot_dup,  1)

    bin_width            = bin_centers_angstrom[1] - bin_centers_angstrom[0]
    box_length_angstrom  = bin_centers_angstrom[-1] + bin_width / 2.0

    if is_bulk:
        rising_slice  = slice(COLD_BIN + margin, HOT_BIN - margin)
        x_rising      = bin_centers_angstrom[rising_slice]
        T_rising      = temps_avg[rising_slice]
        rising_fit    = np.polyfit(x_rising, T_rising, 1)

        rising_slope  = rising_fit[0]
        falling_slope = (-hot_dup_fit[0] + (-cold_dup_fit[0])) / 2.0

        dTdx_rising_SI  = rising_slope  * 1e10
        dTdx_falling_SI = falling_slope * 1e10
        dTdx_SI         = (rising_slope + falling_slope) / 2.0 * 1e10

        kappa_rising  = abs(J / dTdx_rising_SI)  if dTdx_rising_SI  > 0 else np.nan
        kappa_falling = abs(J / dTdx_falling_SI) if dTdx_falling_SI > 0 else np.nan
        kappa         = np.nanmean([kappa_rising, kappa_falling])

        return {
            "R_K_SI": np.nan,
            "kappa_SI": kappa,
            "kappa_cold_SI": kappa_rising,
            "kappa_hot_SI": kappa_falling,
            "J_SI": J,
            "delta_T": np.nan,
            "dTdx_K_per_m": dTdx_SI,
            "left_fit": rising_fit,
            "right_fit": rising_fit,
            "cold_dup_fit": cold_dup_fit,
            "hot_dup_fit": hot_dup_fit,
        }

    x_left   = bin_centers_angstrom[left_slice]
    T_left   = temps_avg[left_slice]
    x_right  = bin_centers_angstrom[right_slice]
    T_right  = temps_avg[right_slice]
    left_fit  = np.polyfit(x_left,  T_left,  1)
    right_fit = np.polyfit(x_right, T_right, 1)

    cold_slope   = (left_fit[0]  + (-cold_dup_fit[0])) / 2.0
    hot_slope    = (right_fit[0] + (-hot_dup_fit[0]))  / 2.0
    avg_slope    = (cold_slope + hot_slope) / 2.0
    dTdx_SI      = avg_slope  * 1e10
    cold_dTdx_SI = cold_slope * 1e10
    hot_dTdx_SI  = hot_slope  * 1e10

    kappa      = abs(J / dTdx_SI)      if abs(dTdx_SI)      > 0 else np.nan
    kappa_cold = abs(J / cold_dTdx_SI) if abs(cold_dTdx_SI) > 0 else np.nan
    kappa_hot  = abs(J / hot_dTdx_SI)  if abs(hot_dTdx_SI)  > 0 else np.nan

    x_gb           = bin_centers_angstrom[gb_bin]
    T_left_at_gb   = np.polyval(left_fit,  x_gb)
    T_right_at_gb  = np.polyval(right_fit, x_gb)
    delta_T        = abs(T_left_at_gb - T_right_at_gb)
    R_K            = delta_T / J if J > 0 else np.nan

    T_cold_dup_at_dup_gb = np.polyval(cold_dup_fit, 0.0)
    T_hot_dup_at_dup_gb  = np.polyval(hot_dup_fit,  box_length_angstrom)
    delta_T_dup = abs(T_cold_dup_at_dup_gb - T_hot_dup_at_dup_gb)
    R_K_dup     = delta_T_dup / J if J > 0 else np.nan
    R_K_avg     = np.nanmean([R_K, R_K_dup])

    return {
        "R_K_SI": R_K_avg,
        "kappa_SI": kappa,
        "kappa_cold_SI": kappa_cold,
        "kappa_hot_SI": kappa_hot,
        "J_SI": J,
        "delta_T": delta_T,
        "dTdx_K_per_m": dTdx_SI,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "cold_dup_fit": cold_dup_fit,
        "hot_dup_fit": hot_dup_fit,
    }

# ---------------------------------------------------------------------------
# Per-run reconstruction
# ---------------------------------------------------------------------------

def recompute_run(run_dir, structure_index, run_index, gb_label_str):
    """Load saved numpy data and recompute physics for one completed run."""
    temps_times   = np.load(os.path.join(run_dir, "temps_times.npy"))
    velocities_hc = np.load(os.path.join(run_dir, "velocities_hc.npy"))
    bin_centers   = np.load(os.path.join(run_dir, "bin_centers.npy"))
    atoms         = read(os.path.join(run_dir, "final_atoms.traj"))

    cross_section = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))
    total_time_fs = N_CYCLES * STEPS_PER_CYCLE * TIMESTEP_FS
    energy_ev     = atoms.info.get("energy_ev", np.nan)

    converged, _, _ = check_steady_state(temps_times)

    cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, len(temps_times) + 1)[:, None]
    temps_avg      = cumulative_avg[-1]

    result = compute_tbr_and_kappa(
        temps_avg, velocities_hc, bin_centers,
        cross_section, total_time_fs,
        is_bulk=(gb_label_str == BULK_SI_LABEL),
    )
    result.update({
        "structure_index": structure_index,
        "run_index": run_index,
        "energy_ev": energy_ev,
        "n_atoms": len(atoms),
        "converged": converged,
    })
    return result

# ---------------------------------------------------------------------------
# Per-GB summary rebuild
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "structure_index", "run_index", "energy_ev",
    "R_K_SI", "kappa_SI", "kappa_cold_SI", "kappa_hot_SI",
    "J_SI", "delta_T", "n_atoms", "converged",
]
AGG_FIELDS = [
    "structure_index", "n_runs",
    "kappa_mean", "kappa_std", "R_K_mean", "R_K_std", "J_mean", "J_std",
]


def rebuild_gb(gb_label_str):
    gb_dir = os.path.join(RNEMD_RESULTS_DIR, gb_label_str)
    if not os.path.isdir(gb_dir):
        print(f"  Directory not found: {gb_dir}")
        return

    all_run_results = []

    struct_dirs = sorted(
        d for d in os.listdir(gb_dir)
        if d.startswith("structure_") and os.path.isdir(os.path.join(gb_dir, d))
    )
    if not struct_dirs:
        print(f"  No structure_* directories found in {gb_dir}")
        return

    for struct_name in struct_dirs:
        struct_idx  = int(struct_name.split("_")[1])
        struct_path = os.path.join(gb_dir, struct_name)

        run_dirs = sorted(
            d for d in os.listdir(struct_path)
            if d.startswith("run_") and os.path.isdir(os.path.join(struct_path, d))
            and os.path.exists(os.path.join(struct_path, d, "final_atoms.traj"))
        )
        if not run_dirs:
            print(f"  {struct_name}: no completed runs, skipping")
            continue

        print(f"  {struct_name}: {len(run_dirs)} completed run(s)")
        for run_name in run_dirs:
            run_idx  = int(run_name.split("_")[1])
            run_path = os.path.join(struct_path, run_name)
            try:
                result = recompute_run(run_path, struct_idx, run_idx, gb_label_str)
                all_run_results.append(result)
                print(f"    run_{run_idx}: κ={result['kappa_SI']:.2f} W/(m·K), "
                      f"R_K={result['R_K_SI']:.3e} K·m²/W, "
                      f"conv={'yes' if result['converged'] else 'NO'}")
            except Exception as exc:
                print(f"    run_{run_idx}: ERROR — {exc}")

    if not all_run_results:
        print(f"  No results recovered for {gb_label_str}, skipping CSV write")
        return

    # Write summary.csv (overwrite with all recovered runs)
    summary_path = os.path.join(gb_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_run_results)
    print(f"  Wrote {summary_path} ({len(all_run_results)} rows)")

    # Write aggregate.csv
    # Use only the first structure's runs (matches run_rnemd.py behaviour: one
    # best structure is selected; rebuild uses whichever structure(s) exist).
    aggregate = aggregate_run_results(all_run_results)
    agg_row   = {"structure_index": all_run_results[0]["structure_index"], **aggregate}
    agg_path  = os.path.join(gb_dir, "aggregate.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(agg_row)
    print(f"  Wrote {agg_path} (n={aggregate['n_runs']} runs, "
          f"κ={aggregate['kappa_mean']:.2f} ± {aggregate['kappa_std']:.2f} W/(m·K))")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(RNEMD_RESULTS_DIR):
        print(f"rNEMD results directory not found: {RNEMD_RESULTS_DIR}")
        return

    if args.gb:
        gb_labels = [args.gb]
    else:
        gb_labels = sorted(
            d for d in os.listdir(RNEMD_RESULTS_DIR)
            if os.path.isdir(os.path.join(RNEMD_RESULTS_DIR, d))
        )

    print(f"Config: {CONFIG_NAME}  |  rNEMD dir: {RNEMD_RESULTS_DIR}")
    print(f"Processing {len(gb_labels)} GB label(s)\n")

    for label in gb_labels:
        print(f"{'='*60}")
        print(f"  {label}")
        rebuild_gb(label)
        print()


if __name__ == "__main__":
    main()
