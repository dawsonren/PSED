"""
convergence_analysis.py

Plots kappa and TBR (R_K) as a function of cycle for each grain boundary in a
config, using cumulative averages of the bin temperatures over time.  One plot
per grain boundary, saved as convergence.png in the GB's rNEMD results directory.

Usage:
    python convergence_analysis.py --config configs/nve_test_xlarge.yaml
    python convergence_analysis.py --config configs/nve_test_xlarge.yaml --gb 100_sigma13_0-32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.work_coordination import gb_label

M_SI_AMU = 28.085  # Si atomic mass in amu


def compute_convergence(temps_times, velocities_hc, bin_centers, cross_section_A2,
                        steps_per_cycle, timestep_fs, nbins, cold_bin, hot_bin):
    """
    Compute kappa [W/(m·K)] and R_K [K·m²/W] at every cycle N using the
    cumulative-average temperature profile and cumulative heat flux up to N.

    Parameters
    ----------
    temps_times : ndarray, shape (n_cycles, nbins)
    velocities_hc : ndarray, shape (n_cycles, 2)  — [v_hot, v_cold] in ASE units
    bin_centers : ndarray, shape (nbins,)  — positions in Å
    cross_section_A2 : float  — cross-section area in Å²
    steps_per_cycle : int
    timestep_fs : float
    nbins, cold_bin, hot_bin : int

    Returns
    -------
    kappas, R_Ks : ndarray, shape (n_cycles,)
    """
    n_cycles = len(temps_times)
    A_m2 = cross_section_A2 * 1e-20

    v_hot  = velocities_hc[:, 0]
    v_cold = velocities_hc[:, 1]
    delta_KE_eV = 0.5 * M_SI_AMU * (v_hot**2 - v_cold**2)
    cumulative_energy_J = np.cumsum(delta_KE_eV) * 1.602176634e-19

    margin = 1
    gb_bin = nbins // 2
    x_left  = bin_centers[cold_bin + margin : gb_bin]
    x_right = bin_centers[gb_bin : hot_bin - margin]
    x_gb    = bin_centers[gb_bin]

    kappas = np.full(n_cycles, np.nan)
    R_Ks   = np.full(n_cycles, np.nan)

    cumul_sum = np.zeros(nbins)
    for N in range(1, n_cycles + 1):
        cumul_sum += temps_times[N - 1]
        cumul_avg  = cumul_sum / N

        t_s = N * steps_per_cycle * timestep_fs * 1e-15
        J   = cumulative_energy_J[N - 1] / (2.0 * A_m2 * t_s)

        T_left  = cumul_avg[cold_bin + margin : gb_bin]
        T_right = cumul_avg[gb_bin : hot_bin - margin]

        left_fit  = np.polyfit(x_left,  T_left,  1)
        right_fit = np.polyfit(x_right, T_right, 1)

        avg_slope = (left_fit[0] + right_fit[0]) / 2.0  # K/Å
        dTdx_SI   = avg_slope * 1e10                     # K/m

        kappas[N - 1] = abs(J / dTdx_SI) if abs(dTdx_SI) > 0 else np.nan

        delta_T  = abs(np.polyval(left_fit, x_gb) - np.polyval(right_fit, x_gb))
        R_Ks[N - 1] = delta_T / J if J > 0 else np.nan

    return kappas, R_Ks


def plot_convergence(all_kappas, all_R_Ks, run_labels, gb_label_str, out_path):
    """Save a two-panel convergence plot (kappa and R_K vs cycle)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Convergence — {gb_label_str}", fontsize=12)

    colors = plt.cm.tab10.colors
    for i, (kappas, R_Ks) in enumerate(zip(all_kappas, all_R_Ks)):
        c      = colors[i % len(colors)]
        cycles = np.arange(1, len(kappas) + 1)
        axes[0].plot(cycles, kappas, color=c, linewidth=0.8, label=run_labels[i])
        axes[1].plot(cycles, R_Ks,   color=c, linewidth=0.8, label=run_labels[i])

    axes[0].set_ylabel("κ [W/(m·K)]")
    axes[0].set_title("Bulk thermal conductivity")
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel("R_K [K·m²/W]")
    axes[1].set_title("Kapitza resistance (TBR)")
    axes[1].set_xlabel("Cycle")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def process_gb(gb_label_str, config_name, gpumd_root, rnemd_cfg):
    rnemd_dir = gpumd_root / "results" / config_name / "rnemd" / gb_label_str
    if not rnemd_dir.exists():
        print(f"  No rNEMD results found for {gb_label_str}, skipping.")
        return

    nbins          = int(rnemd_cfg["nbins"])
    cold_bin       = nbins // 4
    hot_bin        = 3 * nbins // 4
    steps_per_cycle = int(rnemd_cfg["steps_per_cycle"])
    timestep_fs    = float(rnemd_cfg["timestep_fs"])

    all_kappas, all_R_Ks, run_labels = [], [], []

    for struct_dir in sorted(rnemd_dir.glob("structure_*")):
        for run_dir in sorted(struct_dir.glob("run_*")):
            required = [
                run_dir / "bin_temps.csv",
                run_dir / "velocities_hc.npy",
                run_dir / "bin_centers.npy",
                run_dir / "final_atoms.traj",
            ]
            if not all(p.exists() for p in required):
                continue

            temps_times    = pd.read_csv(required[0], index_col=0).values
            velocities_hc  = np.load(required[1])
            bin_centers    = np.load(required[2])
            atoms          = read(str(required[3]))
            cross_section  = float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))

            kappas, R_Ks = compute_convergence(
                temps_times, velocities_hc, bin_centers, cross_section,
                steps_per_cycle, timestep_fs, nbins, cold_bin, hot_bin,
            )
            all_kappas.append(kappas)
            all_R_Ks.append(R_Ks)
            run_labels.append(f"{struct_dir.name}/{run_dir.name}")
            print(f"    Loaded {struct_dir.name}/{run_dir.name} "
                  f"({len(temps_times)} cycles, "
                  f"final κ={kappas[-1]:.2f} W/(m·K), "
                  f"R_K={R_Ks[-1]:.3e} K·m²/W)")

    if not all_kappas:
        print(f"  No complete run data found for {gb_label_str}.")
        return

    plot_convergence(all_kappas, all_R_Ks, run_labels, gb_label_str,
                     rnemd_dir / "convergence.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot kappa/TBR convergence from rNEMD bin_temps.csv files"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--gb", default=None, help="Process a specific GB label")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config_name = config_path.stem
    gpumd_root  = Path(__file__).resolve().parent

    _raw_gbs   = config["grain_boundaries"]
    no_gb_mode = len(_raw_gbs) == 1 and _raw_gbs[0].get("sigma") == -1

    if args.gb:
        gb_labels = [args.gb]
    elif no_gb_mode:
        gb_labels = ["bulk_si"]
    else:
        gb_labels = [
            gb_label(tuple(e["axis"]), int(e["sigma"]), tuple(e["plane"]))
            for e in _raw_gbs
        ]

    for label in gb_labels:
        print(f"\nProcessing {label}...")
        process_gb(label, config_name, gpumd_root, config["rnemd"])


if __name__ == "__main__":
    main()
