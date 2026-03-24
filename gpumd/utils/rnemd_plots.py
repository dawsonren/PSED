"""
rnemd_plots.py — Diagnostic plotting utilities for rNEMD simulations.

Functions
---------
plot_temperature_profile
    Three-panel static diagnostic: per-cycle profiles, converged cumulative
    average with linear fits, and steady-state convergence check.
plot_temperature_profile_animated
    Single-line animation: one frame per cycle showing the current bin
    temperatures and the cumulative-average bulk fits evolving over time.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_temperature_profile(temps_times, bin_centers, result, out_dir,
                              label, run_index, converged, max_dev,
                              cold_bin, hot_bin, nbins):
    """
    Plot the evolving and converged temperature profile with the linear fits
    used to extract ΔT and kappa.

    What to look for:
      - Converged profile: later cycles (darker colour) should overlap with
        the cumulative average, indicating steady state.
      - Clear linear regions on each side of the GB — curved profiles suggest
        the system hasn't equilibrated or the box is too short.
      - Visible discontinuity at x_GB: if there's no step, TBR is very small
        or the GB was not preserved (check RDF and atom positions).
    """
    n_cycles = len(temps_times)
    cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, n_cycles + 1)[:, None]

    cmap = cm.Oranges
    norm = Normalize(vmin=0, vmax=n_cycles)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.35)
    fig.suptitle(f"{label} — run {run_index}", fontsize=12)

    # Panel 1: Per-cycle temperature profiles
    for i, cycle_temps in enumerate(temps_times):
        axes[0].plot(bin_centers, cycle_temps, marker="o", markersize=2,
                     linewidth=0.8, color=cmap(norm(i)), alpha=0.7)
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Per-cycle temperature profiles (light→dark = early→late)")
    axes[0].axvline(bin_centers[cold_bin], color="blue", linestyle="--",
                    linewidth=0.8, label="cold bin")
    axes[0].axvline(bin_centers[hot_bin], color="red", linestyle="--",
                    linewidth=0.8, label="hot bin")
    axes[0].legend(fontsize=8)

    # Panel 2: Cumulative average + linear fits
    for i, avg in enumerate(cumulative_avg):
        axes[1].plot(bin_centers, avg, marker="o", markersize=2,
                     linewidth=0.8, color=cmap(norm(i)))

    # Overlay final linear fits
    left_fit = result["left_fit"]
    right_fit = result["right_fit"]
    gb_bin = nbins // 2
    margin = 1

    x_left = bin_centers[cold_bin + margin : gb_bin]
    x_right = bin_centers[gb_bin : hot_bin - margin]
    axes[1].plot(x_left, np.polyval(left_fit, x_left), color="blue",
                 linewidth=2, linestyle="--", label="left bulk fit")
    axes[1].plot(x_right, np.polyval(right_fit, x_right), color="red",
                 linewidth=2, linestyle="--", label="right bulk fit")

    # Mark ΔT at GB
    x_gb = bin_centers[gb_bin]
    T_l = np.polyval(left_fit, x_gb)
    T_r = np.polyval(right_fit, x_gb)
    axes[1].annotate(
        f"ΔT = {result['delta_T']:.1f} K",
        xy=(x_gb, (T_l + T_r) / 2), fontsize=9,
        arrowprops=dict(arrowstyle="->"), xytext=(x_gb + 5, (T_l + T_r) / 2 + 20),
    )
    axes[1].axvline(x_gb, color="green", linestyle=":", linewidth=0.8, label="GB plane")
    axes[1].set_ylabel("Cumulative avg T [K]")
    axes[1].set_title(
        f"Converged profile — κ = {result['kappa_SI']:.2f} W/(m·K), "
        f"R_K = {result['R_K_SI']:.3e} K·m²/W"
    )
    axes[1].legend(fontsize=8)

    # Panel 3: Steady-state convergence
    window = max(int(n_cycles * 0.25), 1)
    cycle_indices = np.arange(n_cycles)
    per_cycle_mean_T = np.mean(temps_times, axis=1)  # mean T across bins per cycle
    axes[2].plot(cycle_indices, per_cycle_mean_T, color="tomato", linewidth=0.8)
    axes[2].axhline(np.mean(per_cycle_mean_T[-window:]), color="steelblue",
                    linestyle="--", linewidth=1.5, label=f"last {window} cycle avg")
    conv_str = "CONVERGED" if converged else f"NOT converged (max dev = {max_dev:.1f} K)"
    axes[2].set_title(f"Steady-state check: {conv_str}", fontsize=10)
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("Mean bin T [K]")
    axes[2].legend(fontsize=8)

    plt.savefig(os.path.join(out_dir, "temperature_profile.png"), dpi=150)
    plt.close()


def plot_temperature_profile_animated(temps_times, bin_centers, out_dir,
                                       label, run_index, cold_bin, hot_bin, nbins):
    """
    Animate the per-cycle temperature profile as a movie: one frame per cycle,
    showing the current cycle's bin temperatures as a single line plus the
    left/right bulk linear fits recomputed from the cumulative average up to
    that cycle (so you can watch the fits stabilize toward steady state).
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    n_cycles = len(temps_times)
    T_min = temps_times.min() * 0.98
    T_max = temps_times.max() * 1.02

    # Precompute cumulative averages and fits for every frame.
    margin = 1
    gb_bin = nbins // 2
    x_left = bin_centers[cold_bin + margin : gb_bin]
    x_right = bin_centers[gb_bin : hot_bin - margin]
    cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, n_cycles + 1)[:, None]
    left_fits  = [np.polyfit(x_left,  cumulative_avg[i, cold_bin + margin : gb_bin],  1) for i in range(n_cycles)]
    right_fits = [np.polyfit(x_right, cumulative_avg[i, gb_bin : hot_bin - margin], 1) for i in range(n_cycles)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(bin_centers[0], bin_centers[-1])
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel("Position [Å]")
    ax.set_ylabel("Temperature [K]")
    ax.axvline(bin_centers[cold_bin], color="blue",  linestyle="--", linewidth=0.8, label="cold bin")
    ax.axvline(bin_centers[hot_bin],  color="red",   linestyle="--", linewidth=0.8, label="hot bin")
    ax.axvline(bin_centers[gb_bin],   color="green", linestyle=":",  linewidth=0.8, label="GB plane")
    ax.legend(fontsize=8)

    line,      = ax.plot([], [], marker="o", markersize=3, linewidth=1.2, color="darkorange", label="cycle T")
    fit_left,  = ax.plot([], [], color="blue", linewidth=1.8, linestyle="--")
    fit_right, = ax.plot([], [], color="red",  linewidth=1.8, linestyle="--")
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        fit_left.set_data([], [])
        fit_right.set_data([], [])
        title.set_text("")
        return line, fit_left, fit_right, title

    def update(frame):
        line.set_data(bin_centers, temps_times[frame])
        fit_left.set_data(x_left,   np.polyval(left_fits[frame],  x_left))
        fit_right.set_data(x_right, np.polyval(right_fits[frame], x_right))
        title.set_text(f"{label} — run {run_index} — cycle {frame + 1}/{n_cycles}")
        return line, fit_left, fit_right, title

    anim = FuncAnimation(fig, update, frames=n_cycles, init_func=init,
                         blit=True, interval=50)

    out_path = os.path.join(out_dir, "temperature_profile.mp4")
    try:
        writer = FFMpegWriter(fps=20, bitrate=1800)
        anim.save(out_path, writer=writer)
        print(f"    Animation saved to {out_path}")
    except Exception as e:
        out_path = out_path.replace(".mp4", ".gif")
        anim.save(out_path, writer="pillow", fps=20)
        print(f"    Animation saved to {out_path} (mp4 unavailable: {e})")

    plt.close()
