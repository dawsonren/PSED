"""
rnemd_plots.py — Diagnostic plotting utilities for rNEMD simulations.

Functions
---------
plot_temperature_profile
    Two-panel static diagnostic: per-cycle profiles and converged cumulative
    average with linear fits.
plot_energy_diagnostics
    Three-panel energy diagnostic (DEBUG_DIAGNOSTICS only): potential energy,
    mean temperature (steady-state check), and total energy conservation.
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
                              label, run_index,
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

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
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

    # Overlay final linear fits (primary + periodic duplicates)
    left_fit     = result["left_fit"]
    right_fit    = result["right_fit"]
    cold_dup_fit = result["cold_dup_fit"]
    hot_dup_fit  = result["hot_dup_fit"]
    gb_bin = nbins // 2
    margin = 1

    x_left     = bin_centers[cold_bin + margin : gb_bin]
    x_right    = bin_centers[gb_bin : hot_bin - margin]
    x_cold_dup = bin_centers[margin : cold_bin - margin]
    x_hot_dup  = bin_centers[hot_bin + margin : nbins - margin]

    axes[1].plot(x_left,     np.polyval(left_fit,     x_left),     color="blue",
                 linewidth=2, linestyle="--", label="cold grain fit (cold→GB)")
    axes[1].plot(x_right,    np.polyval(right_fit,    x_right),    color="red",
                 linewidth=2, linestyle="--", label="hot grain fit (GB→hot)")
    axes[1].plot(x_cold_dup, np.polyval(cold_dup_fit, x_cold_dup), color="cornflowerblue",
                 linewidth=2, linestyle=":",  label="cold grain dup fit (start→cold)")
    axes[1].plot(x_hot_dup,  np.polyval(hot_dup_fit,  x_hot_dup),  color="salmon",
                 linewidth=2, linestyle=":",  label="hot grain dup fit (hot→end)")

    # Mark ΔT at primary GB
    x_gb = bin_centers[gb_bin]
    T_l = np.polyval(left_fit, x_gb)
    T_r = np.polyval(right_fit, x_gb)
    axes[1].annotate(
        f"ΔT = {result['delta_T']:.1f} K",
        xy=(x_gb, (T_l + T_r) / 2), fontsize=9,
        arrowprops=dict(arrowstyle="->"), xytext=(x_gb + 5, (T_l + T_r) / 2 + 20),
    )
    axes[1].axvline(x_gb, color="green", linestyle=":", linewidth=0.8, label="GB plane")

    # Mark ΔT at duplicate GB (periodic boundary: x=0 / x=box_length)
    bin_width = bin_centers[1] - bin_centers[0]
    box_length = bin_centers[-1] + bin_width / 2.0
    T_cold_dup_at_0 = np.polyval(cold_dup_fit, 0.0)
    T_hot_dup_at_L  = np.polyval(hot_dup_fit, box_length)
    delta_T_dup = abs(T_cold_dup_at_0 - T_hot_dup_at_L)
    mid_T_dup = (T_cold_dup_at_0 + T_hot_dup_at_L) / 2.0
    axes[1].axvline(bin_centers[0], color="purple", linestyle=":", linewidth=0.8,
                    label="dup GB plane (periodic boundary)")
    axes[1].annotate(
        f"ΔT_dup = {delta_T_dup:.1f} K",
        xy=(bin_centers[0], mid_T_dup), fontsize=9,
        arrowprops=dict(arrowstyle="->"),
        xytext=(bin_centers[0] + 10, mid_T_dup + 20),
    )
    axes[1].set_xlabel("Position [Å]")
    axes[1].set_ylabel("Cumulative avg T [K]")
    axes[1].set_title(
        f"Converged profile — κ = {result['kappa_SI']:.2f} W/(m·K) "
        f"[cold: {result['kappa_cold_SI']:.2f}, hot: {result['kappa_hot_SI']:.2f}], "
        f"R_K = {result['R_K_SI']:.3e} K·m²/W"
    )
    axes[1].legend(fontsize=8)

    plt.savefig(os.path.join(out_dir, "temperature_profile.png"), dpi=150)
    plt.close()


def plot_energy_diagnostics(temps_times, ke_per_cycle, pe_per_cycle, n_atoms, out_dir,
                             label, run_index, converged, max_dev):
    """
    Three-panel energy diagnostic for NVE rNEMD simulations.

    Panel 1: Potential energy per cycle (from dump_thermo, column U).
    Panel 2: Mean system temperature per cycle (steady-state convergence check).
    Panel 3: Total energy (KE + PE) per cycle (from dump_thermo, columns K and U).

    Physical background
    -------------------
    The Müller-Plathe velocity swap exchanges momenta between two atoms, which
    conserves total kinetic energy (Σ½mv² is unchanged system-wide).  Combined
    with NVE MD — which conserves total energy between swaps — the whole-run
    total energy should therefore be approximately constant.  Any systematic
    drift in Panel 3 is a warning sign that the timestep is too large or there
    is a numerical instability, not an expected artefact of the method.

    What to look for
    ----------------
    - PE (Panel 1): should stabilise to a roughly flat trend once the
      temperature gradient has established.  A monotone drift suggests
      the integration timestep is too large.
    - Temperature (Panel 2): should plateau in steady state.  Convergence
      status (from check_steady_state) is shown in the title.
    - Total energy (Panel 3): should be approximately flat.  Drift beyond
      ~0.1 eV/atom over the full run warrants reducing the timestep.
    """
    n_cycles = len(temps_times)
    cycle_indices = np.arange(n_cycles)
    per_cycle_mean_T = np.mean(temps_times, axis=1)   # K, shape (n_cycles,)

    total_energy = ke_per_cycle + pe_per_cycle  # eV

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(f"{label} — run {run_index} — energy diagnostics (NVE)", fontsize=12)

    # Panel 1: Potential energy
    axes[0].plot(cycle_indices, pe_per_cycle, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Potential energy [eV]")
    axes[0].set_title("Potential energy per cycle")
    axes[0].set_xlabel("Cycle")

    # Panel 2: Mean temperature — steady-state convergence check
    window = max(int(n_cycles * 0.25), 1)
    axes[1].plot(cycle_indices, per_cycle_mean_T, color="tomato", linewidth=0.8)
    axes[1].axhline(np.mean(per_cycle_mean_T[-window:]), color="steelblue",
                    linestyle="--", linewidth=1.5, label=f"last {window} cycle avg")
    conv_str = "CONVERGED" if converged else f"NOT converged (max dev = {max_dev:.1f} K)"
    axes[1].set_title(f"Steady-state check: {conv_str}", fontsize=10)
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Mean bin T [K]")
    axes[1].legend(fontsize=8)

    # Panel 3: Total energy
    axes[2].plot(cycle_indices, total_energy, color="darkorange", linewidth=0.8)
    axes[2].set_ylabel("Total energy (KE + PE) [eV]")
    axes[2].set_title("Total energy per cycle — should be flat for well-behaved NVE")
    axes[2].set_xlabel("Cycle")

    plt.savefig(os.path.join(out_dir, "energy_diagnostics.png"), dpi=150)
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
    x_left     = bin_centers[cold_bin + margin : gb_bin]
    x_right    = bin_centers[gb_bin : hot_bin - margin]
    x_cold_dup = bin_centers[margin : cold_bin - margin]
    x_hot_dup  = bin_centers[hot_bin + margin : nbins - margin]
    cumulative_avg = np.cumsum(temps_times, axis=0) / np.arange(1, n_cycles + 1)[:, None]
    left_fits      = [np.polyfit(x_left,     cumulative_avg[i, cold_bin + margin : gb_bin],      1) for i in range(n_cycles)]
    right_fits     = [np.polyfit(x_right,    cumulative_avg[i, gb_bin : hot_bin - margin],        1) for i in range(n_cycles)]
    cold_dup_fits  = [np.polyfit(x_cold_dup, cumulative_avg[i, margin : cold_bin - margin],       1) for i in range(n_cycles)]
    hot_dup_fits   = [np.polyfit(x_hot_dup,  cumulative_avg[i, hot_bin + margin : nbins - margin], 1) for i in range(n_cycles)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(bin_centers[0], bin_centers[-1])
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel("Position [Å]")
    ax.set_ylabel("Temperature [K]")
    ax.axvline(bin_centers[cold_bin], color="blue",  linestyle="--", linewidth=0.8, label="cold bin")
    ax.axvline(bin_centers[hot_bin],  color="red",   linestyle="--", linewidth=0.8, label="hot bin")
    ax.axvline(bin_centers[gb_bin],   color="green", linestyle=":",  linewidth=0.8, label="GB plane")
    ax.legend(fontsize=8)

    line,          = ax.plot([], [], marker="o", markersize=3, linewidth=1.2, color="darkorange", label="cycle T")
    fit_left,      = ax.plot([], [], color="blue",          linewidth=1.8, linestyle="--")
    fit_right,     = ax.plot([], [], color="red",           linewidth=1.8, linestyle="--")
    fit_cold_dup,  = ax.plot([], [], color="cornflowerblue", linewidth=1.8, linestyle=":")
    fit_hot_dup,   = ax.plot([], [], color="salmon",        linewidth=1.8, linestyle=":")
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        fit_left.set_data([], [])
        fit_right.set_data([], [])
        fit_cold_dup.set_data([], [])
        fit_hot_dup.set_data([], [])
        title.set_text("")
        return line, fit_left, fit_right, fit_cold_dup, fit_hot_dup, title

    def update(frame):
        line.set_data(bin_centers, temps_times[frame])
        fit_left.set_data(x_left,     np.polyval(left_fits[frame],     x_left))
        fit_right.set_data(x_right,   np.polyval(right_fits[frame],    x_right))
        fit_cold_dup.set_data(x_cold_dup, np.polyval(cold_dup_fits[frame], x_cold_dup))
        fit_hot_dup.set_data(x_hot_dup,   np.polyval(hot_dup_fits[frame],  x_hot_dup))
        title.set_text(f"{label} — run {run_index} — cycle {frame + 1}/{n_cycles}")
        return line, fit_left, fit_right, fit_cold_dup, fit_hot_dup, title

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
