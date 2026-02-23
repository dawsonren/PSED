"""
Statistical utilities for rNEMD multi-run analysis.

Functions:
  check_steady_state   — compare late vs early cycle temperature profiles
  aggregate_run_results — mean +/- std of kappa and R_K across independent runs
  format_result_summary — pretty-print for terminal output
"""

import numpy as np


def check_steady_state(temps_times, window_fraction=0.25, tolerance_K=5.0):
    """
    Check whether the temperature profile has converged to steady state.

    Compares the mean profile over the last *window_fraction* of cycles to
    the mean over the preceding window.  If the max absolute bin-wise
    difference is below *tolerance_K*, the run is considered converged.

    Parameters
    ----------
    temps_times : ndarray, shape (n_cycles, nbins)
        Per-cycle bin temperatures from the production phase.
    window_fraction : float
        Fraction of total cycles used for each comparison window.
    tolerance_K : float
        Maximum acceptable deviation (K) between the two windows.

    Returns
    -------
    converged : bool
    max_deviation : float
        Largest bin-wise temperature difference between windows (K).
    details : dict
        Diagnostic info (window indices, per-window averages).
    """
    n = len(temps_times)
    window = max(int(n * window_fraction), 1)

    if n < 2 * window:
        return False, np.nan, {"reason": "not enough cycles for two windows"}

    late_avg = np.mean(temps_times[-window:], axis=0)
    early_avg = np.mean(temps_times[-2 * window : -window], axis=0)

    max_dev = float(np.max(np.abs(late_avg - early_avg)))
    converged = max_dev < tolerance_K

    return converged, max_dev, {
        "late_window": (n - window, n),
        "early_window": (n - 2 * window, n - window),
        "late_avg": late_avg,
        "early_avg": early_avg,
    }


def aggregate_run_results(run_results):
    """
    Compute mean +/- std of kappa and R_K across independent runs.

    Parameters
    ----------
    run_results : list[dict]
        One dict per run, each containing at least
        'kappa_SI', 'R_K_SI', 'J_SI'.

    Returns
    -------
    dict with keys: kappa_mean, kappa_std, R_K_mean, R_K_std,
                    J_mean, J_std, n_runs.
    """
    kappas = np.array([r["kappa_SI"] for r in run_results])
    R_Ks = np.array([r["R_K_SI"] for r in run_results])
    Js = np.array([r["J_SI"] for r in run_results])

    n = len(run_results)
    ddof = 1 if n > 1 else 0

    return {
        "kappa_mean": float(np.nanmean(kappas)),
        "kappa_std": float(np.nanstd(kappas, ddof=ddof)),
        "R_K_mean": float(np.nanmean(R_Ks)),
        "R_K_std": float(np.nanstd(R_Ks, ddof=ddof)),
        "J_mean": float(np.nanmean(Js)),
        "J_std": float(np.nanstd(Js, ddof=ddof)),
        "n_runs": n,
    }


def format_result_summary(agg, gb_label):
    """Return a formatted string summarising aggregated rNEMD results."""
    lines = [
        f"{'=' * 60}",
        f"  {gb_label} -- {agg['n_runs']} independent runs",
        f"  kappa = {agg['kappa_mean']:.2f} +/- {agg['kappa_std']:.2f} W/(m*K)",
        f"  R_K   = {agg['R_K_mean']:.3e} +/- {agg['R_K_std']:.3e} K*m^2/W",
        f"  J     = {agg['J_mean']:.3e} +/- {agg['J_std']:.3e} W/m^2",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)
