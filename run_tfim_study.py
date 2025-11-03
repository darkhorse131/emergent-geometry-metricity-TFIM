# run_tfim_study.py
# Campaign runner for the TFIM exact solver (JW + BdG, OBC).
# Production‑aligned with the PRB Rapid plan:
#   • Exact TFIM (JW+BdG) → Gaussian two‑mode MI (nats)
#   • Site-averaged I(r) with optional bulk trimming (α)
#   • d_E(r) = I(r)^(-1/2); MI‑floor policy for diagnostics (+ explicit mask)
#   • Robust fits for X with windowing + method record (WLS/OLS/Theil–Sen)
#   • Records pair stats (N_r, sigma_r_jk) when available
#   • Triangle‑defect Δ(r,r) and FSS payload d_E/L vs r/L
#   • Energy QA: E0/L recorded (reference −4/π at h = J)
#
# Notes on backward compatibility and minimal changes:
#   - Prefer pair-level MI (tfim_exact.compute_pair_mi_by_r) → aggregate + jackknife (mi_tools).
#   - Fallbacks:
#       (a) tfim_exact.calculate_averaged_I_r(..., bulk_trim_fraction=α, return_pair_stats=True)
#           may return (r, I, N_r, sigma_r_jk); if unsupported, fall back to (r, I) only.
#   - Fit defaults preserve prior behavior (OLS) unless overridden in the config.
#   - Output JSON preserves previous keys and adds optional fields:
#       N_r_values, sigma_r_jk, mask_floor, fss_scaled, plus richer 'fit' metadata.

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import math  # <-- added

import tfim_exact
import mi_tools  # ensure this is importable (same folder / PYTHONPATH)

# --- added: BCFT chord-distance helper ---
def _chord_distance(r_array, L):
    r = np.asarray(r_array, dtype=float)
    return (L / np.pi) * np.sin(np.pi * r / L)
# --- end added helper ---


# -----------------------
# JSON helpers
# -----------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            x = float(obj)
            if np.isnan(x):
                return "NaN"
            if np.isinf(x):
                return "Infinity" if x > 0 else "-Infinity"
            return x
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# -----------------------
# Internal utilities
# -----------------------
def _normalize_fit_method(name: str) -> str:
    """Map user-provided string to one of {'OLS','WLS','THEILSEN'}."""
    n = (name or "").strip().lower()
    if "theil" in n:
        return "THEILSEN"
    if "wls" in n:
        return "WLS"
    return "OLS"


def _try_pair_level_MI(
    L: int,
    C: np.ndarray,
    F: np.ndarray,
    r_max: int,
    bulk_trim_fraction: float
) -> Optional[Dict[int, List[float]]]:
    """
    Try to obtain per-pair JW two-mode MI grouped by r from tfim_exact if available.
    Expected signature:
        tfim_exact.compute_pair_mi_by_r(L, C, F, r_max, bulk_trim_fraction=alpha) -> {r: [I_ij]}
    Returns dict or None if unsupported.
    """
    fn = getattr(tfim_exact, "compute_pair_mi_by_r", None)
    if callable(fn):
        try:
            return fn(L, C, F, r_max, bulk_trim_fraction=bulk_trim_fraction)
        except TypeError:
            # Older signature without bulk_trim_fraction
            return fn(L, C, F, r_max)
        except Exception:
            # If anything goes wrong, fall back to averaged interface
            return None
    return None


def _compute_Ir_and_uncertainty(
    L: int,
    C: np.ndarray,
    F: np.ndarray,
    r_max: int,
    bulk_trim_fraction: float,
    mi_floor: float
) -> Tuple[List[int], List[float], Optional[List[int]], Optional[List[float]], Optional[List[bool]], bool]:
    """
    Compute (r, I(r)) and, if possible, N_r and jackknife std per r.
    Returns:
        r_values, I_r_values, N_r_values|None, sigma_r_jk|None, mask_floor|None, bulk_applied(bool)
    """
    # Preferred path: pair-level MI → average + jackknife via mi_tools
    pairs_by_r = _try_pair_level_MI(L, C, F, r_max, bulk_trim_fraction)
    if pairs_by_r is not None:
        agg = mi_tools.site_average_I_by_r_from_pairs(pairs_by_r, mi_floor=mi_floor)
        jk = mi_tools.jackknife_std_per_r(pairs_by_r, mi_floor=mi_floor)

        r_values = agg["r"].astype(int).tolist()
        I_r_values = agg["I_r"].astype(float).tolist()
        N_r_values = agg["N_r"].astype(int).tolist()
        sigma_r_jk = jk["std_jk"].astype(float).tolist()
        mask_floor = agg["mask_floor"].astype(bool).tolist()
        bulk_applied = bulk_trim_fraction > 0.0
        return r_values, I_r_values, N_r_values, sigma_r_jk, mask_floor, bulk_applied

    # Fallback: backend-averaged I(r); try to request pair stats if supported
    bulk_applied = False
    try:
        # Attempt new signature with pair stats
        r_values, I_r_values, N_r_values, sigma_r_jk = tfim_exact.calculate_averaged_I_r(
            L, C, F, r_max, bulk_trim_fraction=bulk_trim_fraction, return_pair_stats=True
        )
        bulk_applied = bulk_trim_fraction > 0.0
    except TypeError:
        # Older signature with optional bulk trimming only
        try:
            r_values, I_r_values = tfim_exact.calculate_averaged_I_r(
                L, C, F, r_max, bulk_trim_fraction=bulk_trim_fraction
            )
            bulk_applied = bulk_trim_fraction > 0.0
            N_r_values, sigma_r_jk = None, None
        except TypeError:
            # Legacy earliest signature
            r_values, I_r_values = tfim_exact.calculate_averaged_I_r(L, C, F, r_max)
            N_r_values, sigma_r_jk = None, None

    # Provide a floor mask for transparency
    mask_floor = [np.isfinite(v) and (v > mi_floor) for v in I_r_values]
    return r_values, I_r_values, N_r_values, sigma_r_jk, mask_floor, bulk_applied


def _fit_exponent(
    r_values: List[int],
    I_r_values: List[float],
    N_r_values: Optional[List[int]],
    sigma_r_jk: Optional[List[float]],
    rmin: int,
    rmax: int,
    mi_floor: float,
    method: str
) -> Tuple[Dict[str, float], str]:
    """
    Fit X in I(r) ~ r^{-X} using mi_tools fitter with windowing and (if available) weights.
    Weight priority for WLS: N_r (default) → sigma_r_jk → unweighted.
    Returns (fit_result_dict, weights_used_str).
    """
    weights_used = ""

    if method == "THEILSEN":
        return mi_tools.fit_powerlaw_exponent(
            r_values, I_r_values, rmin=rmin, rmax=rmax, mi_floor=mi_floor, method="THEILSEN"
        ), weights_used

    if method == "WLS":
        # Prefer N_r weights if present; otherwise use sigma_r_jk
        if (N_r_values is not None) and (np.sum(np.array(N_r_values) > 0) >= 2):
            weights_used = "N_r"
            return mi_tools.fit_powerlaw_exponent(
                r_values, I_r_values, N_r=N_r_values, rmin=rmin, rmax=rmax,
                mi_floor=mi_floor, method="WLS"
            ), weights_used
        if (sigma_r_jk is not None) and (np.sum(np.isfinite(sigma_r_jk)) >= 2):
            weights_used = "sigma_r_jk"
            return mi_tools.fit_powerlaw_exponent(
                r_values, I_r_values, sigma_r=sigma_r_jk, rmin=rmin, rmax=rmax,
                mi_floor=mi_floor, method="WLS"
            ), weights_used
        # Fall through to OLS if no usable weights
        method = "OLS"

    # OLS (default)
    return mi_tools.fit_powerlaw_exponent(
        r_values, I_r_values, rmin=rmin, rmax=rmax, mi_floor=mi_floor, method="OLS"
    ), weights_used


# -----------------------
# Main campaign driver
# -----------------------
def run_tfim_campaign(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    outdir = config["output_directory"]
    os.makedirs(outdir, exist_ok=True)

    J = float(config.get("J", 1.0))
    h_list = list(config["h_values_to_run"])
    L_list = list(config["L_values"])

    # Analysis parameters (defaults aligned to plan but minimally disruptive)
    analysis_params = dict(config.get("analysis_params", {}))
    r_max_factor = float(analysis_params.get("r_max_factor", 0.5))
    fit_r_min = int(analysis_params.get("fit_r_min", 10))
    fit_r_max_factor = float(analysis_params.get("fit_r_max_factor", 0.8))
    mi_floor = float(analysis_params.get("mi_floor", getattr(mi_tools, "DEFAULT_MI_FLOOR", 1e-12)))
    bulk_trim_fraction = float(analysis_params.get("bulk_trim_fraction", 0.0))
    fit_method = _normalize_fit_method(str(analysis_params.get("fit_method", "ols")))

    print("===== TFIM EXACT (OBC) — PRODUCTION RUNNER =====")
    print(f"Output dir: {outdir}")
    print(f"J = {J}, h values = {h_list}, L values = {L_list}")
    print(f"[analysis] r_max_factor={r_max_factor}, fit_r_min={fit_r_min}, "
          f"fit_r_max_factor={fit_r_max_factor}, mi_floor={mi_floor}, "
          f"bulk_trim_fraction={bulk_trim_fraction}, fit_method={fit_method}")

    for h in h_list:
        for L in L_list:
            tag = f"L{L}_H{h:.6f}".replace(".", "p")
            outfile = os.path.join(outdir, f"results_{tag}.json")
            if os.path.exists(outfile):
                print(f"[skip] {outfile} exists.")
                continue

            print(f"\n--- Run: L={L}, h={h} ---")
            t0 = time.time()

            # 1) Solve TFIM exactly (BdG): correlators and ground-state energy
            C, F, E0, evals = tfim_exact.solve_bdg(L, h, J)

            # 2) I(r): site-averaged MI in nats (prefer pair-level → average; else legacy averaged)
            r_max = max(1, int(L * r_max_factor))
            r_values, I_r_values, N_r_values, sigma_r_jk, mask_floor, bulk_applied = _compute_Ir_and_uncertainty(
                L, C, F, r_max, bulk_trim_fraction, mi_floor
            )

            # 3) d_E(r) with MI-floor policy (points with I <= floor or non-finite → NaN)
            d_E_r_values = mi_tools.calculate_entanglement_distance_d_E_r(r_values, I_r_values, mi_floor=mi_floor)

            # 4) Fit X on a reproducible window (r >= fit_r_min, r <= r_upper), honoring the floor
            r_upper = max(fit_r_min, int(r_max * fit_r_max_factor))
            # Respect an absolute cap on r for the fit window (to match FSS note)
            cap = analysis_params.get("fit_r_max_cap", None)
            if cap is not None:
                r_upper = min(r_upper, int(cap))
            fit_result, weights_used = _fit_exponent(
                r_values=r_values,
                I_r_values=I_r_values,
                N_r_values=N_r_values,
                sigma_r_jk=sigma_r_jk,
                rmin=fit_r_min,
                rmax=r_upper,
                mi_floor=mi_floor,
                method=fit_method
            )
            X = float(fit_result.get("X", np.nan))
            X_err = float(fit_result.get("X_stderr", np.nan))
            X_lo = float(fit_result.get("X_lo", np.nan))
            X_hi = float(fit_result.get("X_hi", np.nan))
            R2_I = float(fit_result.get("R2", np.nan))
            n_used = int(fit_result.get("n_used", 0))

            # --- Additional fits saved for clarity & reproducibility ---
            fit_chord = {}
            fit_exponential = {}

            # (A) At criticality (h≈J): BCFT chord-distance refit of the SAME r-window
            if abs(h - J) < 1e-12:
                # restrict to the r-window used above, then map r -> rho
                r_arr = np.asarray(r_values, dtype=float)
                I_arr = np.asarray(I_r_values, dtype=float)
                in_win = (r_arr >= float(fit_r_min)) & (r_arr <= float(r_upper))
                r_fit = r_arr[in_win]
                I_fit = I_arr[in_win]
                Nr_fit = (np.asarray(N_r_values, dtype=float)[in_win]
                          if N_r_values else None)
                sig_fit = (np.asarray(sigma_r_jk, dtype=float)[in_win]
                           if sigma_r_jk else None)
                rho_fit = _chord_distance(r_fit, L)

                # Weighted choice mirrors the r-based fit
                chord_res = mi_tools.fit_powerlaw_exponent(
                    rho_fit, I_fit,
                    N_r=Nr_fit if (fit_method == "WLS" and Nr_fit is not None) else None,
                    sigma_r=sig_fit if (fit_method == "WLS" and Nr_fit is None and sig_fit is not None) else None,
                    mi_floor=mi_floor,
                    method=("WLS" if fit_method == "WLS" and (Nr_fit is not None or sig_fit is not None) else "OLS")
                )
                fit_chord = {
                    "variable": "chord_distance_rho",
                    "definition": "rho = (L/pi) * sin(pi*r/L)",
                    "beta_X": float(chord_res["X"]),
                    "beta_X_err": float(chord_res.get("X_stderr", np.nan)),
                    "beta_X_lo": float(chord_res.get("X_lo", np.nan)),
                    "beta_X_hi": float(chord_res.get("X_hi", np.nan)),
                    "R2_I": float(chord_res.get("R2", np.nan)),
                    "window_r": [int(fit_r_min), int(r_upper)],
                    "r_points_used": int(chord_res.get("n_used", 0))
                }

            # (B) Off criticality (h != J): exponential correlation length
            else:
                exp_res = mi_tools.fit_exponential_corr_length(
                    r_values, I_r_values,
                    N_r=N_r_values if (fit_method == "WLS" and N_r_values) else None,
                    sigma_r=sigma_r_jk if (fit_method == "WLS" and (not N_r_values) and sigma_r_jk) else None,
                    rmin=fit_r_min, rmax=r_upper, mi_floor=mi_floor,
                    method=("WLS" if fit_method == "WLS" else "OLS")
                )
                fit_exponential = {
                    "model": "logI ~ -r/xi",
                    "xi": float(exp_res["xi"]),
                    "xi_err": float(exp_res.get("xi_stderr", np.nan)),
                    "xi_lo": float(exp_res.get("xi_lo", np.nan)),
                    "xi_hi": float(exp_res.get("xi_hi", np.nan)),
                    "R2_logI_vs_r": float(exp_res.get("R2", np.nan)),
                    "window_r": [int(fit_r_min), int(r_upper)],
                    "r_points_used": int(exp_res.get("n_used", 0))
                }

            # 5) Triangle-defect Δ(r,r) via helper (propagates NaNs where floor masked)
            tri = mi_tools.triangle_defect_rr(r_values, d_E_r_values)
            delta_r_vals = tri["r"].astype(int).tolist()
            delta_rr_vals = tri["delta_rr"].astype(float).tolist()

            # 6) Energy QA: record E/L and reference value at h=J
            energy_per_site = float(E0) / float(L)
            energy_validation = {
                "value_E_over_L": energy_per_site,
                "reference_at_h_equals_J": float(-4.0 / np.pi),  # <-- updated reference
                "applicable": bool(abs(h - J) < 1e-12)
            }

            # 7) Optional: scaled curves for FSS plotting (stored for convenience)
            scaled = mi_tools.scaled_dE_over_L(r_values, d_E_r_values, L)
            x_scaled = scaled["x"].astype(float).tolist()
            y_scaled = scaled["y"].astype(float).tolist()

            # 8) Save JSON (preserving old keys; adding richer metadata)
            data = {
                "model": config.get("model", "TFIM_EXACT_OBC"),
                "basis_MI": "JW_fermionic_two_mode",
                "entropy_units": "nats",
                "J": J,
                "h": h,
                "L": L,
                "energy_per_site": energy_per_site,
                "energy_validation": energy_validation,
                "eigvals_first10": [float(x) for x in sorted(np.asarray(evals).ravel())[:10]],
                "r_values": r_values,
                "I_r_values": I_r_values,
                "N_r_values": (N_r_values if N_r_values is not None else []),
                "sigma_r_jk": (sigma_r_jk if sigma_r_jk is not None else []),
                "mask_floor": (mask_floor if mask_floor is not None else []),
                "d_E_r_values": d_E_r_values,
                "delta_rr": {
                    "r_values": delta_r_vals,
                    "values": delta_rr_vals,
                    "mi_floor_used": mi_floor
                },
                "fss_scaled": {
                    "x_r_over_L": x_scaled,
                    "y_dE_over_L": y_scaled
                },
                "fit": {
                    "method": fit_method,        # "OLS", "WLS" or "THEILSEN"
                    "weights": weights_used,     # "", "N_r", or "sigma_r_jk"
                    "fit_r_min": fit_r_min,
                    "fit_r_upper": r_upper,
                    "r_points_used": n_used,
                    "beta_X": X,                 # keep legacy key name
                    "beta_X_err": X_err,
                    "beta_X_lo": X_lo,
                    "beta_X_hi": X_hi,
                    "R2_I": R2_I,
                    "mi_floor_used": mi_floor,
                    "kind": ("powerlaw_r" if abs(h - J) < 1e-12 else "diagnostic_powerlaw_r")
                },
                # SCHEMA FIX: use canonical key without stray space so postprocessor finds it.
                "fit_chord": fit_chord,                 # present and populated only at criticality
                "fit_exponential": fit_exponential,     # present and populated only off criticality
                "analysis_params": {
                    "r_max_factor": r_max_factor,
                    "fit_r_min": fit_r_min,
                    "fit_r_max_factor": fit_r_max_factor,
                    "fit_r_max_cap": cap,  # absolute cap recorded for transparency
                    "mi_floor": mi_floor,
                    "bulk_trim_fraction_requested": bulk_trim_fraction,
                    "bulk_trim_applied": bool(bulk_applied),
                    "bulk_trim_fraction_applied": (bulk_trim_fraction if bulk_applied else 0.0),
                },
            }
            with open(outfile, "w") as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)

            dt = time.time() - t0
            X_print = X if np.isfinite(X) else np.nan
            wls_str = f", weights={weights_used}" if fit_method == "WLS" and weights_used else ""
            print(f"--- Finished in {dt:.2f}s | E/L={energy_per_site:.8f} | X≈{X_print} "
                  f"| fit_points={n_used} | method={fit_method}{wls_str} "
                  f"| bulk_trim_applied={bulk_applied}")

    print("\nAll requested runs complete.")


if __name__ == "__main__":
    # Try canonical and legacy config names to be robust
    cfg_candidates = ["config_TFIM_exact.json", "config_tfim_exact.json"]
    cfg = next((c for c in cfg_candidates if os.path.exists(c)), None)
    if cfg is not None:
        run_tfim_campaign(cfg)
    else:
        print(f"Config file not found. Tried: {cfg_candidates}")