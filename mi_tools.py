"""
mi_tools.py — Tools for calculating Mutual Information (MI), Emergent Distance (d_E),
and related fitting/diagnostic quantities.

This module has two roles:
  (i)  Legacy MPS utilities (quimb-based) for MI/entropy, central charge, etc.  (lazy import)
  (ii) Model‑agnostic post‑processing used by the TFIM JW+BdG pipeline:
       d_E(r) = I(r)^(-1/2), robust exponent fits, triangle‑defect diagnostics, etc. (entropies in nats)

Minimal PRB‑alignment changes in this "production-ready" version:
  • Keep quimb imports LAZY (no top‑level dependency) to keep TFIM pipeline quimb‑free.
  • Add MI‑floor handling (default 1e‑12 nats) for d_E and fits (policy: exclude below‑floor values).
  • Add model‑agnostic helpers:
        - site‑averaging from per‑pair MI with bulk‑trim option;
        - jackknife std for I(r);
        - robust exponent/ξ fits with windows & (approx.) CIs (WLS/OLS/Theil–Sen);
        - triangle‑defect Δ(r,r) and Δ(r1,r2) helpers;
        - simple FSS helper d_E/L vs r/L.
  • Preserve all existing function names & signatures (backward compatible). `fit_beta_from_I_r`
    now delegates to the new `fit_powerlaw_exponent` and returns β=X for compatibility.

Conceptual references:
  - Calibrated ansatz d_E = K0 / sqrt(I), Euclidean benchmark (X=2 ⇒ d_E=r), and independence
    of metricity from the numerical value of K0 (scale choice).   [oai_citation:0‡Emergent_Distance_Draft_21_SM.pdf](sediment://file_000000001e7461f79ce9dafd2bbbee1b)  [oai_citation:1‡Emergent_Distance_Draft_21.pdf](sediment://file_00000000995461f7a77da721a69c6860)
  - Metricity criterion: if I(r)=C r^{-X} with 0<X≤2 then d_E(r)∝r^{X/2} is a metric; exponential
    (or X>2) violates triangle inequality; use MI floor in diagnostics.                       [oai_citation:2‡Emergent_Distance_Draft_21_SM.pdf](sediment://file_000000001e7461f79ce9dafd2bbbee1b)
  - PRB Rapid “Contract”: we use d_E strictly as a global metricity diagnostic; all entropies in
    nats; bulk trimming & analysis policy (windows/CI/Δ).                                      [oai_citation:3‡Revision_Plan_v5.pdf](sediment://file_00000000f7ec61f7911f54bff7c27331)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from scipy.stats import linregress, theilslopes
try:
    # Student-t for confidence intervals (optional but available with SciPy)
    from scipy.stats import t as _student_t
except Exception:  # pragma: no cover - fallback if missing
    _student_t = None  # type: ignore

# -----------------------
# Global analysis defaults
# -----------------------
DEFAULT_MI_FLOOR = 1e-12  # in nats; points below this are excluded from fits/diagnostics


# --------------------------------------------------------------------------
# Helper: Von Neumann Entropy of a subsystem (legacy MPS path; lazy quimb)
# --------------------------------------------------------------------------
def _calculate_subsystem_entropy(psi_gs, sites_A_to_keep, num_lanczos_vecs: int = 20) -> float:
    """
    Calculate von Neumann entropy S(A) = -Tr(rho_A log rho_A) (nats) of a subsystem A from a
    pure-state MPS using quimb (lazy-imported). Returns np.nan if quimb unavailable or upon failure.

    Args:
        psi_gs: quimb.tensor.MatrixProductState (pure-state MPS).
        sites_A_to_keep: list[int] — indices of the subsystem.
        num_lanczos_vecs: number of Lanczos vectors for approx_spectral_function.

    Returns:
        float: S_A in nats (>=0), 0.0 if A empty, np.nan on failure.
    """
    try:
        import quimb
    except ImportError:
        # Quimb not available -> cannot compute; keep pipeline alive
        return np.nan

    L = psi_gs.L
    if not sites_A_to_keep:
        return 0.0
    if not all(isinstance(s, int) and 0 <= s < L for s in sites_A_to_keep):
        raise ValueError(f"All site indices in {sites_A_to_keep} must be integers within [0, {L-1}]")

    # --- 1) Construct reduced density matrix rho_A as a linear operator ---
    psi_ket_local = psi_gs.copy()
    psi_bra_local = psi_gs.H.copy()

    ket_phys_id_pattern = psi_ket_local.site_ind_id if psi_ket_local.site_ind_id else 'k{}'
    if ket_phys_id_pattern and len(ket_phys_id_pattern) > 1 and ket_phys_id_pattern[0].isalpha():
        bra_phys_id_pattern = f"b{ket_phys_id_pattern[1:]}"
    elif ket_phys_id_pattern:
        bra_phys_id_pattern = f"b_{ket_phys_id_pattern}"
    else:
        bra_phys_id_pattern = 'b{}'
    if ket_phys_id_pattern == bra_phys_id_pattern:
        bra_phys_id_pattern = f"bra_phys_{ket_phys_id_pattern}"

    idx_map_bra_phys_globally = {
        ket_phys_id_pattern.format(s): bra_phys_id_pattern.format(s)
        for s in range(L)
    }
    psi_bra_local.reindex(idx_map_bra_phys_globally, inplace=True)

    sites_to_trace_out = [s for s in range(L) if s not in sites_A_to_keep]
    trace_idx_map = {
        bra_phys_id_pattern.format(s): ket_phys_id_pattern.format(s)
        for s in sites_to_trace_out
    }
    psi_bra_local.reindex(trace_idx_map, inplace=True)

    rho_A_tn = (psi_bra_local & psi_ket_local)

    open_ket_indices = [ket_phys_id_pattern.format(s) for s in sites_A_to_keep]
    open_bra_indices = [bra_phys_id_pattern.format(s) for s in sites_A_to_keep]
    current_open_rho_A_inds = rho_A_tn.outer_inds()
    if not (all(idx in current_open_rho_A_inds for idx in open_ket_indices) and
            all(idx in current_open_rho_A_inds for idx in open_bra_indices)):
        if sites_A_to_keep and len(sites_A_to_keep) == L:
            pass
        elif sites_A_to_keep:
            return np.nan

    rho_A_lo = rho_A_tn.aslinearoperator(open_ket_indices, open_bra_indices)

    # --- 2) Spectral functional for S = -x log x (nats) via quimb.approx_spectral_function ---
    nlogn = lambda x_val: -x_val.real * np.log(x_val.real) if x_val.real > 1e-15 else 0.0
    dim_subsystem = 2 ** len(sites_A_to_keep)
    current_R = min(num_lanczos_vecs, dim_subsystem)
    if current_R <= 0:
        return 0.0

    try:
        raw_S_A = quimb.approx_spectral_function(rho_A_lo, f=nlogn, R=current_R)
        S_A = raw_S_A.real
    except Exception:
        return np.nan

    return S_A if S_A is not None else np.nan


# --------------------------------------------------------------------------------------
# Legacy MPS MI and convenience wrappers (kept for backward compatibility; lazy quimb)
# --------------------------------------------------------------------------------------
def calculate_mutual_information_I_r(psi_gs, L: int, r_max: int,
                                     num_lanczos_vecs: int = 20,
                                     bulk_trim_fraction: float = 0.0) -> Tuple[List[int], List[float]]:
    """
    Legacy MPS path (quimb): site‑averaged single‑site MI I(r) in nats.
    Adds optional bulk trimming to reduce OBC edge effects (default: off).

    Args:
        psi_gs: quimb.tensor.MatrixProductState
        L (int): system length
        r_max (int): max separation
        num_lanczos_vecs (int): spectral function Lanczos vectors
        bulk_trim_fraction (float in [0, 0.5)): trim αL sites from each end

    Returns:
        r_values (list[int]), avg_I_r_values (list[float])
    """
    alpha = float(bulk_trim_fraction) if bulk_trim_fraction is not None else 0.0
    alpha = max(0.0, min(alpha, 0.49))
    left = int(alpha * L)
    right_limit = L - left

    avg_I_r_values = []
    r_values = list(range(1, r_max + 1))
    for r_dist in r_values:
        I_at_this_r = []
        i_start = left
        i_stop = max(left, right_limit - r_dist)
        num_pairs = 0
        for i in range(i_start, i_stop):
            j = i + r_dist
            S_i = _calculate_subsystem_entropy(psi_gs, [i], num_lanczos_vecs)
            S_j = _calculate_subsystem_entropy(psi_gs, [j], num_lanczos_vecs)
            S_ij = _calculate_subsystem_entropy(psi_gs, [i, j], num_lanczos_vecs)
            if not (np.isfinite(S_i) and np.isfinite(S_j) and np.isfinite(S_ij)):
                continue
            mutual_info_ij = S_i + S_j - S_ij
            I_at_this_r.append(max(0.0, mutual_info_ij))  # clip tiny negatives
            num_pairs += 1
        avg_I_r_values.append(np.mean(I_at_this_r) if num_pairs > 0 else np.nan)
    return r_values, avg_I_r_values


def calculate_entanglement_distance_d_E_r(r_values: Sequence[int],
                                          avg_I_r_values: Sequence[float],
                                          mi_floor: float = DEFAULT_MI_FLOOR) -> List[float]:
    """
    Map I(r) -> d_E(r) = I(r)^(-1/2) with MI‑floor policy.
    Any I <= mi_floor or non‑finite is mapped to np.nan (excluded from diagnostics).

    Args:
        r_values: list[int]
        avg_I_r_values: list[float]
        mi_floor: floor in nats (default 1e-12)

    Returns:
        list[float]: d_E(r) with np.nan where I is too small/non‑finite.
    """
    d_E_r_values = []
    for I_r in avg_I_r_values:
        if not np.isfinite(I_r) or I_r <= mi_floor:
            d_E_r_values.append(np.nan)
        else:
            d_E_r_values.append(I_r ** (-0.5))
    return d_E_r_values


# -----------------------
# Model-agnostic helpers (NEW)
# -----------------------
def site_average_I_by_r_from_pairs(
    pairs_by_r: Mapping[int, Sequence[float]],
    mi_floor: float = DEFAULT_MI_FLOOR
) -> Dict[str, np.ndarray]:
    """
    Site-average MI across bulk pairs when per-pair values are provided.

    Args:
        pairs_by_r: dict {r: [I(i,i+r) values in nats]}. Pairs with I < mi_floor are excluded.
        mi_floor: MI floor in nats; applied at the PAIR level.

    Returns:
        dict with fields:
          - "r": sorted r values (np.ndarray[int])
          - "I_r": averaged I(r) (np.ndarray[float], np.nan if no surviving pairs)
          - "N_r": effective number of pairs used per r (np.ndarray[int])
          - "mask_floor": bool mask (True where I_r is finite and > mi_floor)
    """
    if not isinstance(pairs_by_r, Mapping):
        raise TypeError("pairs_by_r must be a mapping: {r: sequence of MI values}")

    r_sorted = sorted(k for k in pairs_by_r.keys() if isinstance(k, (int, np.integer)) and k > 0)
    I_r = []
    N_r = []
    for r in r_sorted:
        vals = np.asarray(pairs_by_r[r], dtype=float)
        if vals.size == 0:
            I_r.append(np.nan)
            N_r.append(0)
            continue
        # Apply pair-level MI floor and clip tiny negatives to zero
        vals = np.where(vals > mi_floor, vals, np.nan)
        vals = np.where(vals < 0.0, 0.0, vals)
        vals_finite = vals[np.isfinite(vals)]
        if vals_finite.size == 0:
            I_r.append(np.nan)
            N_r.append(0)
        else:
            I_r.append(float(np.mean(vals_finite)))
            N_r.append(int(vals_finite.size))

    I_r_arr = np.asarray(I_r, dtype=float)
    r_arr = np.asarray(r_sorted, dtype=int)
    N_r_arr = np.asarray(N_r, dtype=int)
    mask_floor = np.isfinite(I_r_arr) & (I_r_arr > mi_floor)
    return {"r": r_arr, "I_r": I_r_arr, "N_r": N_r_arr, "mask_floor": mask_floor}


def jackknife_std_per_r(pairs_by_r: Mapping[int, Sequence[float]],
                        mi_floor: float = DEFAULT_MI_FLOOR) -> Dict[str, np.ndarray]:
    """
    Jackknife std (delete-one) for averaged I(r). Excludes pairs with I < mi_floor.

    Returns:
        dict with fields:
          - "r": r values (sorted)
          - "I_r": jackknife mean (equals plain mean when finite)
          - "std_jk": jackknife std of the mean per r (np.nan if <2 pairs)
          - "N_r": number of pairs used per r
    """
    r_sorted = sorted(k for k in pairs_by_r.keys() if isinstance(k, (int, np.integer)) and k > 0)
    I_r_mean = []
    std_jk = []
    N_r = []

    for r in r_sorted:
        vals = np.asarray(pairs_by_r[r], dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > mi_floor]
        n = vals.size
        if n == 0:
            I_r_mean.append(np.nan)
            std_jk.append(np.nan)
            N_r.append(0)
            continue
        mu = float(np.mean(vals))
        I_r_mean.append(mu)
        if n < 2:
            std_jk.append(np.nan)
            N_r.append(n)
            continue
        # Delete-one jackknife of the mean
        jk_means = (n * mu - vals) / (n - 1)
        mu_bar = float(np.mean(jk_means))
        var_jk = (n - 1) * float(np.mean((jk_means - mu_bar) ** 2))
        std_jk.append(math.sqrt(var_jk))
        N_r.append(n)

    return {
        "r": np.asarray(r_sorted, dtype=int),
        "I_r": np.asarray(I_r_mean, dtype=float),
        "std_jk": np.asarray(std_jk, dtype=float),
        "N_r": np.asarray(N_r, dtype=int),
    }


# -----------------------
# Fitting utilities
# -----------------------
def _t_crit(alpha: float, df: float) -> float:
    """Two-sided t critical value; falls back to normal if scipy t not available."""
    if _student_t is None or not np.isfinite(df) or df <= 0:
        return 1.959963984540054  # ~N(0,1) 97.5% quantile
    try:
        return float(_student_t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        return 1.959963984540054


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
                         ) -> Tuple[float, float, float, float, float]:
    """
    Weighted linear fit y = a*x + b.
    Returns: slope a, intercept b, stderr_slope, R2 (weighted), df (effective dof).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        # Unweighted OLS via linregress
        lr = linregress(x, y)
        slope = float(lr.slope)
        intercept = float(lr.intercept)
        stderr = float(lr.stderr) if hasattr(lr, "stderr") and np.isfinite(lr.stderr) else np.nan
        R2 = float(lr.rvalue ** 2) if np.isfinite(lr.rvalue) else np.nan
        df = max(1.0, x.size - 2.0)
        return slope, intercept, stderr, R2, df

    w = np.asarray(w, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("All weights must be finite and > 0.")
    W = float(np.sum(w))
    x_bar = float(np.sum(w * x) / W)
    y_bar = float(np.sum(w * y) / W)
    Sxx = float(np.sum(w * (x - x_bar) ** 2))
    Sxy = float(np.sum(w * (x - x_bar) * (y - y_bar)))
    if Sxx <= 0:
        return np.nan, np.nan, np.nan, np.nan, 1.0

    slope = Sxy / Sxx
    intercept = y_bar - slope * x_bar

    y_hat = slope * x + intercept
    resid = y - y_hat
    RSS = float(np.sum(w * resid ** 2))
    TSS = float(np.sum(w * (y - y_bar) ** 2))
    R2 = 1.0 - RSS / TSS if TSS > 0 else (1.0 if RSS < 1e-15 else 0.0)

    # Effective sample size & DOF for WLS (approx.)
    n_eff = (W ** 2) / float(np.sum(w ** 2))
    df = max(1.0, n_eff - 2.0)
    sigma2 = RSS / max(1.0, (n_eff - 2.0))
    var_slope = sigma2 / Sxx
    stderr = math.sqrt(var_slope) if var_slope >= 0 else np.nan
    return float(slope), float(intercept), float(stderr), float(R2), float(df)


def fit_powerlaw_exponent(
    r_values: Sequence[Union[int, float]],
    I_r_values: Sequence[float],
    *,
    N_r: Optional[Sequence[int]] = None,
    sigma_r: Optional[Sequence[float]] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
    mi_floor: float = DEFAULT_MI_FLOOR,
    method: str = "WLS",         # "WLS", "OLS", or "THEILSEN"
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Fit I(r) ≈ C * r^{-X} on log–log axes and report X with CI and R^2.

    Args:
        r_values, I_r_values: sequences of r>0 and I(r) in nats.
        N_r: optional counts per r (used as WLS weights if sigma_r is None).
        sigma_r: optional std per r for WLS weights 1/sigma_r^2.
        rmin, rmax: fit window (inclusive).
        mi_floor: exclude points with I <= mi_floor or non-finite.
        method: "WLS" (default; uses sigma_r or N_r), "OLS", or "THEILSEN" (robust).
        alpha: two-sided CI level (default 0.05 ⇒ 95% CI).

    Returns:
        dict: {"X", "X_stderr", "X_lo", "X_hi", "R2", "rmin", "rmax", "n_used"}
    """
    r = np.asarray(r_values, dtype=float)
    I = np.asarray(I_r_values, dtype=float)
    mask = np.isfinite(r) & np.isfinite(I) & (r > 0) & (I > mi_floor)
    if rmin is not None:
        mask &= (r >= float(rmin))
    if rmax is not None:
        mask &= (r <= float(rmax))
    if not np.any(mask):
        return {"X": np.nan, "X_stderr": np.nan, "X_lo": np.nan, "X_hi": np.nan,
                "R2": np.nan, "rmin": float(rmin) if rmin is not None else np.nan,
                "rmax": float(rmax) if rmax is not None else np.nan, "n_used": 0.0}

    x = np.log(r[mask])
    y = np.log(I[mask])

    w = None
    df = None
    if method.upper() == "THEILSEN":
        # Robust slope on log–log; SciPy returns slope & its CI directly
        slope, intercept, lo_slope, hi_slope = theilslopes(y, x, alpha=1 - alpha)
        X = -float(slope)
        X_lo = -float(hi_slope)  # note: slope CI maps with sign flip
        X_hi = -float(lo_slope)
        # R2 surrogate: compute unweighted R2 of the median-slope fit
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res < 1e-15 else 0.0)
        return {"X": X, "X_stderr": np.nan, "X_lo": X_lo, "X_hi": X_hi,
                "R2": float(R2), "rmin": float(np.min(r[mask])), "rmax": float(np.max(r[mask])),
                "n_used": float(np.count_nonzero(mask))}
    else:
        if method.upper() == "WLS":
            if sigma_r is not None:
                sigma = np.asarray(sigma_r, dtype=float)[mask]
                w = 1.0 / (sigma ** 2)
                # remove any nonpositive weights (sigma<=0)
                keep = np.isfinite(w) & (w > 0)
                x, y, w = x[keep], y[keep], w[keep]
            elif N_r is not None:
                w = np.asarray(N_r, dtype=float)[mask]
                keep = np.isfinite(w) & (w > 0)
                x, y, w = x[keep], y[keep], w[keep]
        # else: OLS (w=None)

        slope, intercept, stderr, R2, df = _weighted_linear_fit(x, y, w)

        X = -slope
        X_stderr = float(stderr) if np.isfinite(stderr) else np.nan
        tcrit = _t_crit(alpha, df if df is not None else max(1.0, x.size - 2.0))
        X_lo = X - tcrit * X_stderr if np.isfinite(X_stderr) else np.nan
        X_hi = X + tcrit * X_stderr if np.isfinite(X_stderr) else np.nan

        return {"X": float(X), "X_stderr": float(X_stderr),
                "X_lo": float(X_lo), "X_hi": float(X_hi),
                "R2": float(R2),
                "rmin": float(np.min(np.exp(x))), "rmax": float(np.max(np.exp(x))),
                "n_used": float(x.size if w is None else np.sum(w > 0))}


def fit_exponential_corr_length(
    r_values: Sequence[Union[int, float]],
    I_r_values: Sequence[float],
    *,
    N_r: Optional[Sequence[int]] = None,
    sigma_r: Optional[Sequence[float]] = None,
    rmin: Optional[float] = None,
    rmax: Optional[float] = None,
    mi_floor: float = DEFAULT_MI_FLOOR,
    method: str = "WLS",
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Fit log I(r) ≈ log C - r/ξ on log–lin axes to estimate ξ with CI.

    Returns:
        dict: {"xi", "xi_stderr", "xi_lo", "xi_hi", "R2", "rmin", "rmax", "n_used"}
    """
    r = np.asarray(r_values, dtype=float)
    I = np.asarray(I_r_values, dtype=float)
    mask = np.isfinite(r) & np.isfinite(I) & (r > 0) & (I > mi_floor)
    if rmin is not None:
        mask &= (r >= float(rmin))
    if rmax is not None:
        mask &= (r <= float(rmax))
    if not np.any(mask):
        return {"xi": np.nan, "xi_stderr": np.nan, "xi_lo": np.nan, "xi_hi": np.nan,
                "R2": np.nan, "rmin": float(rmin) if rmin is not None else np.nan,
                "rmax": float(rmax) if rmax is not None else np.nan, "n_used": 0.0}

    x = r[mask]
    y = np.log(I[mask])

    w = None
    if method.upper() == "WLS":
        if sigma_r is not None:
            sigma = np.asarray(sigma_r, dtype=float)[mask]
            w = 1.0 / (sigma ** 2)
            keep = np.isfinite(w) & (w > 0)
            x, y, w = x[keep], y[keep], w[keep]
        elif N_r is not None:
            w = np.asarray(N_r, dtype=float)[mask]
            keep = np.isfinite(w) & (w > 0)
            x, y, w = x[keep], y[keep], w[keep]

    slope, intercept, stderr, R2, df = _weighted_linear_fit(x, y, w)

    # slope ≈ -1/xi
    if not np.isfinite(slope) or slope >= 0:
        xi = np.inf if np.isfinite(slope) and slope >= 0 else np.nan
        xi_stderr = np.nan
        xi_lo = np.nan
        xi_hi = np.nan
    else:
        xi = -1.0 / slope
        # Propagate stderr of slope to xi via delta method: Var[1/(-slope)] ≈ (1/slope^4) Var[slope]
        if np.isfinite(stderr):
            xi_stderr = float(stderr / (slope ** 2))
            tcrit = _t_crit(alpha, df if df is not None else max(1.0, x.size - 2.0))
            xi_lo = float(xi - tcrit * xi_stderr)
            xi_hi = float(xi + tcrit * xi_stderr)
        else:
            xi_stderr = np.nan
            xi_lo = np.nan
            xi_hi = np.nan

    return {"xi": float(xi), "xi_stderr": float(xi_stderr),
            "xi_lo": float(xi_lo), "xi_hi": float(xi_hi), "R2": float(R2),
            "rmin": float(np.min(x)), "rmax": float(np.max(x)), "n_used": float(x.size if w is None else np.sum(w > 0))}


# -----------------------
# Triangle-inequality diagnostics (NEW)
# -----------------------
def triangle_defect_rr(r_values: Sequence[int],
                       d_E_r_values: Sequence[float]) -> Dict[str, np.ndarray]:
    """
    Compute Δ(r,r) = d_E(2r) - 2 d_E(r). Values with missing/NaN d_E are returned as NaN.

    Returns:
        dict with fields:
          - "r": r where 2r is available
          - "delta_rr": Δ(r,r)
          - "mask_finite": finite entries mask
    """
    r_arr = np.asarray(r_values, dtype=int)
    dE = np.asarray(d_E_r_values, dtype=float)
    # Map r -> dE for quick lookup
    r_to_idx = {int(r): i for i, r in enumerate(r_arr)}
    r_list = []
    delta_list = []
    for r in r_arr:
        r2 = int(2 * r)
        if r <= 0 or r2 not in r_to_idx:
            continue
        i = r_to_idx[r]
        j = r_to_idx[r2]
        val_r = dE[i]
        val_2r = dE[j]
        if np.isfinite(val_r) and np.isfinite(val_2r):
            delta_list.append(float(val_2r - 2.0 * val_r))
        else:
            delta_list.append(np.nan)
        r_list.append(int(r))
    r_out = np.asarray(r_list, dtype=int)
    delta_rr = np.asarray(delta_list, dtype=float)
    mask_finite = np.isfinite(delta_rr)
    return {"r": r_out, "delta_rr": delta_rr, "mask_finite": mask_finite}


def triangle_defect_grid(r_values: Sequence[int],
                         d_E_r_values: Sequence[float],
                         r1_list: Optional[Sequence[int]] = None,
                         r2_list: Optional[Sequence[int]] = None
                         ) -> Dict[str, np.ndarray]:
    """
    Compute a compact grid Δ(r1, r2) = d_E(r1+r2) - d_E(r1) - d_E(r2).

    Args:
        r_values, d_E_r_values: arrays of r and d_E(r)
        r1_list, r2_list: optional sequences of r1, r2. If None, use the set of available r (clipped).

    Returns:
        dict with fields: {"r1", "r2", "delta", "mask_finite"}
          where "delta" has shape (len(r1), len(r2)).
    """
    r_arr = np.asarray(r_values, dtype=int)
    dE = np.asarray(d_E_r_values, dtype=float)
    r_to_val = {int(r): float(v) for r, v in zip(r_arr, dE)}
    r_all = np.array(sorted(list(set(int(r) for r in r_arr if r > 0))), dtype=int)

    r1 = np.asarray(r1_list, dtype=int) if r1_list is not None else r_all
    r2 = np.asarray(r2_list, dtype=int) if r2_list is not None else r_all

    delta = np.full((r1.size, r2.size), np.nan, dtype=float)
    for i, a in enumerate(r1):
        for j, b in enumerate(r2):
            c = a + b
            va = r_to_val.get(int(a), np.nan)
            vb = r_to_val.get(int(b), np.nan)
            vc = r_to_val.get(int(c), np.nan)
            if np.isfinite(va) and np.isfinite(vb) and np.isfinite(vc):
                delta[i, j] = vc - va - vb
            else:
                delta[i, j] = np.nan

    return {
        "r1": r1,
        "r2": r2,
        "delta": delta,
        "mask_finite": np.isfinite(delta),
    }


# -----------------------
# FSS helper (NEW)
# -----------------------
def scaled_dE_over_L(r_values: Sequence[int],
                     d_E_r_values: Sequence[float],
                     L: int) -> Dict[str, np.ndarray]:
    """
    Return scaled curves for the FSS plot: x = r/L, y = d_E/L with finite-only mask.

    Args:
        r_values: separations
        d_E_r_values: emergent distance
        L: system size

    Returns:
        {"x": r/L, "y": d_E/L, "mask_finite": finite mask}
    """
    r = np.asarray(r_values, dtype=float)
    dE = np.asarray(d_E_r_values, dtype=float)
    x = r / float(L)
    y = dE / float(L)
    mask = np.isfinite(x) & np.isfinite(y)
    return {"x": x[mask], "y": y[mask], "mask_finite": mask[mask]}


# -----------------------
# Backward-compatible fit wrappers (existing names)
# -----------------------
def fit_beta_from_I_r(r_values_fit: Sequence[Union[int, float]],
                      I_r_values_fit: Sequence[float],
                      mi_floor: float = DEFAULT_MI_FLOOR) -> Tuple[float, float, float]:
    """
    Backward-compatible wrapper: returns (beta, beta_err, R2) where beta ≡ X.
    Internally calls `fit_powerlaw_exponent` with default method="WLS" (falls back to OLS).

    NOTE: Excludes points with I <= mi_floor and non-finite values.
    """
    res = fit_powerlaw_exponent(r_values_fit, I_r_values_fit, mi_floor=mi_floor, method="WLS")
    beta = float(res["X"])
    beta_err = float(res.get("X_stderr", np.nan))
    R2 = float(res.get("R2", np.nan))
    return beta, beta_err, R2


def fit_k_from_d_E_r(r_values_fit: Sequence[Union[int, float]],
                     d_E_r_values_fit: Sequence[float]) -> Tuple[float, float, float]:
    """
    Fit d_E ≈ k * r (through the origin). Returns (k, k_err, R2).
    The standard error is computed for an origin-constrained regression.

    Returns:
        k (float), k_err (float), R2 (float)
    """
    r = np.asarray(r_values_fit, dtype=float)
    dE = np.asarray(d_E_r_values_fit, dtype=float)
    mask = np.isfinite(r) & np.isfinite(dE) & (r > 0)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan, np.nan

    x = r[mask]
    y = dE[mask]

    # Slope through the origin
    Sxx = float(np.sum(x ** 2))
    Sxy = float(np.sum(x * y))
    if Sxx <= 0:
        return np.nan, np.nan, np.nan
    k = Sxy / Sxx

    # Residuals and R^2 (origin model)
    y_hat = k * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else (1.0 if ss_res < 1e-15 else 0.0)

    # Correct SE for origin-constrained regression: Var(k) = s^2 / Sxx, s^2 = RSS / (n-1)
    n = x.size
    if n <= 1:
        return float(k), np.nan, float(R2)
    s2 = ss_res / (n - 1)
    k_err = math.sqrt(s2 / Sxx) if s2 >= 0 else np.nan
    return float(k), float(k_err), float(R2)


# -----------------------
# Legacy analysis helpers
# -----------------------
def calculate_central_charge(psi_gs, L: int) -> Tuple[float, float]:
    """
    Legacy: estimate c from block entanglement scaling S(ell) vs log[(L/pi) sin(pi ell/L)] (nats).
    """
    if L < 4:
        return np.nan, np.nan
    min_block_for_fit = 2
    max_block_for_fit = L // 2
    ells = np.arange(min_block_for_fit, max_block_for_fit + 1, 2)
    if len(ells) < 2:
        ells = np.arange(min_block_for_fit, max_block_for_fit + 1)
        if len(ells) < 2:
            return np.nan, np.nan
    # quimb returns bits; convert to nats
    entropies_block_bits = np.array([psi_gs.entropy(ell) for ell in ells], dtype=float)
    entropies_block_nats = entropies_block_bits * np.log(2.0)
    valid_indices = np.isfinite(entropies_block_nats)
    if np.sum(valid_indices) < 2:
        return np.nan, np.nan
    ells_fit = ells[valid_indices]
    entropies_fit_nats = entropies_block_nats[valid_indices]
    x_cc = np.log((L / np.pi) * np.sin(np.pi * ells_fit / L))
    slope, _, _, _, std_err = linregress(x_cc, entropies_fit_nats)
    c_calabrese = 3.0 * slope
    c_err = 3.0 * std_err
    return float(c_calabrese), float(c_err)


def calculate_correlation_length(psi_gs, L: int, r_max: int, op_for_corr) -> float:
    """
    Legacy: estimate correlation length from averaged two-point correlations with operator op_for_corr.
    """
    avg_C_r_values = []
    r_values_corr = list(range(1, r_max + 1))
    for r_dist in r_values_corr:
        r_corrs_vals = []
        num_pairs = 0
        for i_ref in range(L - r_dist):
            j_curr = i_ref + r_dist
            try:
                corr_val_raw = psi_gs.correlation(A=op_for_corr, i=i_ref, j=j_curr, B=None)
                corr_val = corr_val_raw.item() if hasattr(corr_val_raw, 'item') else corr_val_raw
                r_corrs_vals.append(corr_val.real if isinstance(corr_val, complex) else corr_val)
                num_pairs += 1
            except Exception:
                pass
        avg_C_r_values.append(np.mean(r_corrs_vals) if num_pairs > 0 else np.nan)

    valid_indices = [i for i, C_r in enumerate(avg_C_r_values)
                     if np.isfinite(C_r) and (abs(C_r) > 1e-12)]
    if len(valid_indices) < 2:
        return np.nan

    r_fit = np.array([r_values_corr[i] for i in valid_indices], dtype=float)
    log_abs_C_fit = np.log(np.abs([avg_C_r_values[i] for i in valid_indices]))
    slope, _, _, _, _ = linregress(r_fit, log_abs_C_fit)
    if not np.isfinite(slope) or slope >= -1e-9:
        return np.inf if abs(slope) < 1e-9 else np.nan
    xi = -1.0 / slope
    return float(xi)


def get_max_bond_entropy(psi_gs) -> float:
    """
    Legacy: maximum bond entropy across the chain (nats).
    """
    L = psi_gs.L
    if L < 2:
        return 0.0
    bond_entropies_bits = [psi_gs.entropy(i) for i in range(1, L)]
    bond_entropies_nats = [s * np.log(2.0) for s in bond_entropies_bits if np.isfinite(s)]
    return float(np.max(bond_entropies_nats)) if bond_entropies_nats else np.nan


def extract_all_properties(psi_gs, L: int, r_max_mi: int, r_max_corr: int,
                           fit_r_min: int = 1, fit_r_max_factor: float = 0.8) -> Dict[str, Union[float, List, np.ndarray]]:
    """
    Legacy convenience wrapper for MPS analyses (kept for backward compatibility).
    """
    results = {}
    num_lanczos_mi = 20
    r_vals_mi, I_r_vals = calculate_mutual_information_I_r(
        psi_gs, L, r_max_mi, num_lanczos_vecs=num_lanczos_mi
    )
    results['r_values_mi'] = r_vals_mi
    results['I_r_values'] = I_r_vals

    d_E_r_vals = calculate_entanglement_distance_d_E_r(r_vals_mi, I_r_vals)
    results['d_E_r_values'] = d_E_r_vals

    actual_r_max_fit_mi_val = 0
    if r_vals_mi:
        idx_limit = min(int(len(r_vals_mi) * fit_r_max_factor), len(r_vals_mi) - 1)
        if idx_limit >= 0:
            actual_r_max_fit_mi_val = r_vals_mi[idx_limit]
        elif r_vals_mi:
            actual_r_max_fit_mi_val = r_vals_mi[0]
    actual_r_max_fit_mi_val = max(actual_r_max_fit_mi_val, fit_r_min)

    fit_indices = [idx for idx, r_val in enumerate(r_vals_mi) if fit_r_min <= r_val <= actual_r_max_fit_mi_val]
    if not fit_indices:
        beta, beta_err, R2_I, k, k_err, R2_dE = (np.nan,) * 6
    else:
        r_for_fit_beta = [r_vals_mi[i] for i in fit_indices]
        I_for_fit_beta = [I_r_vals[i] for i in fit_indices]
        d_E_for_fit_k = [d_E_r_vals[i] for i in fit_indices]
        # Use the production exponent fitter (WLS/OLS fallback)
        beta, beta_err, R2_I = fit_beta_from_I_r(r_for_fit_beta, I_for_fit_beta)
        k, k_err, R2_dE = fit_k_from_d_E_r(r_for_fit_beta, d_E_for_fit_k)

    results.update({
        'beta': beta, 'beta_err': beta_err, 'R2_I': R2_I,
        'k': k, 'k_err': k_err, 'R2_dE': R2_dE
    })

    # Central charge from block entropies (nats)
    c, c_err = calculate_central_charge(psi_gs, L)
    results.update({'central_charge': c, 'central_charge_err': c_err})

    # Correlation length using σ^z operator (if quimb available)
    try:
        import quimb
        sz_op = quimb.spin_operator('Z', S=0.5)
        xi = calculate_correlation_length(psi_gs, L, r_max_corr, sz_op)
    except ImportError:
        xi = np.nan
    results['correlation_length_xi'] = xi

    results['max_bond_entropy'] = get_max_bond_entropy(psi_gs)
    return results


# -----------------------
# Self-test (optional)
# -----------------------
if __name__ == '__main__':
    print("--- mi_tools.py: basic self-checks (production-ready) ---")
    # Synthetic power-law I(r) ~ r^{-X} with noise: test exponent fit
    rng = np.random.default_rng(123)
    X_true = 1.80
    r_test = np.arange(2, 60, 1, dtype=float)
    I_clean = r_test ** (-X_true)
    I_noisy = I_clean * np.exp(rng.normal(0.0, 0.05, size=r_test.size))  # log-normal noise
    res = fit_powerlaw_exponent(r_test, I_noisy, method="OLS", rmin=8, rmax=40)
    print(f"Synthetic exponent fit: X_true={X_true:.2f} -> X_fit={res['X']:.3f} (R2={res['R2']:.3f})")

    # Triangle defect Δ(r,r) sanity on perfect power-law (should be ≤ 0 for X<=2)
    dE = I_clean ** (-0.5)
    tri = triangle_defect_rr(r_test.astype(int), dE)
    print(f"Δ(r,r) finite fraction: {np.mean(tri['mask_finite']):.2f}, example Δ at r=4: "
          f"{tri['delta_rr'][np.where(tri['r']==4)[0][0]] if 4 in tri['r'] else 'n/a'}")

    # Legacy MPS quick test (skipped if quimb not installed)
    L_test = 8
    bond_dim_test = 4
    try:
        import quimb.tensor as qtn
        print(f"Creating a random MPS for L={L_test}, bond_dim={bond_dim_test}...")
        psi_test = qtn.MPS_rand_state(L_test, bond_dim_test, phys_dim=2, normalize=True)
        try:
            psi_test.left_canonise()
        except AttributeError:
            psi_test.left_canonize()
        print("Random MPS created and canonicalized.")
        r_max_mi_test = L_test // 2
        r_max_corr_test = L_test // 2
        print(f"Extracting properties with r_max_mi={r_max_mi_test}, r_max_corr={r_max_corr_test}...")

        print("\nTesting _calculate_subsystem_entropy...")
        S_0 = _calculate_subsystem_entropy(psi_test, [0])
        print(f"  S(site 0): {S_0}")
        if L_test > 1:
            S_01 = _calculate_subsystem_entropy(psi_test, [0, 1])
            print(f"  S(sites 0,1): {S_01}")
            S_0_L_minus_1 = _calculate_subsystem_entropy(psi_test, [0, L_test - 1])
            print(f"  S(sites 0, {L_test-1}): {S_0_L_minus_1}")

        print("\nTesting extract_all_properties (legacy path)...")
        all_props = extract_all_properties(
            psi_test, L_test, r_max_mi_test, r_max_corr_test, fit_r_min=1, fit_r_max_factor=1.0
        )
        print("\n--- Extracted Properties (Test MPS) ---")
        for key, val in all_props.items():
            if isinstance(val, (list, np.ndarray)) and hasattr(val, '__len__'):
                if len(val) > 5:
                    print(f"  {key}: Array of length {len(val)}, first 5: {np.array(val)[:5]}")
                else:
                    print(f"  {key}: {val}")
            elif isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
        print("\nBasic tests for mi_tools.py finished.")
    except ImportError:
        print("Quimb not found. Skipping MPS-specific tests (post-processing utils verified above).")
    except Exception as e:
        print(f"An error: {e}")
        import traceback
        traceback.print_exc()