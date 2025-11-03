#!/usr/bin/env python3
# postprocess_tfim_results.py
# Create compact “lite” outputs + a single CSV summary from the large results_*.json files.
# Reporting policy clarification (unchanged):
#  - At criticality (h == J), use the chord-distance exponent for reporting: fit_chord.beta_X
#  - In the CSV, the lattice-r fit is recorded under beta_X_r (QA only).
#  - A consolidated column beta_X_reported is provided: beta_X_chord at h==J, else blank.
#
# CHANGELOG (minimal, surgical):
#  - Robustly read the chord-fit block whether the key is "fit_chord" or "fit_ chord".
#  - Guard lite truncation against cutting below ANY recorded fit window:
#       max(keep-r-up-to, fit.fit_r_upper, fit_chord.window_r[1], fit_exponential.window_r[1]).
#  - Propagate analysis_params.fit_r_max_factor / fit_r_max_cap into lite JSON (non-breaking).
#
# Motivation/examples (from your uploaded data):
#  * Pre-cap critical L=256 had fit_r_upper=89 (legacy window) and a populated chord block.  
#  * After adding the cap, critical L=2048 reports fit_r_upper=256 with a chord block present.  
#  These fixes ensure the lite trimming and summary.csv reflect that cap exactly and reproducibly.

import os, json, csv, argparse, glob, math

def _safe_float(x):
    try:
        if isinstance(x, str) and x.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return x  # leave as string marker
        return float(x)
    except Exception:
        return x

def _truncate_series(r_vals, arr, r_max_keep):
    """Return r<=r_max_keep subset for (r_vals, arr)."""
    out_r, out_a = [], []
    for r, a in zip(r_vals, arr):
        try:
            r_num = int(r)
        except Exception:
            continue
        if r_num <= r_max_keep:
            out_r.append(r_num)
            out_a.append(a)
    return out_r, out_a

def _get_block_robust(d, primary_key, fallback_keys=()):
    """
    Return d[primary_key] if present; otherwise try known fallbacks.
    Also accept minor variants that differ only by whitespace (e.g., "fit_ chord").
    """
    if primary_key in d:
        return d[primary_key]
    for k in fallback_keys:
        if k in d:
            return d[k]
    # whitespace-tolerant scan for the primary_key
    target = primary_key.replace("_", "").lower()
    for k in d.keys():
        if k.replace(" ", "").replace("_", "").lower() == target:
            return d[k]
    return {}

def _coalesce_upper_bounds(fit, fit_chord, fit_exp):
    """
    Collect all known sources of the 'upper r' bound and return the maximum integer upper bound.
    Sources (if present):
      - fit['fit_r_upper']
      - fit_chord['window_r'][1]
      - fit_exponential['window_r'][1]
    """
    candidates = []

    # 1) legacy r-fit upper
    try:
        fu = fit.get("fit_r_upper", None)
        if fu is not None:
            candidates.append(int(math.ceil(float(fu))))
    except Exception:
        pass

    # 2) chord window upper
    try:
        w = fit_chord.get("window_r", None)
        if isinstance(w, (list, tuple)) and len(w) >= 2 and w[1] is not None:
            candidates.append(int(math.ceil(float(w[1]))))
    except Exception:
        pass

    # 3) exponential window upper
    try:
        w = fit_exp.get("window_r", None)
        if isinstance(w, (list, tuple)) and len(w) >= 2 and w[1] is not None:
            candidates.append(int(math.ceil(float(w[1]))))
    except Exception:
        pass

    return max(candidates) if candidates else None

def main():
    ap = argparse.ArgumentParser(description="Make reproducible 'lite' outputs from big TFIM JSONs.")
    ap.add_argument("--input", required=True, help="Folder with results_*.json (from run_tfim_study.py)")
    ap.add_argument("--out", required=True, help="Output folder for lite files and summary.csv")
    ap.add_argument("--keep-r-up-to", type=int, default=256, help="Keep data only up to this r (default 256)")
    args = ap.parse_args()

    in_dir = args.input
    out_dir = args.out
    r_keep = int(max(1, args.keep_r_up_to))

    os.makedirs(out_dir, exist_ok=True)

    # Find all campaign outputs
    paths = sorted(glob.glob(os.path.join(in_dir, "results_*.json")))
    if not paths:
        print(f"[postprocess] No files found in '{in_dir}'. Expected results_*.json.")
        return

    summary_path = os.path.join(out_dir, "summary.csv")

    # CSV schema (clarified naming + consolidated 'reported' column)
    fieldnames = [
        "file", "L", "h", "J",
        "energy_per_site",
        "fit_method", "weights",
        "fit_r_min", "fit_r_upper", "r_points_used",
        # r-lattice fit (QA only; not used for reporting)
        "beta_X_r", "beta_X_r_err", "beta_X_r_lo", "beta_X_r_hi", "R2_I_r",
        # chord-distance fit (used for reporting at criticality)
        "beta_X_chord", "beta_X_chord_lo", "beta_X_chord_hi", "R2_I_chord",
        # consolidated column the paper should quote (critical only)
        "beta_X_reported", "beta_X_reported_lo", "beta_X_reported_hi", "X_source",
        # off-critical diagnostics
        "xi", "xi_lo", "xi_hi", "R2_logI_vs_r",
        # analysis policy recap
        "mi_floor_used",
        "r_max_factor",
        "bulk_trim_fraction_applied",
        "bulk_trim_applied",
    ]
    rows = []
    index_entries = []

    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[skip] Could not read {p}: {e}")
            continue

        # Core identifiers
        J = _safe_float(data.get("J", 1.0))
        h = _safe_float(data.get("h", None))
        L = int(data.get("L", -1))

        # Arrays we’ll compact
        r_values = data.get("r_values", [])
        I_r_values = data.get("I_r_values", [])
        d_E_r_values = data.get("d_E_r_values", [])
        mask_floor = data.get("mask_floor", [])
        N_r_values = data.get("N_r_values", [])
        sigma_r_jk = data.get("sigma_r_jk", [])

        # Optional FSS payload
        fss = data.get("fss_scaled", {})
        x_scaled = fss.get("x_r_over_L", [])
        y_scaled = fss.get("y_dE_over_L", [])

        # Fit + analysis metadata
        fit = data.get("fit", {})
        method = fit.get("method", "")
        weights = fit.get("weights", "")  # may be absent in older files
        fit_r_min = _safe_float(fit.get("fit_r_min", None))
        fit_r_upper = _safe_float(fit.get("fit_r_upper", None))
        n_used = int(fit.get("r_points_used", 0))
        X_r = _safe_float(fit.get("beta_X", None))
        X_r_err = _safe_float(fit.get("beta_X_err", None))
        X_r_lo = _safe_float(fit.get("beta_X_lo", None))
        X_r_hi = _safe_float(fit.get("beta_X_hi", None))
        R2_I_r = _safe_float(fit.get("R2_I", None))
        mi_floor_used = _safe_float(fit.get("mi_floor_used", None))

        # Preserve additional fit blocks when present (robust chord key handling)
        fit_chord = _get_block_robust(
            data,
            primary_key="fit_chord",
            fallback_keys=("fit chord", "fit_ chord")
        )
        fit_exp = data.get("fit_exponential", {})

        # Determine effective upper bound used by *any* fit block and guard truncation against it
        fit_r_upper_num = None
        try:
            fit_r_upper_num = int(math.ceil(float(fit_r_upper))) if fit_r_upper is not None else None
        except Exception:
            fit_r_upper_num = None

        # Coalesce all available window uppers
        eff_upper = _coalesce_upper_bounds(fit, fit_chord, fit_exp)
        if eff_upper is None:
            eff_upper = fit_r_upper_num  # fall back to the legacy
        # Effective keep threshold: never truncate below the used fit window
        if eff_upper is not None:
            r_keep_effective = max(r_keep, eff_upper)
        else:
            r_keep_effective = r_keep

        # Truncate to r<=r_keep_effective
        r_short, I_short = _truncate_series(r_values, I_r_values, r_keep_effective)
        _, dE_short = _truncate_series(r_values, d_E_r_values, r_keep_effective)
        _, mask_short = _truncate_series(r_values, mask_floor, r_keep_effective)
        _, Nr_short = _truncate_series(r_values, N_r_values, r_keep_effective) if N_r_values else ([], [])
        _, sig_short = _truncate_series(r_values, sigma_r_jk, r_keep_effective) if sigma_r_jk else ([], [])

        # Δ(r,r) — also trim sensibly (2r must be in range, so we cap at r_keep//2)
        delta = data.get("delta_rr", {})
        delta_r_vals = delta.get("r_values", [])
        delta_vals = delta.get("values", [])
        drr_r_short, drr_short = _truncate_series(delta_r_vals, delta_vals, r_keep // 2)

        # Analysis params (propagate extra knobs for transparency in lite JSON)
        analysis = data.get("analysis_params", {})
        r_max_factor = _safe_float(analysis.get("r_max_factor", None))
        bulk_applied = bool(analysis.get("bulk_trim_applied", False))
        bulk_alpha_applied = _safe_float(analysis.get("bulk_trim_fraction_applied", 0.0))
        fit_r_max_factor = _safe_float(analysis.get("fit_r_max_factor", None))
        fit_r_max_cap = analysis.get("fit_r_max_cap", None)

        # Energy
        e_per_L = _safe_float(data.get("energy_per_site", None))

        # --- Determine the exponent to report (critical: chord; gapped: none) ---
        is_critical = (
            h is not None and J is not None
            and isinstance(h, (int, float)) and isinstance(J, (int, float))
            and abs(float(h) - float(J)) < 1e-12
        )
        Xc = _safe_float(fit_chord.get("beta_X", None))
        Xc_lo = _safe_float(fit_chord.get("beta_X_lo", None))
        Xc_hi = _safe_float(fit_chord.get("beta_X_hi", None))
        R2c = _safe_float(fit_chord.get("R2_I", None))

        if is_critical and isinstance(Xc, (int, float)) and math.isfinite(float(Xc)):
            X_reported = Xc
            X_reported_lo = Xc_lo
            X_reported_hi = Xc_hi
            X_source = "chord (critical)"
        else:
            # off critical or no chord fit present
            X_reported = ""
            X_reported_lo = ""
            X_reported_hi = ""
            X_source = "n/a"

        # Tag/name for lite artifact
        tag = f"L{L}_H{str(h).replace('.', 'p')}"
        lite_name = f"lite_results_{tag}.json"
        lite_path = os.path.join(out_dir, lite_name)

        # Write per-run lite JSON (schema preserved; extra analysis fields appended)
        lite = {
            "model": data.get("model", "TFIM_EXACT_OBC"),
            "basis_MI": data.get("basis_MI", ""),
            "units": data.get("entropy_units", "nats"),
            "J": J, "h": h, "L": L,
            "energy_per_site": e_per_L,
            "fit": {
                "method": method,
                "weights": weights,
                "fit_r_min": fit_r_min,
                "fit_r_upper": fit_r_upper,  # keep original reporting
                "r_points_used": n_used,
                "beta_X": X_r, "beta_X_err": X_r_err,
                "beta_X_lo": X_r_lo, "beta_X_hi": X_r_hi,
                "R2_I": R2_I_r,
                "mi_floor_used": mi_floor_used
            },
            "fit_chord": fit_chord,
            "fit_exponential": fit_exp,
            "analysis_params": {
                "r_max_factor": r_max_factor,
                "bulk_trim_applied": bulk_applied,
                "bulk_trim_fraction_applied": bulk_alpha_applied,
                # added (non-breaking; helps referee reproducibility)
                "fit_r_max_factor": fit_r_max_factor,
                "fit_r_max_cap": fit_r_max_cap
            },
            "r_values": r_short,
            "I_r_values": I_short,
            "d_E_r_values": dE_short,
            "mask_floor": mask_short,
            "N_r_values": Nr_short,
            "sigma_r_jk": sig_short,
            "delta_rr": {"r_values": drr_r_short, "values": drr_short},
            "fss_scaled": {"x_r_over_L": x_scaled, "y_dE_over_L": y_scaled}
        }
        with open(lite_path, "w") as f:
            json.dump(lite, f, indent=2)

        # For the CSV, prefer to record the effective window upper (so the table reflects the true guard)
        fit_r_upper_for_csv = eff_upper if eff_upper is not None else fit_r_upper

        # Add to CSV summary
        rows.append({
            "file": lite_name,
            "L": L, "h": h, "J": J,
            "energy_per_site": e_per_L,
            "fit_method": method,
            "weights": weights,
            "fit_r_min": fit_r_min,
            "fit_r_upper": fit_r_upper_for_csv,
            "r_points_used": n_used,
            # lattice-r fit
            "beta_X_r": X_r,
            "beta_X_r_err": X_r_err,
            "beta_X_r_lo": X_r_lo,
            "beta_X_r_hi": X_r_hi,
            "R2_I_r": R2_I_r,
            # chord fit
            "beta_X_chord": Xc,
            "beta_X_chord_lo": Xc_lo,
            "beta_X_chord_hi": Xc_hi,
            "R2_I_chord": R2c,
            # consolidated reporting value
            "beta_X_reported": X_reported,
            "beta_X_reported_lo": X_reported_lo,
            "beta_X_reported_hi": X_reported_hi,
            "X_source": X_source,
            # off-critical exponential
            "xi": _safe_float(fit_exp.get("xi", None)),
            "xi_lo": _safe_float(fit_exp.get("xi_lo", None)),
            "xi_hi": _safe_float(fit_exp.get("xi_hi", None)),
            "R2_logI_vs_r": _safe_float(fit_exp.get("R2_logI_vs_r", None)),
            # policy recap
            "mi_floor_used": mi_floor_used,
            "r_max_factor": r_max_factor,
            "bulk_trim_fraction_applied": bulk_alpha_applied,
            "bulk_trim_applied": bulk_applied,
        })

        # Index entry
        index_entries.append({"file": lite_name, "L": L, "h": h})

    # Write summary.csv
    with open(summary_path, "w", newline="") as csvf:
        w = csv.DictWriter(csvf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write a small index.json
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"runs": index_entries}, f, indent=2)

    print(f"[postprocess] Wrote {len(rows)} lite files, {summary_path}, and index.json into '{out_dir}'.")

if __name__ == "__main__":
    main()