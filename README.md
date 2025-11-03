Reproducibility README — TFIM (JW+BdG, OBC) MI & FSS pipeline

This repository accompanies the submission. It contains an **exact** TFIM solver (Jordan–Wigner + BdG) and the **post‑processing/analysis pipeline**. It establishes:
- Two‑mode JW **mutual information** `I(r)` in **nats**
- Emergent distance `d_E(r) = I(r)^(-1/2)`
- Robust **exponent fits** `I(r) \sim r^{-X}` and off‑critical **correlation length** `\xi`
- **BCFT chord‑distance refit** at criticality: `\rho(r,L) = (L/\pi)\sin(\pi r/L)`
- **Finite‑size scaling (FSS)** of the chord‑distance exponent `X(L)` to `X_\infty`

The code is numerically deterministic (no randomness). For bit‑for‑bit reproducibility, pin your BLAS/LAPACK implementation and set single‑threaded BLAS (e.g. OMP_NUM_THREADS=1, MKL_NUM_THREADS=1) so reductions are ordered.

---

## 0) Repository layout

tfim_exact.py           JW+BdG solver + Gaussian entropies (nats)
mi_tools.py             Model‑agnostic analysis helpers (fitting, Δ, FSS utilities)
run_tfim_study.py       Campaign driver: builds full results_*.json files
postprocess_tfim_results.py      Creates compact "lite" JSONs and summary.csv from full results
config_tfim_exact.json  Canonical configuration used in the paper

Outputs land in:

simulation_results_tfim_gold_master/  full JSONs (one per (L,h))
simulation_results_tfim_lite/         compact “lite” JSONs + summary.csv

---

## 1) Environment

Tested with Python ≥3.10.
bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy pandas matplotlib

Alternatively (pinned), to reproduce the exact environment used for the figures, create a requirements.txt file with:

contourpy==1.3.3
cycler==0.12.1
fonttools==4.60.1
kiwisolver==1.4.9
matplotlib==3.10.7
numpy==2.3.3
packaging==25.0
pandas==2.3.3
pillow==11.3.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
scipy==1.16.2
six==1.17.0
tzdata==2025.2

then install with:

pip install -r requirements.txt

No external quantum libraries are required for the exact solver. (Some legacy MPS helpers in mi_tools.py load quimb lazily if present; they are not used by the TFIM pipeline.)

## 2) Reproduce the Campaign (end-to-end)

(A) Generate the full outputs
python run_tfim_study.py

This consumes config_tfim_exact.json:
{
  "output_directory": "simulation_results_tfim_gold_master",
  "model": "TFIM_EXACT_OBC",
  "J": 1.0,
  "h_values_to_run": [1.0, 0.5, 2.0],
  "L_values": [256, 512, 1024, 2048],
  "analysis_params": {
    "r_max_factor": 0.5,
    "fit_r_min": 12,
    "fit_r_max_factor": 0.7,
    "fit_r_max_cap": 256,
    "mi_floor": 1e-12,
    "bulk_trim_fraction": 0.15,
    "fit_method": "wls"
  }
}

- The script **skips** a run if the corresponding results_*.json already exists in output_directory.

Each results_L{L}_H{h}.json contains arrays ([r, I(r), d_E(r)]), pair statistics (N_r | sigma_r_jk), the r‑based power‑law fit ("fit"), plus additional blocks:

-fit_chord (critical, h = J): power‑law refit vs chord distance \rho on the same r‑window.

-fit_exponential (off critical): \log I \sim -r/\xi.

- At criticality (h = J), the chord‑distance refit uses the **same r‑window** and **mirrors the weighting choice** (WLS/OLS) used by the r‑fit.

Reporting policy (what we quote):

-At criticality (h = J), we quote the chord‑distance exponent from the JSON block fit_chord.beta_X (BCFT‑correct for OBC). The lattice‑r fit in fit.beta_X is kept for QA only and is not used in text or tables.

-Off criticality (h ≠ J), we quote the exponential correlation length fit_exponential.xi.

-The postprocessor writes a single “official” column in summary.csv:

-beta_X_reported = beta_X_chord at h = J, blank otherwise.

-The r‑based value appears under beta_X_r solely for QA to avoid confusion.

Sanity check (automatic): at (h = J = 1), the energy density satisfies E_0/L → -4/π ≈ -1.27324; each JSON records energy_per_site and the reference so you can verify this automatically.

(B) Create compact “lite” artifacts + table

```python postprocess_tfim_results.py \
  --input simulation_results_tfim_gold_master \
  --out   simulation_results_tfim_lite \
  --keep-r-up-to 200```

This writes one lite_results_*.json per run and a summary.csv.

-The lite JSON may retain series values beyond the fit cap for plotting/QA; fits always and only use the recorded windows (fit.fit_r_min/fit.fit_r_upper for the r‑fit, and fit_chord.window_r/fit_exponential.window_r for the BCFT/off‑critical blocks).

-The lite creator never truncates below the fit window actually used; it keeps r ≤ max(keep-r-up-to, effective_upper_bound) where the effective upper bound is the maximum of the window uppers used by any saved fit block. In the CSV, the fit_r_upper column records this effective upper bound so the table reflects the true guard.

## 3) Reproduce the Campaign (end-to-end)

Critical (h = J):
Use the chord‑distance exponent in each file:

results_....json → fit_chord.beta_X (with 95% CI and R²)

or, from the lite & table,

summary.csv → columns: beta_X_chord, beta_X_chord_lo, beta_X_chord_hi, R2_I_chord

This is the value appropriate for extrapolating X(L) to L → ∞ (the r‑based fit.beta_X is kept for within‑L diagnostics only). See the representative critical file structure, e.g. lite_results_L512_H1p0.json.

Off critical (h ≠ J):
Use the exponential fit:

results_....json → fit_exponential.xi (with 95% CI and R²)
summary.csv → xi, xi_lo, xi_hi, R2_logI_vs_r

You can also inspect the large‑X r‑based fit at (h = 0.5, 2.0) as a diagnostic (MI decays exponentially; the r‑power is not physically meaningful off criticality).

# 4) Minimal FSS recipe (critical point)
Here is a compact, self‑contained snippet that reads summary.csv, selects h=1, and extrapolates X(L) to L→∞ by a linear fit in 1/L^2 (the default choice used in the paper; other smooth 1/L^{\alpha} forms give consistent limits):

```python
import pandas as pd, numpy as np

tbl = pd.read_csv("simulation_results_tfim_lite/summary.csv")

# Keep only critical runs (h == 1) and coerce reported exponents to numeric
crit = tbl[np.isclose(tbl["h"], 1.0)].copy()
crit["beta_X_reported"] = pd.to_numeric(crit["beta_X_reported"], errors="coerce")
crit = crit.dropna(subset=["beta_X_reported"])

L  = crit["L"].to_numpy(dtype=float)
Xc = crit["beta_X_reported"].to_numpy(dtype=float)  # chord-distance exponent

# Leading BCFT correction ∝ 1/L^2
x = 1.0 / (L**2)
A = np.vstack([x, np.ones_like(x)]).T

# Unweighted least squares is sufficient
coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)
X_inf = coef[1]  # intercept at x=0 is X(∞)

print("X_infinity (chord-distance) =", X_inf)```

This one cell reproduces the **thermodynamic exponent** reported in the manuscript.

---

## 5) Data dictionary (lite JSON)

Each `lite_results_L{L}_H{h}.json` contains (keys shown in full):

- **Identifiers**: 
  - `model`
  - `basis_MI` (JW two‑mode; value is `'JW_fermionic_two_mode'`)
  - `units="nats"`
  - `J`, `h`, `L`.

- **Energy QA**: `energy_per_site`.

- **Fit (r‑based diagnostics)**: \fit.method`, `fit.weights`, `fit_r_min`, `fit_r_upper`, `r_points_used`, `beta_X`, `beta_X_lo/hi` (95% CI), `R2_I`, `mi_floor_used`.`


- **BCFT / off‑critical fits**: `fit_chord` (*critical only*) and `fit_exponential` (*off critical only*).

 At h = J, the chord‑distance refit uses the same r‑window and mirrors the weighting choice used by the r‑fit.

- **Analysis parameters**: `r_max_factor`, `fit_r_max_factor`, `fit_r_max_cap` (if set), `bulk_trim_applied`, `bulk_trim_fraction_applied`.

- **Series (possibly truncated, but never below the fit window actually used)**: `r_values`, `I_r_values`, `d_E_r_values`, `mask_floor`, and, when available, `N_r_values`, `sigma_r_jk`.

- **Diagnostics**: `delta_rr` for the triangle‑inequality check and `fss_scaled` for convenience.

---

## 6) Reproducibility notes & common pitfalls

- **Chord distance matters (critical point)**: For OBC critical chains, the correct scaling variable is  
  `\rho(r,L) = (L/\pi)\sin(\pi r/L)` (the BCFT chord distance).

- **All `L→∞` extrapolations use `fit_chord.beta_X`**, not the stored r‑based exponent.

- **Units**: entropies/MI are **nats**; to convert to bits divide by `\ln 2`.

- **MI floor**: values `I ≤ 10^{-12}` (nats) are masked; below‑floor points do not enter fits.  
  In this repository’s pipeline the floor is applied **at the pair level before averaging `I(r)`** (preferred path). For backward compatibility with older backends only, a fallback path applies the floor **after** averaging; this fallback is not used when running the code in this repo. In all cases, points with averaged `I(r) ≤` floor are omitted from `d_E` (set to NaN), from fits, and from `Δ`.

- **Bulk trim**: by default 15% of sites are trimmed on each edge when averaging over pairs.

- **Fit‑window policy**: r‑fits use `r ∈ [fit_r_min, r_upper]` with

r_max   = floor(r_max_factor * L)
r_upper = floor(fit_r_max_factor * r_max)
r_upper = max(fit_r_min, r_upper)
if fit_r_max_cap is set: r_upper = min(r_upper, fit_r_max_cap)

This exact r‑window is mirrored in the chord‑distance refit at criticality (`fit_chord.window_r = [fit_r_min, r_upper]`).

- **Determinism**: solver is exact and uses dense linear algebra. Results are deterministic given a fixed Python/NumPy/SciPy/BLAS stack. In practice we observe insensitivity to thread counts; exact bit‑for‑bit identity can depend on the BLAS implementation and configuration.

- **Energy QA**: at `h = J = 1`, `E_0/L → -4/\pi` (confirmed in produced files). See `lite_results_L512_H1p0.json`.

---

## 7) How to rerun with different grids or windows

Edit `config_tfim_exact.json`:

- `L_values`, `h_values_to_run` to change the grid,
- `fit_r_min`, `fit_r_max_factor`, and (optionally) `fit_r_max_cap` to change the (shared) r‑window used inside each `L`,
- `bulk_trim_fraction` and `mi_floor` if you need stricter masking.
- \fit_method` to choose “wls”, “ols”, or “theil-sen”.`

Re‑run step 2(A) and then 2(B). The table will reflect the new choices.

---

## 8) Contact

Please reach out to me with any issues running the code. The repository includes everything required to regenerate the full JSONs and the BCFT‑consistent FSS numbers reported in the manuscript.
