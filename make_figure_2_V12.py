# -*- coding: utf-8 -*-
"""
make_figure_2_V12.py

Two-panel PRB-ready figure:
(a) Mutual information I(r) vs. chord distance ρ(r) at L=2048.
(b) Finite-size scaling (FSS) collapse of d_E(r)/L vs r/L at h/J=1.0,
    with an inset showing fractional deviation (percent) from L=2048.

Data expected in ./simulation_results_tfim_lite/
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator, LogFormatterMathtext, AutoMinorLocator, MultipleLocator, MaxNLocator
)
try:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except Exception:
    inset_axes = None

# ----------------------------
# Matplotlib style tuned for APS journals (PRB/PRL)
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.color": "#999999",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

# ----------------------------
# Colors (Okabe-Ito colorblind-safe + neutrals)
# ----------------------------
COL = {
    "critical":   "#E69F00",  # orange
    "gapped_hi":  "#D55E00",  # vermillion
    "gapped_lo":  "#CC79A7",  # reddish purple
    "ref":        "#333333",  # dark gray (reference line)
}

# Distinct colors per size for FSS panel
FSS_COLS = {
    512:  "#56B4E9",  # Sky Blue
    1024: "#009E73",  # Bluish Green
    2048: "#E69F00",  # Orange (also serves as reference)
}
FSS_MARKERS = {512: "o", 1024: "s", 2048: "^"}

# ----------------------------
# Helpers
# ----------------------------
MI_FLOOR = 1e-12  # consistent with analysis

def chord_distance(r: np.ndarray, L: int) -> np.ndarray:
    """ρ(r) = (L/π) sin(π r / L)."""
    r = np.asarray(r, dtype=float)
    return (L / np.pi) * np.sin(np.pi * r / L)

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def select_valid(x, y, floor=None, x_positive=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if x_positive:
        m &= (x > 0)
    if floor is not None:
        m &= (y > floor)
    return x[m], y[m]

def compute_dE(I: np.ndarray, mi_floor=1e-12) -> np.ndarray:
    """Compute d_E = I^{-1/2} with MI floor handling."""
    I = np.asarray(I, float)
    out = np.full_like(I, np.nan, dtype=float)
    m = np.isfinite(I) & (I > mi_floor)
    out[m] = 1.0 / np.sqrt(I[m])
    return out

def add_panel_label(ax, label, fontsize=12):
    ax.text(0.04, 0.96, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

# ----------------------------
# Panel (a): Mutual information vs chord distance (log-log)
# ----------------------------
def plot_panel_a(ax, files):
    crit = load_json(files["crit"])
    L = int(crit["L"])

    # Critical
    rho_c = chord_distance(crit["r_values"], L)
    I_c = np.asarray(crit["I_r_values"], float)
    rho_c, I_c = select_valid(rho_c, I_c, floor=MI_FLOOR, x_positive=True)
    ax.plot(rho_c, I_c, "-", lw=1.8, color=COL["critical"],
            label=r"$h/J = 1.0$ (Critical)")

    # Gapped (h/J=2.0)
    gap_hi = load_json(files["gap_high"])
    rho_hi = chord_distance(gap_hi["r_values"], L)
    I_hi = np.asarray(gap_hi["I_r_values"], float)
    rho_hi, I_hi = select_valid(rho_hi, I_hi, floor=MI_FLOOR, x_positive=True)
    markevery_hi = max(1, len(rho_hi)//10) if len(rho_hi) > 0 else 1
    ax.plot(rho_hi, I_hi, "--", lw=1.2, color=COL["gapped_hi"],
            marker="s", ms=3.4, mec='none', markevery=markevery_hi,
            label=r"$h/J = 2.0$ (Gapped)")

    # Gapped (h/J=0.5)
    gap_lo = load_json(files["gap_low"])
    rho_lo = chord_distance(gap_lo["r_values"], L)
    I_lo = np.asarray(gap_lo["I_r_values"], float)
    rho_lo, I_lo = select_valid(rho_lo, I_lo, floor=MI_FLOOR, x_positive=True)
    markevery_lo = max(1, len(rho_lo)//10) if len(rho_lo) > 0 else 1
    ax.plot(rho_lo, I_lo, "-.", lw=1.2, color=COL["gapped_lo"],
            marker="^", ms=3.8, mec='none', markevery=markevery_lo,
            label=r"$h/J = 0.5$ (Gapped)")

    # Reference: I ∝ ρ^{-2} anchored near ρ ≈ 20
    if len(rho_c) > 0:
        idx = np.argmin(np.abs(rho_c - 20.0))
        C = I_c[idx] * (rho_c[idx] ** 2)
        rho_ref = np.logspace(np.log10(max(1.0, rho_c.min()*0.9)),
                              np.log10(rho_c.max()*1.1), 100)
        ax.plot(rho_ref, C / (rho_ref**2), ":", lw=1.2, color=COL["ref"],
                label=r"$I \propto \rho^{-2}$")

    # Cosmetics
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Chord Distance $\rho(r)$")
    ax.set_ylabel(r"Mutual Information $I(r)$ [nats]")
    ax.legend(loc="upper right", ncol=1, handlelength=2.2)
    ax.set_xlim(left=1)

    # Log ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(LogLocator(base=10.0, numticks=10))
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=50))
        axis.set_major_formatter(LogFormatterMathtext())

    ax.tick_params(which="major", length=5, width=1.0)
    ax.tick_params(which="minor", length=2.5, width=0.8)


# ----------------------------
# Panel (b): FSS collapse with percent-deviation inset
# ----------------------------
def plot_panel_b(ax, files, x_max=0.40, bulk_low=0.02):
    sizes = [512, 1024, 2048]
    xy_by_L = {}

    # Load and scale each size
    for L in sizes:
        data = load_json(files[L])
        r = np.asarray(data["r_values"], float)
        I = np.asarray(data["I_r_values"], float)

        dE = compute_dE(I, mi_floor=MI_FLOOR)
        x = r / L
        y = dE / L

        x, y = select_valid(x, y, x_positive=True)
        if len(x) == 0:
            continue

        m = (x <= x_max)
        if np.any(m):
            # sort to ensure clean lines
            order = np.argsort(x[m])
            xy_by_L[L] = (x[m][order], y[m][order])

    if not xy_by_L:
        return

    # Light band to indicate bulk scaling window (optional but helpful)
    ax.axvspan(bulk_low, x_max, color="#000000", alpha=0.04, zorder=0)

    # Plot each curve with distinct colors & markers
    for L in sizes:
        if L not in xy_by_L:
            continue
        x, y = xy_by_L[L]
        color = FSS_COLS[L]
        ax.plot(x, y, "-", lw=1.6, color=color, zorder=2)
        # sparse markers to avoid clutter
        markevery = max(1, len(x)//16)
        ax.plot(x[::markevery], y[::markevery], linestyle="None",
                marker=FSS_MARKERS[L], ms=4.4, color=color, zorder=3,
                label=fr"$L={L}$")

    ax.set_xlabel(r"$r/L$")
    ax.set_ylabel(r"Scaled Emergent Distance $d_E(r)/L$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    # Legend (largest L on top)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[::-1], labels[::-1], loc="upper left",
                  title=r"$h/J = 1.0$ (critical)")

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=5, width=1.0)
    ax.tick_params(which="minor", length=2.5, width=0.8)

    # --- MODIFICATION START: Activate Inset ---
    L_ref = 2048
    # Ensure the reference L exists, there is data to compare, and the inset library is loaded
    if L_ref in xy_by_L and len(xy_by_L) > 1 and inset_axes is not None:
         # Pass the data; the inset function will determine the effective range.
         add_percent_deviation_inset(ax, xy_by_L, L_ref, bulk_low, x_max)
    # --- MODIFICATION END ---


# ----------------------------------------------------------------------
# MODIFICATION AREA: Optimized Inset Rendering
# ----------------------------------------------------------------------
def add_percent_deviation_inset(ax, xy_by_L, L_ref, x_min, x_max):
    """Inset with fractional deviation δ[%] = 100*(y/y_ref - 1) vs r/L."""
    
    # Configuration for improved clarity
    Y_LIM = 0.25      # +/- 0.25%
    SHADE_BAND = 0.1  # +/- 0.1%
    LW_INSET = 1.6
    
    try:
        # 1. Determine dynamic X range based on reference data extent
        if L_ref not in xy_by_L:
            return
        x_ref_data, _ = xy_by_L[L_ref]
        if len(x_ref_data) == 0:
            return
        # Limit the X axis to where the reference data actually exists (e.g. ~0.125)
        x_max_eff = min(x_max, np.max(x_ref_data))
        
        # Ensure x_min respects the start of the data
        x_min_eff = max(x_min, np.min(x_ref_data))

        if x_max_eff <= x_min_eff:
            return

        # 2. Create inset axes (Slightly smaller dimensions as requested)
        # OLD: width="50%", height="45%"
        # NEW: width="45%", height="40.5%" (10% reduction)
        axins = inset_axes(ax, width="45%", height="40.5%",
                           loc="lower right", borderpad=1.2)
                           
        # 3. Improve Contrast: Make the background slightly opaque
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.85)

        # 4. Build grid (restricted to effective range) and interpolate
        x_grid = np.linspace(x_min_eff, x_max_eff, 600)
        Y = {}
        for L, (x, y) in xy_by_L.items():
            order = np.argsort(x)
            # Interpolation is safe here as x_grid is constrained by L_ref extent
            Y[L] = np.interp(x_grid, x[order], y[order], left=np.nan, right=np.nan)

        y_ref = Y[L_ref]

        # 5. Plotting
        # Shading band
        axins.axhspan(-SHADE_BAND, +SHADE_BAND, color="#777777", alpha=0.15, zorder=0)

        # Data lines
        for L in sorted(xy_by_L.keys()):
            if L == L_ref:
                continue
                
            # Robust calculation of deviation
            with np.errstate(divide='ignore', invalid='ignore'):
                dev = 100.0 * (Y[L] / y_ref - 1.0)
            
            color = FSS_COLS.get(L) 

            # Plot line (label is used for the legend)
            axins.plot(x_grid, dev, lw=LW_INSET, color=color, label=fr"$L={L}$")
            
        # Zero line
        axins.axhline(0.0, color="#000000", lw=0.8, ls="--", alpha=0.8)

        # 6. Aesthetics and Configuration
        axins.set_xlim(x_grid.min(), x_grid.max())
        axins.set_ylim(-Y_LIM, +Y_LIM) # Focused Y-axis
        
        # Increased font sizes (from 8/7pt to 9/8pt)
        axins.set_title(fr"Deviation vs $L={L_ref}$ [$\%$]", fontsize=9, pad=4)
        
        # Ticks configuration
        axins.xaxis.set_major_locator(MaxNLocator(4))
        axins.yaxis.set_major_locator(MultipleLocator(0.1)) # Adjusted locator for Y_LIM
        
        # Add minor ticks for better resolution
        axins.xaxis.set_minor_locator(AutoMinorLocator())
        axins.yaxis.set_minor_locator(AutoMinorLocator())

        
        # Ensure PRB style (inward, boxed) and legible ticks
        # Ensure direction='in', top=True, right=True are explicitly set for both major and minor
        axins.tick_params(axis='both', which='both', labelsize=8, length=3.0, width=0.8, pad=1.5, direction='in', top=True, right=True)
        
        axins.grid(True, alpha=0.35, linewidth=0.5, linestyle=":")
        
        # 7. Add Legend
        if axins.get_legend_handles_labels()[0]:
                 # Use a simple frame that matches the background patch
                 axins.legend(loc="upper right", fontsize=8, frameon=True, facecolor='white', edgecolor='lightgray', framealpha=0.9, handlelength=1.5)

    except Exception as e:
        print(f"[WARN] Could not create inset: {e}")

# ----------------------------
# Main
# ----------------------------
def main():
    here = Path(__file__).resolve().parent
    data_dir = here / "simulation_results_tfim_lite"

    files = {
        # panel (a) uses L=2048 sets
        "crit":     data_dir / "lite_results_L2048_H1p0.json",
        "gap_high": data_dir / "lite_results_L2048_H2p0.json",
        "gap_low":  data_dir / "lite_results_L2048_H0p5.json",
        # panel (b) FSS at h/J=1.0
        512:  data_dir / "lite_results_L512_H1p0.json",
        1024: data_dir / "lite_results_L1024_H1p0.json",
        2048: data_dir / "lite_results_L2048_H1p0.json",
    }

    # Figure sized for two-column width in PRB (~7.0 in)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    left, right = axes

    plot_panel_a(left, files)
    plot_panel_b(right, files, x_max=0.40, bulk_low=0.02)

    # MODIFICATION: Changed output filenames to V12
    out_png = here / "Figure_2_V12.png"
    out_pdf = here / "Figure_2_V12.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    main()
