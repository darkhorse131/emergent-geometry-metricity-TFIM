# -*- coding: utf-8 -*-
"""
make_figure_1_V19.py

Generates the final, PRB-quality Figure 1 with a broken Y-axis in Panel (a).
This version aligns the aesthetics (lineweights, fonts, colors, PRB ticks) with Figure 2.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

# ----------------------------
# Matplotlib style tuned for APS journals (PRB/PRL) - Adopted from Figure 2
# ----------------------------
# We rely entirely on rcParams, removing external style sheets (like seaborn-paper).
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
    # Font sizes adopted from Figure 2
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    # Grid style adopted from Figure 2
    "axes.grid": True,
    "grid.color": "#999999",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.55,
    # PRB/PRL style ticks (inward facing, on all sides)
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    # Defining tick lengths and widths (matching Figure 2 implementation)
    "xtick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.major.size": 5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
})

# --- Color Palette (Aligned with Figure 2) ---
COLORS = {
    'critical': "#E69F00",  # Orange
    'gapped':   "#D55E00",  # Vermillion/Red-Orange
    'ref':      "#333333",  # Dark gray (primary reference lines)
    'gray':     "#999999",  # Lighter gray (secondary reference lines/grid)
    'shading':  "#EAEAEA"   # Light gray for inset shading
}

# --- Line Weights (Compromise between Fig 1 delicacy and Fig 2 weight) ---
LW_DATA = 1.5
LW_REF_PRIMARY = 1.2
LW_REF_SECONDARY = 1.0

# ----------------------------
# Data Loading & Helpers
# ----------------------------

def chord_distance(r: np.ndarray, L: int) -> np.ndarray:
    """Computes the chord distance rho(r) = (L/pi) * sin(pi*r/L)."""
    return (L / np.pi) * np.sin(np.pi * r / L)

def load_json_data(path: Path) -> dict:
    """Loads a JSON file."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading JSON file: {path}")
        return None

def select_valid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filters out non-finite values from paired arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

# ----------------------------
# Plotting Functions
# ----------------------------

def plot_panel_a_broken_axis(fig, crit_data, gap_data):
    """Plots Panel (a) using two subplots to create a broken y-axis."""
    
    # Define height ratios and create gridspec with minimal hspace
    H_RATIOS = [1, 2]
    # hspace=0.1 provides a small gap for the break marks
    gs = fig.add_gridspec(2, 1, height_ratios=H_RATIOS, hspace=0.1)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    if crit_data is None or gap_data is None:
        return ax_top, ax_bottom

    # --- Load and process data ---
    Lc = crit_data['L']
    r_c = np.asarray(crit_data['r_values'])
    # Compute dE from MI if available (using the recorded MI floor); otherwise fall back to stored d_E_r_values
    if 'I_r_values' in crit_data:
        I_c = np.asarray(crit_data['I_r_values'], float)
        mi_floor_c = crit_data.get('fit', {}).get('mi_floor_used', 1e-12)
        dE_c = np.full_like(I_c, np.nan, dtype=float)
        m_c = np.isfinite(I_c) & (I_c > mi_floor_c)
        dE_c[m_c] = 1.0 / np.sqrt(I_c[m_c])
    else:
        dE_c = np.asarray(crit_data['d_E_r_values'])
    rho_c = chord_distance(r_c, Lc)
    rho_c, dE_c = select_valid(rho_c, dE_c)

    Lg = gap_data['L']
    r_g = np.asarray(gap_data['r_values'])
    if 'I_r_values' in gap_data:
        I_g = np.asarray(gap_data['I_r_values'], float)
        mi_floor_g = gap_data.get('fit', {}).get('mi_floor_used', 1e-12)
        dE_g = np.full_like(I_g, np.nan, dtype=float)
        m_g = np.isfinite(I_g) & (I_g > mi_floor_g)
        dE_g[m_g] = 1.0 / np.sqrt(I_g[m_g])
    else:
        dE_g = np.asarray(gap_data['d_E_r_values'])
    rho_g = chord_distance(r_g, Lg)
    rho_g, dE_g = select_valid(rho_g, dE_g)
    
    rho_line = np.linspace(0, 250, 500)

    # --- Plot on both axes ---
    for ax in [ax_top, ax_bottom]:
        ax.plot(rho_c, dE_c, color=COLORS['critical'], linestyle='-', linewidth=LW_DATA)
        ax.plot(rho_g, dE_g, color=COLORS['gapped'], linestyle='--', linewidth=LW_DATA)
        # Primary reference line uses 'ref' color (dark gray)
        ax.plot(rho_line, rho_line, color=COLORS['ref'], linestyle=':', linewidth=LW_REF_PRIMARY)
        
        # Add minor ticks (styling handled by rcParams)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        # Use fewer minor ticks on Y axis (2) to reduce clutter near the break
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # --- Configure axis limits and spines at the break ---
    ax_top.set_ylim(40000, 750000)
    ax_bottom.set_ylim(0, 2000)

    # Hide spines at the interface
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)

    # Hide ticks/labels at the interface
    # Top plot: hide bottom labels and bottom ticks. Ensure top ticks remain active (PRB style).
    ax_top.tick_params(labelbottom=False, bottom=False, top=True)
    # Bottom plot: hide top ticks (to prevent overlap at the break).
    ax_bottom.tick_params(top=False)

    # --- Add break marks (diagonal lines) ---
    d_x = .015  # Horizontal length in normalized axis coordinates
    # Use the axis linewidth for the break marks
    kwargs = dict(color='k', clip_on=False, linewidth=plt.rcParams["axes.linewidth"])
    
    # To ensure the visual angle of the break marks is the same, we must adjust
    # the vertical component (d_y) for the aspect ratio difference between the plots.
    # Ratio of (Bottom Height / Top Height) = H_RATIOS[1] / H_RATIOS[0] = 2/1 = 2.
    v_stretch_top = H_RATIOS[1] / H_RATIOS[0]
    d_y_top = d_x * v_stretch_top
    d_y_bottom = d_x # For the bottom plot, the ratio is 1.

    # Top axis (slanted downwards)
    trans_top = ax_top.transAxes
    ax_top.plot((-d_x, +d_x), (-d_y_top, +d_y_top), transform=trans_top, **kwargs)
    ax_top.plot((1 - d_x, 1 + d_x), (-d_y_top, +d_y_top), transform=trans_top, **kwargs)

    # Bottom axis (slanted upwards)
    trans_bottom = ax_bottom.transAxes
    ax_bottom.plot((-d_x, +d_x), (1 - d_y_bottom, 1 + d_y_bottom), transform=trans_bottom, **kwargs)
    ax_bottom.plot((1 - d_x, 1 + d_x), (1 - d_y_bottom, 1 + d_y_bottom), transform=trans_bottom, **kwargs)

    # --- Labels, Annotations, and Legend ---
    ax_bottom.set_xlabel(r'Chord Distance $\rho(r)$')
    
    # Calculate rotation angle
    angle = 40 # Default angle
    try:
        x1_data, x2_data = 75, 175
        y1_data = np.interp(x1_data, rho_c, dE_c)
        y2_data = np.interp(x2_data, rho_c, dE_c)
        
        # We must draw the canvas to ensure transforms are updated for this complex layout
        # 'fig' here refers to the subfigure object passed into the function
        fig.canvas.draw()
        
        x1_disp, y1_disp = ax_bottom.transData.transform((x1_data, y1_data))
        x2_disp, y2_disp = ax_bottom.transData.transform((x2_data, y2_data))
        if np.isfinite(x1_disp) and np.isfinite(x2_disp) and np.isfinite(y1_disp) and np.isfinite(y2_disp):
            angle = np.degrees(np.arctan2(y2_disp - y1_disp, x2_disp - x1_disp))
    except Exception as e:
        print(f"Warning: Could not calculate annotation angle ({e}). Using default.")


    # Annotations (Fontsize 9 matches legend font size)
    ax_bottom.text(13, 950, 'Emergent Distance $d_E$ (Gapped)', rotation=90, va='center', ha='left', fontsize=9)
    ax_bottom.text(125, 450, 'Emergent Distance $d_E$ (Critical)', ha='center', va='center', fontsize=9, rotation=angle - 1.5)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['critical'], lw=LW_DATA, label=r'$h/J = 1.0$ (Critical)'),
        Line2D([0], [0], color=COLORS['gapped'], lw=LW_DATA, linestyle='--', label=r'$h/J = 2.0$ (Gapped)'),
        Line2D([0], [0], color=COLORS['ref'], lw=LW_REF_PRIMARY, linestyle=':', label=r'$d_E \propto \rho$ (Euclidean ref.)')
    ]

    ax_bottom.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98))
    
    return ax_top, ax_bottom

def plot_panel_b(ax, crit_data, gap_data):
    """Plots Panel (b): Triangle Defect with Inset."""
    
    if crit_data is None or gap_data is None:
        return

    # --- Load and process data ---
    Lc = crit_data['L']
    r_c_del, delta_c = np.asarray(crit_data['delta_rr']['r_values']), np.asarray(crit_data['delta_rr']['values'])
    rho_c_del = chord_distance(r_c_del, Lc)
    rho_c_del, delta_c = select_valid(rho_c_del, delta_c)

    Lg = gap_data['L']
    r_g_del, delta_g = np.asarray(gap_data['delta_rr']['r_values']), np.asarray(gap_data['delta_rr']['values'])
    rho_g_del = chord_distance(r_g_del, Lg)
    rho_g_del, delta_g = select_valid(rho_g_del, delta_g)

    # --- Plotting ---
    MSIZE = 4
    ax.plot(rho_c_del, delta_c, color=COLORS['critical'], linestyle='-', marker='.', markersize=MSIZE, lw=LW_DATA, label=r'$h/J = 1.0$ (Critical)')
    ax.plot(rho_g_del, delta_g, color=COLORS['gapped'], linestyle='--', lw=LW_DATA, label=r'$h/J = 2.0$ (Gapped)')
    
    # Use lighter 'gray' color for the Delta=0 line (secondary reference)
    ax.axhline(0.0, linestyle='--', color=COLORS['gray'], linewidth=LW_REF_SECONDARY, label=r'$\Delta = 0$')

    # --- Configuration ---
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_xlim(0, 100)
    ax.set_xlabel(r'Chord Distance $\rho(r)$')
    ax.set_ylabel(r'$\Delta(r,r) = d_E(2r) - 2d_E(r)$')
    ax.legend(loc='lower right')
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # Note: Symlog minor ticks are handled automatically by matplotlib.

    # --- Inset ---
    ax_in = ax.inset_axes([0.55, 0.58, 0.4, 0.35])
    
    # Strategic Grey Shading
    ax_in.set_facecolor(COLORS['shading'])
    ax_in.patch.set_alpha(0.5) # Subtle shading
    
    # Slightly thinner lines and smaller markers in inset
    ax_in.plot(rho_c_del, delta_c, color=COLORS['critical'], marker='.', markersize=MSIZE*0.8, lw=LW_DATA*0.8)
    ax_in.axhline(0.0, linestyle='--', color=COLORS['gray'], linewidth=LW_REF_SECONDARY)
    
    ax_in.set_xlim(0, 40); ax_in.set_ylim(-5, 5)
    ax_in.set_xticks([0, 10, 20, 30, 40])
    
    # Inset styling aligned with FIG 2 aesthetics
    ax_in.set_title('Critical Regime Zoom', fontsize=9, pad=3)
    
    # Ensure grid is enabled (inherits style from rcParams)
    ax_in.grid(True, which='major')
    
    # Smaller tick labels for inset, ensure PRB style (inward, boxed)
    ax_in.tick_params(axis='both', which='major', labelsize=8)
    # Ensure the inset ticks follow the global settings (inward, top/right)
    ax_in.tick_params(axis='both', direction='in', top=True, right=True)
    ax_in.xaxis.set_minor_locator(AutoMinorLocator())
    ax_in.yaxis.set_minor_locator(AutoMinorLocator())


# ----------------------------
# Main Driver
# ----------------------------

def main():
    """Main function to generate and save the complete Figure 1."""
    
    # Define paths
    # Handle execution environments where __file__ might not be defined (e.g., interactive consoles)
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path(".").resolve()
        
    data_dir = here / "simulation_results_tfim_lite"
    
    crit_file = data_dir / "lite_results_L2048_H1p0.json"
    gap_file = data_dir / "lite_results_L2048_H2p0.json"

    crit_data = load_json_data(crit_file)
    gap_data = load_json_data(gap_file)

    # If data is missing, generate dummy data to allow aesthetic review of the plot structure
    if crit_data is None or gap_data is None:
        print(f"Warning: Data files not found or could not be read in {data_dir}.")
        print("Using dummy data to generate plot structure for aesthetic review.")
        # Create dummy data representative of the expected behavior
        L_dummy = 2048
        r_dummy = np.arange(1, L_dummy//4)
        rho_dummy = chord_distance(r_dummy, L_dummy)
        
        # Approximate behaviors
        dE_crit_dummy = 1.6 * rho_dummy**0.99
        # Exponential behavior that breaks the axis scale
        dE_gap_dummy = 50 * np.exp(rho_dummy/8)
        
        # Simplified delta calculation for dummy data visualization
        r_del_dummy = r_dummy[:len(r_dummy)//2]
        rho_del_dummy = rho_dummy[:len(r_dummy)//2]
        delta_crit_dummy = -0.5 * rho_del_dummy**0.5
        delta_gap_dummy = 10 * dE_gap_dummy[:len(r_dummy)//2]

        crit_data = {'L': L_dummy, 'r_values': r_dummy.tolist(), 'd_E_r_values': dE_crit_dummy.tolist(), 
                     'delta_rr': {'r_values': r_del_dummy.tolist(), 'values': delta_crit_dummy.tolist()}}
        gap_data = {'L': L_dummy, 'r_values': r_dummy.tolist(), 'd_E_r_values': dE_gap_dummy.tolist(),
                    'delta_rr': {'r_values': r_del_dummy.tolist(), 'values': delta_gap_dummy.tolist()}}

    # --- Create Composite Figure ---
    # Optimized size for PRB two-column width (7.0 inches) and suitable height
    fig = plt.figure(figsize=(7.0, 3.8), dpi=300)
    
    # Use gridspec for main layout, setting wspace for separation between panels
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)

    # Panel A uses a subfigure to manage the broken axis layout internally
    fig_a_sub = fig.add_subfigure(gs[0])
    # Pass the subfigure container to the plotting function
    plot_panel_a_broken_axis(fig_a_sub, crit_data, gap_data)

    # Panel B is a standard subplot
    ax_b = fig.add_subplot(gs[1])
    plot_panel_b(ax_b, crit_data, gap_data)
    
    # Note: We do not use fig.tight_layout() as manual gridspec management is required for broken axes.
    
    # --- Save Figure ---
    output_filename = "Figure_1_V19"
    try:
        fig.savefig(here / f"{output_filename}.png", bbox_inches='tight')
        fig.savefig(here / f"{output_filename}.pdf", bbox_inches='tight')
        print(f"✨ Successfully generated and saved '{output_filename}.png' and '.pdf'")
    except Exception as e:
        print(f"Error saving figure: {e}")

if __name__ == "__main__":
    main()