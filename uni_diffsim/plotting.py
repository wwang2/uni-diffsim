"""Shared plotting utilities for uni-diffsim.

Provides consistent Nord-inspired, editorial styling across all demos and scripts.
"""

import matplotlib.pyplot as plt


# Publication-ready figure sizes (in inches)
FIG_WIDTH_SINGLE = 3.25  # Single column (e.g. Science/Nature/NeurIPS half-width)
FIG_WIDTH_DOUBLE = 6.75  # Double column (full width)
GOLDEN_RATIO = (5**0.5 - 1) / 2

# Font sizes
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 8

# Standard line width and marker size
LW = 1.5
MS = 4

# Nord-inspired, editorial plotting style
PLOT_STYLE = {
    "font.family": "monospace",
    "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Monaco"],
    "font.size": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "axes.labelpad": 5.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.9",
    "figure.facecolor": "#FAFBFC",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FAFBFC",
    "lines.linewidth": LW,
}

# Shared color palette (Nord-inspired)
COLORS = {
    # Primary colors
    "blue": "#5E81AC",        # Steel blue
    "orange": "#D08770",      # Warm orange
    "green": "#A3BE8C",       # Sage green
    "red": "#BF616A",         # Muted red
    "purple": "#B48EAD",      # Lavender
    "cyan": "#88C0D0",        # Cyan
    "gray": "#4C566A",        # Slate gray
    
    # Semantic aliases
    "overdamped": "#5E81AC",
    "baoab": "#D08770",
    "gle": "#A3BE8C",
    "nh": "#BF616A",          # Nosé-Hoover
    "esh": "#B48EAD",
    "verlet": "#4C566A",
    "nhc": "#88C0D0",         # Nosé-Hoover Chain
    
    # Gradient method colors
    "bptt": "#D08770",
    "reinforce": "#5E81AC",
    "implicit": "#A3BE8C",
    "girsanov": "#B48EAD",
    "adjoint": "#5E81AC",
    "forward": "#A3BE8C",
    
    # Status colors
    "theory": "#4C566A",
    "target": "#BF616A",
    "optimal": "#B48EAD",
    "invalid": "#BF616A",
    "error": "#BF616A",
    
    # UI colors
    "fill": "#E5E9F0",
    "trajectory": "#88C0D0",
    "neural": "#88C0D0",
    "barrier": "#D08770",
}


def apply_style():
    """Apply the shared plotting style to matplotlib."""
    plt.rcParams.update(PLOT_STYLE)


def get_figsize(width, nrows=1, ncols=1, aspect=None):
    """
    Calculate figure size based on width and subplot grid.

    Args:
        width (float): Figure width in inches (use FIG_WIDTH_SINGLE or FIG_WIDTH_DOUBLE).
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        aspect (float, optional): Target aspect ratio (height/width) per subplot.
                                  Defaults to GOLDEN_RATIO.

    Returns:
        tuple: (width, height) in inches.
    """
    if aspect is None:
        aspect = GOLDEN_RATIO

    height = width * (nrows / ncols) * aspect
    return (width, height)


def get_assets_dir():
    """Get the assets directory path relative to this module."""
    import os
    return os.path.join(os.path.dirname(__file__), "..", "assets")

