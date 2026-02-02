"""
Paper-grade Matplotlib styling.

Goals:
- Clean, minimal, LaTeX-friendly aesthetics
- Vector export support (PDF/SVG/EPS) and high-res PNG
- Colorblind-friendly defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclass
class PaperStyle:
    """Style configuration for paper-ready static plots."""
    font_family: str = "serif"
    serif_fonts: Sequence[str] = ("Times New Roman", "CMU Serif", "STIXGeneral", "DejaVu Serif")
    base_fontsize: int = 11
    title_fontsize: int = 13
    label_fontsize: int = 11
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    line_width: float = 2.0
    axes_line_width: float = 0.8
    grid_alpha: float = 0.18
    figure_dpi: int = 300
    save_dpi: int = 600
    use_grid: bool = True
    grid_which: str = "major"
    grid_axis: str = "both"
    grid_style: str = "-"
    grid_linewidth: float = 0.6

    def to_rcparams(self) -> Dict[str, Any]:
        return {
            "font.family": self.font_family,
            "font.serif": list(self.serif_fonts),
            "font.size": self.base_fontsize,
            "axes.titlesize": self.title_fontsize,
            "axes.labelsize": self.label_fontsize,
            "xtick.labelsize": self.tick_fontsize,
            "ytick.labelsize": self.tick_fontsize,
            "legend.fontsize": self.legend_fontsize,
            "lines.linewidth": self.line_width,
            "axes.linewidth": self.axes_line_width,
            "figure.dpi": self.figure_dpi,
            "savefig.dpi": self.save_dpi,
            "axes.unicode_minus": False,
        }


def set_paper_style(style: Optional[PaperStyle] = None) -> PaperStyle:
    """
    Apply paper-grade Matplotlib rcParams.

    Returns:
        Applied PaperStyle.
    """
    if style is None:
        style = PaperStyle()

    plt.rcParams.update(style.to_rcparams())

    # A clean default color cycle (Matplotlib tab10 is colorblind-friendly)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=plt.get_cmap("tab10").colors)

    return style


def finish_axes(ax: plt.Axes, style: Optional[PaperStyle] = None) -> None:
    """Apply finishing touches to an Axes."""
    if style is None:
        style = PaperStyle()

    # Minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Grid
    if style.use_grid:
        ax.grid(True, which=style.grid_which, axis=style.grid_axis,
                linestyle=style.grid_style, linewidth=style.grid_linewidth, alpha=style.grid_alpha)

    # Ticks outward like many papers
    ax.tick_params(direction="out", length=4, width=0.8)


def save_figure(fig: plt.Figure, path: str, *, tight: bool = True, transparent: bool = False) -> None:
    """
    Save a figure with paper-friendly defaults.

    Args:
        fig: Matplotlib Figure.
        path: Output path (extension decides format: .pdf/.png/.svg/.eps).
        tight: Whether to use tight bounding box.
        transparent: Transparent background.
    """
    kwargs = {}
    if tight:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = 0.02

    fig.savefig(path, transparent=transparent, **kwargs)
