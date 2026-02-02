"""
Power Structure Boxplot (paper-grade, single panel)

Figure: "Who consumes energy?"
- X: modules (Screen / CPU / Radio / Background)
- Y: energy share (ratio of total energy, 0~1)
- Boxplot shows distribution (median/IQR, whiskers by percentiles)
- Overlay scatter (subsampled + jitter) for richness
- Mean marker (diamond)

Expected columns in summary.csv:
- energy_ratio_screen
- energy_ratio_cpu
- energy_ratio_radio
- energy_ratio_background
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import set_paper_style, finish_axes, save_figure, PaperStyle
from .utils import to_numeric_series


@dataclass
class PowerStructureBoxplotConfig:
    # Columns (energy ratios)
    col_screen: str = "energy_ratio_screen"
    col_cpu: str = "energy_ratio_cpu"
    col_radio: str = "energy_ratio_radio"
    col_bg: str = "energy_ratio_background"

    # Display labels (x-axis)
    labels: Sequence[str] = ("Screen", "CPU", "Radio", "Background")

    # Figure
    figsize: Tuple[float, float] = (6.0, 3.6)
    title: str = "Who Consumes Energy? (Energy Share by Module)"
    ylabel: str = "Energy Share (ratio of total energy)"

    # Boxplot robustness
    show_fliers: bool = False  # outliers are shown via scatter anyway
    whisker_percentiles: Tuple[float, float] = (5.0, 95.0)
    box_width: float = 0.62
    median_linewidth: float = 2.2
    whisker_linewidth: float = 1.25
    cap_linewidth: float = 1.25

    # Scatter overlay (subsample + jitter)
    max_points_per_group: int = 250
    jitter: float = 0.27
    point_size: float = 18.0
    point_alpha: float = 0.33
    scatter_seed: int = 31

    # Mean markers
    show_mean: bool = True
    mean_marker_size: float = 56.0
    mean_marker: str = "D"

    # y-limits (robust auto)
    use_robust_ylim: bool = True
    ylim_pad: float = 0.04
    ylim_max_cap: float = 0.70

    # Optional reference line (often unnecessary; keep subtle if on)
    show_half_line: bool = False
    half_line_alpha: float = 0.12

    # Legend
    show_legend: bool = True
    legend_loc: str = "upper right"

    # Style
    style: Optional[PaperStyle] = None


def _pick_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    return to_numeric_series(df[col])


def _subsample_quantile_stratified(
    arr: np.ndarray,
    rng: np.random.Generator,
    max_n: int,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Quantile-stratified subsample to preserve tails and overall shape.
    """
    if arr.size <= max_n:
        return arr

    # If constant-ish, just random sample
    if np.nanstd(arr) < 1e-12:
        idx = rng.choice(arr.size, size=max_n, replace=False)
        return arr[idx]

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(arr, qs)

    chosen = []
    per_bin = max(6, max_n // n_bins)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)

        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue

        take = min(per_bin, idxs.size)
        chosen.append(rng.choice(idxs, size=take, replace=False))

    if not chosen:
        idx = rng.choice(arr.size, size=max_n, replace=False)
        return arr[idx]

    idx = np.concatenate(chosen)
    if idx.size > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return arr[idx]


def plot_power_structure_boxplot(
    df: pd.DataFrame,
    *,
    cfg: Optional[PowerStructureBoxplotConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Single-panel boxplot for energy share distribution across modules.
    Overlay jittered scatter for richer distribution visualization.
    """
    if cfg is None:
        cfg = PowerStructureBoxplotConfig()

    set_paper_style(cfg.style)

    # --------- load & sanitize ----------
    cols = [cfg.col_screen, cfg.col_cpu, cfg.col_radio, cfg.col_bg]
    series_list: List[pd.Series] = []
    for c in cols:
        s = _pick_numeric_column(df, c).dropna()
        s = s.clip(lower=0.0, upper=1.0)
        series_list.append(s)

    if all(len(s) == 0 for s in series_list):
        raise ValueError("All energy ratio columns are empty after numeric conversion.")

    data_arrays = [s.to_numpy(dtype=float) for s in series_list]
    positions = np.arange(1, len(data_arrays) + 1, dtype=float)

    # --------- figure ----------
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # --------- boxplot ----------
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        widths=cfg.box_width,
        whis=cfg.whisker_percentiles,
        showfliers=cfg.show_fliers,
        patch_artist=True,
        medianprops={"linewidth": cfg.median_linewidth},
        whiskerprops={"linewidth": cfg.whisker_linewidth},
        capprops={"linewidth": cfg.cap_linewidth},
    )

    # Subtle fill so points/median remain dominant
    for box in bp["boxes"]:
        box.set_alpha(0.26)

    # --------- overlay scatter + means ----------
    rng = np.random.default_rng(cfg.scatter_seed)

    for i, arr in enumerate(data_arrays):
        if arr.size == 0:
            continue

        y = _subsample_quantile_stratified(arr, rng, cfg.max_points_per_group, n_bins=8)

        x0 = positions[i]
        x = x0 + rng.uniform(-cfg.jitter, cfg.jitter, size=y.size)

        ax.scatter(
            x, y,
            s=cfg.point_size,
            alpha=cfg.point_alpha,
            linewidths=0,
            zorder=2,
        )

        if cfg.show_mean:
            ax.scatter(
                [x0], [float(np.mean(arr))],
                s=cfg.mean_marker_size,
                marker=cfg.mean_marker,
                linewidths=0.0,
                alpha=0.95,
                zorder=4,
            )

    # --------- axes ----------
    ax.set_xticks(positions)
    ax.set_xticklabels(list(cfg.labels))
    ax.set_ylabel(cfg.ylabel, labelpad=6)
    ax.set_title(cfg.title, pad=6)

    # Robust y-limit to remove wasted whitespace
    if cfg.use_robust_ylim:
        all_y = np.concatenate([a for a in data_arrays if a.size > 0], axis=0)
        ymax = float(np.percentile(all_y, 99.5) + cfg.ylim_pad) if all_y.size else 1.0
        ymax = min(ymax, cfg.ylim_max_cap, 1.0)
        ax.set_ylim(0.0, max(0.15, ymax))  # avoid too tiny ymax if data is degenerate
    else:
        ax.set_ylim(0.0, 1.0)

    if cfg.show_half_line:
        ax.axhline(0.5, linestyle="--", linewidth=1.0, alpha=cfg.half_line_alpha, zorder=1)

    finish_axes(ax, cfg.style)

    # --------- legend ----------
    if cfg.show_legend:
        # legend handles (generic + paper-friendly)
        scatter_h = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=6.5,
            alpha=min(0.8, cfg.point_alpha + 0.35),
            label=f"Samples (subsampled, ≤{cfg.max_points_per_group}/module)",
        )
        mean_h = Line2D(
            [0], [0],
            marker=cfg.mean_marker,
            linestyle="None",
            markersize=7.2,
            alpha=0.95,
            label="Mean",
        )
        box_h = Patch(
            alpha=0.26,
            label=f"Box: IQR, Whiskers: P{cfg.whisker_percentiles[0]:g}–P{cfg.whisker_percentiles[1]:g}",
        )

        ax.legend(
            handles=[box_h, scatter_h, mean_h],
            loc=cfg.legend_loc,
            frameon=True,
            fancybox=False,
            edgecolor="#444444",
        )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig
