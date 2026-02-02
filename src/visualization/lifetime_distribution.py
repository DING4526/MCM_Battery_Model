"""
Population Lifetime Distribution plots (paper-grade static).

Implements:
- KDE (Kernel Density Estimate)
- ECDF (Empirical CDF)

Stratification modes:
- none
- by device_age_months (age groups via add_age_group_column)
- by usage_dominant_state

Design goals:
- Publication-ready (clean, balanced, LaTeX-friendly)
- English labels by default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, Dict, Any, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import set_paper_style, finish_axes, save_figure, PaperStyle
from .utils import validate_dataframe, ecdf, robust_xlim, add_age_group_column, order_categories

try:
    from scipy.stats import gaussian_kde
except Exception as e:  # pragma: no cover
    gaussian_kde = None


StratifyMode = Literal["none", "device_age_months", "usage_dominant_state"]
PlotKind = Literal["kde", "ecdf"]


@dataclass
class LifetimePlotConfig:
    ttl_col: str = "ttl_hours"
    device_age_col: str = "device_age_months"
    usage_state_col: str = "usage_dominant_state"

    # For device-age stratification
    age_group_col: str = "age_group"
    age_bins: Optional[Sequence[float]] = None
    age_labels: Optional[Sequence[str]] = None

    # KDE settings
    kde_points: int = 512
    kde_cut: float = 0.04  # relative padding on x-range for KDE curve

    # Figure settings
    figsize: Tuple[float, float] = (6.6, 4.4)
    title_prefix: str = "Population Lifetime Distribution"

    # Legend
    legend_title_age: str = "Device Age"
    legend_title_usage: str = "Usage State"

    # Category ordering / limiting
    max_levels_usage: Optional[int] = None  # e.g. 8 for papers
    usage_order: Optional[Sequence[str]] = None
    age_order: Optional[Sequence[str]] = None

    # Optional vertical reference lines (hours)
    vlines_hours: Optional[Sequence[float]] = None
    vlines_label: str = "Reference"

    # Theme
    style: Optional[PaperStyle] = None


def _ensure_scipy():
    if gaussian_kde is None:
        raise ImportError(
            "scipy is required for KDE plots but was not found. "
            "Please install scipy, or use ECDF plots."
        )


def _kde_curve(values: np.ndarray, *, points: int, cut: float) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_scipy()
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([]), np.array([])

    xmin, xmax = robust_xlim(x, pad_ratio=cut)
    grid = np.linspace(xmin, xmax, points)
    kde = gaussian_kde(x)
    dens = kde(grid)
    return grid, dens


def _plot_reference_lines(ax: plt.Axes, cfg: LifetimePlotConfig) -> None:
    if not cfg.vlines_hours:
        return
    for i, v in enumerate(cfg.vlines_hours):
        label = cfg.vlines_label if i == 0 else None
        ax.axvline(v, linestyle="--", linewidth=1.2, alpha=0.6, label=label)


def plot_population_kde(
    df: pd.DataFrame,
    *,
    cfg: Optional[LifetimePlotConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    KDE plot without stratification.
    """
    if cfg is None:
        cfg = LifetimePlotConfig()
    set_paper_style(cfg.style)

    validate_dataframe(df, ttl_col=cfg.ttl_col)
    ttl = pd.to_numeric(df[cfg.ttl_col], errors="coerce").dropna().values

    grid, dens = _kde_curve(ttl, points=cfg.kde_points, cut=cfg.kde_cut)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    if grid.size:
        ax.plot(grid, dens)
        ax.fill_between(grid, dens, alpha=0.22)

    _plot_reference_lines(ax, cfg)

    ax.set_xlabel("Battery Lifetime (hours)")
    ax.set_ylabel("Probability Density")
    ax.set_title(title or f"{cfg.title_prefix} (KDE)")
    finish_axes(ax, cfg.style)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_population_ecdf(
    df: pd.DataFrame,
    *,
    cfg: Optional[LifetimePlotConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    ECDF plot without stratification.
    """
    if cfg is None:
        cfg = LifetimePlotConfig()
    set_paper_style(cfg.style)

    validate_dataframe(df, ttl_col=cfg.ttl_col)
    ttl = pd.to_numeric(df[cfg.ttl_col], errors="coerce").dropna().values

    x, y = ecdf(ttl)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    if x.size:
        ax.step(x, y, where="post")

    _plot_reference_lines(ax, cfg)

    ax.set_xlabel("Battery Lifetime (hours)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title or f"{cfg.title_prefix} (ECDF)")
    ax.set_ylim(0, 1.0)
    finish_axes(ax, cfg.style)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_kde_stratified(
    df: pd.DataFrame,
    *,
    stratify: StratifyMode,
    cfg: Optional[LifetimePlotConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    KDE plot with stratification.

    stratify:
        - "device_age_months": will create age_group column and stratify by it
        - "usage_dominant_state": categorical stratification
    """
    if cfg is None:
        cfg = LifetimePlotConfig()
    set_paper_style(cfg.style)

    validate_dataframe(
        df,
        ttl_col=cfg.ttl_col,
        strata_cols=[
            cfg.device_age_col if stratify == "device_age_months" else cfg.usage_state_col
        ],
    )

    df2 = df.copy()

    if stratify == "device_age_months":
        df2 = add_age_group_column(
            df2,
            age_col=cfg.device_age_col,
            out_col=cfg.age_group_col,
            bins=cfg.age_bins,
            labels=cfg.age_labels,
        )
        group_col = cfg.age_group_col
        legend_title = cfg.legend_title_age
        order = cfg.age_order
    elif stratify == "usage_dominant_state":
        group_col = cfg.usage_state_col
        legend_title = cfg.legend_title_usage
        order = cfg.usage_order
    else:
        raise ValueError("Stratified KDE requires stratify != 'none'.")

    # Determine category order
    levels = order_categories(df2, group_col, order=order,
                              max_levels=cfg.max_levels_usage if group_col == cfg.usage_state_col else None)

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Global x-range to keep curves comparable
    ttl_all = pd.to_numeric(df2[cfg.ttl_col], errors="coerce").dropna().values
    xmin, xmax = robust_xlim(ttl_all, pad_ratio=cfg.kde_cut)
    xgrid = np.linspace(xmin, xmax, cfg.kde_points)

    for lvl in levels:
        g = df2[df2[group_col] == lvl]
        ttl = pd.to_numeric(g[cfg.ttl_col], errors="coerce").dropna().values
        if ttl.size < 3:
            continue
        _ensure_scipy()
        kde = gaussian_kde(ttl)
        dens = kde(xgrid)
        ax.plot(xgrid, dens, label=str(lvl))

    _plot_reference_lines(ax, cfg)

    ax.set_xlabel("Battery Lifetime (hours)")
    ax.set_ylabel("Probability Density")

    if title:
        ax.set_title(title)
    else:
        if stratify == "device_age_months":
            ax.set_title("Lifetime Distribution by Device Age (KDE)")
        else:
            ax.set_title("Lifetime Distribution by Dominant Usage State (KDE)")

    ax.legend(title=legend_title, frameon=False)
    finish_axes(ax, cfg.style)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_ecdf_stratified(
    df: pd.DataFrame,
    *,
    stratify: StratifyMode,
    cfg: Optional[LifetimePlotConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    ECDF plot with stratification.
    """
    if cfg is None:
        cfg = LifetimePlotConfig()
    set_paper_style(cfg.style)

    validate_dataframe(
        df,
        ttl_col=cfg.ttl_col,
        strata_cols=[
            cfg.device_age_col if stratify == "device_age_months" else cfg.usage_state_col
        ],
    )

    df2 = df.copy()

    if stratify == "device_age_months":
        df2 = add_age_group_column(
            df2,
            age_col=cfg.device_age_col,
            out_col=cfg.age_group_col,
            bins=cfg.age_bins,
            labels=cfg.age_labels,
        )
        group_col = cfg.age_group_col
        legend_title = cfg.legend_title_age
        order = cfg.age_order
    elif stratify == "usage_dominant_state":
        group_col = cfg.usage_state_col
        legend_title = cfg.legend_title_usage
        order = cfg.usage_order
    else:
        raise ValueError("Stratified ECDF requires stratify != 'none'.")

    levels = order_categories(df2, group_col, order=order,
                              max_levels=cfg.max_levels_usage if group_col == cfg.usage_state_col else None)

    fig, ax = plt.subplots(figsize=cfg.figsize)

    for lvl in levels:
        g = df2[df2[group_col] == lvl]
        ttl = pd.to_numeric(g[cfg.ttl_col], errors="coerce").dropna().values
        if ttl.size == 0:
            continue
        x, y = ecdf(ttl)
        ax.step(x, y, where="post", label=str(lvl))

    _plot_reference_lines(ax, cfg)

    ax.set_xlabel("Battery Lifetime (hours)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.0)

    if title:
        ax.set_title(title)
    else:
        if stratify == "device_age_months":
            ax.set_title("ECDF by Device Age")
        else:
            ax.set_title("ECDF by Dominant Usage State")

    ax.legend(title=legend_title, frameon=False)
    finish_axes(ax, cfg.style)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_population_lifetime_distribution(
    df: pd.DataFrame,
    *,
    kind: PlotKind = "kde",
    stratify: StratifyMode = "none",
    cfg: Optional[LifetimePlotConfig] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Unified entry point.

    Args:
        df: DataFrame with ttl_hours and optional stratification columns.
        kind: "kde" or "ecdf"
        stratify: "none" | "device_age_months" | "usage_dominant_state"
        cfg: LifetimePlotConfig
        title: override title
        save_path: if provided, save figure to this path
        show: show figure

    Returns:
        Matplotlib Figure.
    """
    if kind == "kde":
        if stratify == "none":
            return plot_population_kde(df, cfg=cfg, title=title, save_path=save_path, show=show)
        return plot_kde_stratified(df, stratify=stratify, cfg=cfg, title=title, save_path=save_path, show=show)

    if kind == "ecdf":
        if stratify == "none":
            return plot_population_ecdf(df, cfg=cfg, title=title, save_path=save_path, show=show)
        return plot_ecdf_stratified(df, stratify=stratify, cfg=cfg, title=title, save_path=save_path, show=show)

    raise ValueError("kind must be 'kde' or 'ecdf'")
