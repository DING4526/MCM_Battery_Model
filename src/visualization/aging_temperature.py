"""
Aging–Temperature interaction plot (single-panel version).

Features:
- Scatter: TTL vs device age (subsampled for clarity)
- Point color: average battery temperature (blue -> red)
- LOWESS regression curves by temperature band
- Single figure (no usage faceting)

Current design:
- Two temperature bands split at 37°C
- Scatter subsampling (default: 500 points)
- Blue-to-red colormap for temperature
- Clean legend & colorbar placement (no occlusion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

from .style import set_paper_style, finish_axes, save_figure, PaperStyle
from .utils import to_numeric_series


# ============================================================
# Config
# ============================================================

@dataclass
class AgingTemperatureSingleConfig:
    # Columns
    age_col: str = "device_age_months"
    ttl_col: str = "TTL_hours"
    temp_col: str = "avg_battery_temp_celsius"

    # Temperature bands (°C): split at 37°C
    temp_bins: Sequence[float] = (-np.inf, 35.0, np.inf)
    temp_labels: Sequence[str] = ("<35°C", "≥35°C")

    # Scatter control
    max_points: int = 1500          # subsample size (paper-friendly)
    point_size: float = 22.0
    alpha: float = 0.8

    # LOWESS
    lowess_frac: float = 0.35

    # Figure
    figsize: Tuple[float, float] = (6.8, 4.8)

    # Style
    style: Optional[PaperStyle] = None


# ============================================================
# Plot
# ============================================================

def plot_aging_temperature_interaction_single(
    df: pd.DataFrame,
    *,
    cfg: Optional[AgingTemperatureSingleConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot TTL vs device age with temperature modulation (single panel).
    """
    if cfg is None:
        cfg = AgingTemperatureSingleConfig()

    set_paper_style(cfg.style)

    # ---------- sanity check ----------
    for c in (cfg.age_col, cfg.ttl_col, cfg.temp_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    d = df.copy()
    d[cfg.age_col] = to_numeric_series(d[cfg.age_col])
    d[cfg.ttl_col] = to_numeric_series(d[cfg.ttl_col])
    d[cfg.temp_col] = to_numeric_series(d[cfg.temp_col])

    d = d.dropna(subset=[cfg.age_col, cfg.ttl_col, cfg.temp_col])

    # ---------- subsample scatter points ----------
    if len(d) > cfg.max_points:
        d_scatter = d.sample(cfg.max_points, random_state=42)
    else:
        d_scatter = d

    # ---------- temperature bands ----------
    d["temp_band"] = pd.cut(
        d[cfg.temp_col],
        bins=list(cfg.temp_bins),
        labels=list(cfg.temp_labels),
        include_lowest=True,
        right=False,
    )

    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Scatter: continuous temperature (blue -> red)
    sc = ax.scatter(
        d_scatter[cfg.age_col],
        d_scatter[cfg.ttl_col],
        c=d_scatter[cfg.temp_col],
        cmap="coolwarm",
        s=cfg.point_size,
        alpha=cfg.alpha,
        linewidths=0,
    )

    # Fixed colors for temperature bands (mechanism clarity)
    band_colors = {
        "<37°C": "#2B6CB0",   # cool blue
        "≥37°C": "#C53030",   # hot red
    }

    # LOWESS curves per temperature band
    for band in cfg.temp_labels:
        sub = d[d["temp_band"] == band]
        if sub.shape[0] < 30:
            continue

        x = sub[cfg.age_col].to_numpy()
        y = sub[cfg.ttl_col].to_numpy()

        sm = lowess(
            y, x,
            frac=cfg.lowess_frac,
            return_sorted=True,
        )

        ax.plot(
            sm[:, 0],
            sm[:, 1],
            linewidth=2.6,
            color=band_colors.get(str(band)),
            label=str(band),
        )

    # ---------- axes ----------
    ax.set_xlabel("Device Age (months)")
    ax.set_ylabel("Battery Lifetime (hours)")
    ax.set_title("Aging–Temperature Interaction on Battery Lifetime")
    finish_axes(ax, cfg.style)

    # ---------- legend: temperature bands ----------
    ax.legend(
        title="Temperature Band",
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#444444",
    )

    # ---------- colorbar: continuous temperature ----------
    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.02,
    )
    cbar.set_label("Average Battery Temperature (°C)")

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig
