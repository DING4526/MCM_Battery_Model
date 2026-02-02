"""
Active usage composition plots (paper-grade).

Implements:
- Stacked bar chart of usage composition (idle / social / video / gaming)
  grouped by Active Usage Intensity.

Design goals:
- Compact, paper-friendly size (suitable as a secondary / supporting figure)
- Clear visual separation between idle (background) and active states
- Compatible with existing PaperStyle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import set_paper_style, finish_axes, save_figure, PaperStyle
from .utils import to_numeric_series


@dataclass
class UsageCompositionConfig:
    # Columns
    group_col: str = "usage_intensity_group"
    idle_col: str = "ratio_idle"
    social_col: str = "ratio_social"
    video_col: str = "ratio_video"
    game_col: str = "ratio_game"

    # Figure appearance
    figsize: Tuple[float, float] = (5.6, 3.6)   # compact by default
    bar_width: float = 0.55                     # narrower bars (paper-like)

    # Labels
    title: str = "Composition of Usage by Intensity Group"
    xlabel: str = "Active Usage Intensity Group"
    ylabel: str = "Average Usage Ratio"

    # Category order (important for narrative consistency)
    order: Optional[Sequence[str]] = ("low-active", "mid-active", "high-active")

    # Style
    style: Optional[PaperStyle] = None


def plot_active_composition_stacked_bar(
    df: pd.DataFrame,
    *,
    cfg: Optional[UsageCompositionConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot stacked bar chart showing mean usage composition
    (idle / social / video / gaming) within each intensity group.
    """
    if cfg is None:
        cfg = UsageCompositionConfig()

    set_paper_style(cfg.style)

    # --- sanity check ---
    for c in (
        cfg.group_col,
        cfg.social_col,
        cfg.video_col,
        cfg.game_col,
        cfg.idle_col,
    ):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --- numeric conversion ---
    d = df.copy()
    d[cfg.idle_col] = to_numeric_series(d[cfg.idle_col])
    d[cfg.social_col] = to_numeric_series(d[cfg.social_col])
    d[cfg.video_col] = to_numeric_series(d[cfg.video_col])
    d[cfg.game_col] = to_numeric_series(d[cfg.game_col])

    # --- group-wise mean composition ---
    comp = (
        d.groupby(cfg.group_col)[
            [cfg.social_col, cfg.video_col, cfg.game_col, cfg.idle_col]
        ]
        .mean()
        .dropna(how="all")
    )

    # --- stable ordering: low / mid / high ---
    if cfg.order is not None:
        idx = [
            x for x in cfg.order
            if x in comp.index.astype(str).tolist() or x in comp.index.tolist()
        ]
        if idx:
            try:
                comp = comp.loc[idx]
            except Exception:
                comp = comp.reindex(idx)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=cfg.figsize)

    x = np.arange(len(comp.index))
    bottom = np.zeros(len(comp.index), dtype=float)

    # Stacking order: idle at bottom (background), then active states
    series = [
        # Idle as neutral background
        (cfg.social_col, "Social", {}),
        (cfg.video_col, "Video", {}),
        (cfg.game_col, "Gaming", {}),
        (cfg.idle_col, "Idle", {"color": "#D9D9D9"}),
    ]

    for col, label, kwargs in series:
        vals = comp[col].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)

        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=cfg.bar_width,
            label=label,
            **kwargs,
        )
        bottom += vals

    # --- axes & labels ---
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in comp.index])
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    ax.set_title(cfg.title)
    ax.set_ylim(0.0, 1.0)

    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#444444",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
    )
    finish_axes(ax, cfg.style)
    ax.grid(True, axis="y", alpha=0.18)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig
