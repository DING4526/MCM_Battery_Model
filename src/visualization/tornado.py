# src/visualization/tornado.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .style import set_paper_style, finish_axes, save_figure, PaperStyle


@dataclass
class TornadoConfig:
    figsize: Tuple[float, float] = (9.6, 6.2)  # 横向更宽，适合长标签
    title: str = "Tornado Chart"
    xlabel: str = "Δ Metric"
    ylabel: str = ""

    # 是否按影响力排序
    sort_by_impact: bool = True

    # 视觉
    show_zero_line: bool = True
    value_fmt: str = "{:+.3f}"
    annotate: bool = True
    bar_height: float = 0.72

    # styling
    style: Optional[PaperStyle] = None


def plot_tornado(
    effects: Dict[str, Tuple[float, float]],
    *,
    cfg: Optional[TornadoConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    effects:
      { param_name: (low_effect, high_effect) }
    effect 通常是 ΔTTL_hours 或 %ΔTTL 等（围绕 0）
    """
    cfg = cfg or TornadoConfig()
    set_paper_style(cfg.style)

    names = list(effects.keys())
    lows = np.array([effects[k][0] for k in names], dtype=float)
    highs = np.array([effects[k][1] for k in names], dtype=float)

    impact = np.maximum(np.abs(lows), np.abs(highs))

    if cfg.sort_by_impact:
        order = np.argsort(-impact)  # descending
        names = [names[i] for i in order]
        lows = lows[order]
        highs = highs[order]
        impact = impact[order]

    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # 颜色：不要硬编码颜色风格的话，可以用默认循环；但 tornado 需要 low/high 区分
    # 这里用 matplotlib 默认色循环的前两个（不指定具体色值）
    low_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    high_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]

    ax.barh(y, lows, height=cfg.bar_height, label="Low", color=low_color, alpha=0.9)
    ax.barh(y, highs, height=cfg.bar_height, label="High", color=high_color, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    ax.set_xlabel(cfg.xlabel)
    if cfg.ylabel:
        ax.set_ylabel(cfg.ylabel)
    if cfg.title:
        ax.set_title(cfg.title)

    if cfg.show_zero_line:
        ax.axvline(0.0, linewidth=1.0)

    # annotate
    if cfg.annotate:
        # 给文本一点偏移，防止贴条边缘
        xspan = (np.nanmax(np.abs(np.r_[lows, highs])) + 1e-9)
        dx = 0.02 * xspan

        for yi, lo, hi in zip(y, lows, highs):
            if np.isfinite(lo) and abs(lo) > 1e-12:
                ax.text(lo + (dx if lo >= 0 else -dx), yi, cfg.value_fmt.format(lo),
                        va="center", ha="left" if lo >= 0 else "right", fontsize=9)
            if np.isfinite(hi) and abs(hi) > 1e-12:
                ax.text(hi + (dx if hi >= 0 else -dx), yi, cfg.value_fmt.format(hi),
                        va="center", ha="left" if hi >= 0 else "right", fontsize=9)

    ax.legend(loc="lower right", frameon=True, fontsize=9)

    finish_axes(ax, cfg.style)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig
