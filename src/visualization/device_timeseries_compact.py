# src/visualization/device_timeseries_compact.py
"""
Compact single-device dynamics figure (paper-grade, shared-x):

Panel A (top): SOC(t) + T_b(t) in one axes (twin y-axis)
Panel P (thin): total power line (separate row; no overlay on heatmap)
Panel H (middle): module-power heatmap (Screen/CPU/Radio/BG)
Panel C (bottom, very thin): state color band (RLE segments)

Reads:
timeseries/<Device_ID>.json
  ├── time, SOC, Tb, Power, State
  └── Power_screen, Power_cpu, Power_radio, Power_background (optional but recommended)

Design goals:
- Tight and publication-friendly
- Highly interpretable causality chain:
  State → Module power patterns → Temperature response → SOC depletion
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from .style import set_paper_style, finish_axes, save_figure, PaperStyle


def _create_red_gradient_cmap() -> LinearSegmentedColormap:
    """
    Create a custom red gradient colormap:
    low values = light pink/white, high values = deep red.
    """
    colors = [
        (1.0, 0.95, 0.93),   # very light pink/white (low)
        (0.99, 0.85, 0.80),  # light pink
        (0.98, 0.65, 0.55),  # light red/salmon
        (0.90, 0.40, 0.30),  # medium red
        (0.75, 0.20, 0.15),  # darker red
        (0.55, 0.08, 0.05),  # deep red (high)
    ]
    return LinearSegmentedColormap.from_list("red_gradient", colors, N=256)


@dataclass
class DeviceTimeseriesCompactConfig:
    # ---- figure layout ----
    figsize: Tuple[float, float] = (8.5, 4.8)  # wider to accommodate right legend
    # A (SOC+Tb), P (total power thin), H (heatmap), C (state band)
    height_ratios: Tuple[float, float, float, float] = (2.2, 0.38, 1.05, 0.22)
    hspace: float = 0.08

    # ---- labels ----
    title: str = ""  # no title, keep clean
    xlabel: str = "Time (hours)"

    # ---- Panel A: SOC + Tb ----
    soc_ylim: Tuple[float, float] = (0.0, 1.0)

    # SOC fill
    soc_fill_alpha: float = 0.12

    # Temperature axis
    tb_ylim: Tuple[float, float] = (10.0, 45.0)  # fixed range 10-45°C
    tb_line_color: str = "#C0392B"   # red-ish for the line
    tb_axis_color: str = "black"     # black for axis ticks and labels

    # Low SOC band - disabled by default to avoid background fill issue
    show_low_soc_band: bool = False
    low_soc_threshold: float = 0.20

    # ---- Panel P: total power ----
    total_power_ylabel: Optional[str] = None  # set to "Total Power (W)" if you want it visible
    total_power_linewidth: float = 2.0

    # ---- Panel H: power heatmap ----
    normalize_power: str = "device_global"  # "device_global" only (keeps cross-module comparability)
    power_norm_percentile: float = 99.0
    heatmap_interpolation: str = "nearest"
    heatmap_aspect: str = "auto"
    heatmap_cmap: str = "red_gradient"  # custom red gradient colormap
    heatmap_vmin: float = 0.0
    heatmap_vmax: float = 1.0
    heatmap_cbar_label: str = "Power (0–1)"  # shortened label
    show_power_yticks: bool = True

    # ---- Panel C: state band ----
    state_band_use_rle: bool = True

    # ---- legend placement ----
    panelA_legend_outside: bool = True
    legend_on_right: bool = True  # place legends on the right side

    # ---- style ----
    style: Optional[PaperStyle] = None


# ----------------------------
# Helpers
# ----------------------------
def _load_timeseries_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _rle_segments(states: List[str]) -> List[Tuple[int, int, str]]:
    """
    Return run-length segments: (start_idx, end_idx_exclusive, state)
    """
    if not states:
        return []
    segs = []
    s0 = states[0]
    start = 0
    for i in range(1, len(states)):
        if states[i] != s0:
            segs.append((start, i, s0))
            start = i
            s0 = states[i]
    segs.append((start, len(states), s0))
    return segs


def _nice_limits(x: np.ndarray, pad: float = 0.04) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    d = hi - lo
    return lo - pad * d, hi + pad * d


def _default_state_palette() -> Dict[str, str]:
    # Keep IDLE neutral and others distinct.
    # You can change these if you already have a canonical scheme.
    return {
        "IDLE": "#9AA0A6",
        "SOCIAL": "#4C78A8",
        "VIDEO": "#F58518",
        "GAME": "#E45756",
        # If your controller emits other tokens:
        "DeepIdle": "#9AA0A6",
        "Social": "#4C78A8",
        "Video": "#F58518",
        "Gaming": "#E45756",
        "Navigation": "#72B7B2",
    }


def _module_labels() -> List[str]:
    return ["Screen", "CPU", "Radio", "Background"]


def _time_edges(t: np.ndarray) -> np.ndarray:
    """
    Build bin edges from sample-centered time array t (monotonic).
    Used to align imshow(extent) and RLE axvspan segments exactly.
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return t
    if t.size == 1:
        return np.array([t[0], t[0] + 1.0], dtype=float)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        dt = float(t[-1] - t[0]) / max(1, (t.size - 1))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
    return np.concatenate([t[:1], t[1:], [t[-1] + dt]])


# ----------------------------
# Main plot
# ----------------------------
def plot_device_timeseries_compact(
    json_path: str,
    *,
    cfg: Optional[DeviceTimeseriesCompactConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    if cfg is None:
        cfg = DeviceTimeseriesCompactConfig()

    set_paper_style(cfg.style)

    payload = _load_timeseries_json(json_path)

    # ---- core signals ----
    t_sec = _to_np(payload.get("time", []))
    soc = _to_np(payload.get("SOC", []))
    tb_k = _to_np(payload.get("Tb", []))
    p_total = _to_np(payload.get("Power", []))
    states = payload.get("State", [])

    if t_sec.size == 0 or soc.size == 0 or tb_k.size == 0 or p_total.size == 0:
        raise ValueError(f"Timeseries missing required arrays in: {json_path}")

    # align length safely
    T = int(min(t_sec.size, soc.size, tb_k.size, p_total.size, len(states)))
    t_sec = t_sec[:T]
    soc = soc[:T]
    tb_k = tb_k[:T]
    p_total = p_total[:T]
    states = states[:T]

    t_hr = t_sec / 3600.0
    tb_c = tb_k - 273.15
    t_edges = _time_edges(t_hr)
    
    # Define actual data range for consistent x-axis limits
    x_min = float(t_hr[0]) if len(t_hr) > 0 else 0.0
    x_max = float(t_hr[-1]) if len(t_hr) > 0 else 1.0

    # ---- module power (heatmap) ----
    p_screen = _to_np(payload.get("Power_screen", np.full(T, np.nan)))[:T]
    p_cpu = _to_np(payload.get("Power_cpu", np.full(T, np.nan)))[:T]
    p_radio = _to_np(payload.get("Power_radio", np.full(T, np.nan)))[:T]
    p_bg = _to_np(payload.get("Power_background", np.full(T, np.nan)))[:T]

    have_modules = (
        np.isfinite(p_screen).any()
        and np.isfinite(p_cpu).any()
        and np.isfinite(p_radio).any()
        and np.isfinite(p_bg).any()
    )

    # ---- figure layout with space for right legend ----
    fig = plt.figure(figsize=cfg.figsize)
    
    # Reserve right margin for legends and colorbar
    if cfg.legend_on_right:
        # Main plot area: left=0.08 to right=0.78
        # Legend area: ~0.80 to ~0.90
        # Colorbar area: 0.92 to 0.935
        gs = fig.add_gridspec(
            nrows=4,
            ncols=1,
            height_ratios=list(cfg.height_ratios),
            hspace=cfg.hspace,
            left=0.08,
            right=0.78,  # leave more space on right for legends + colorbar
            top=0.95,
            bottom=0.10,
        )
    else:
        gs = fig.add_gridspec(
            nrows=4,
            ncols=1,
            height_ratios=list(cfg.height_ratios),
            hspace=cfg.hspace,
        )

    axA = fig.add_subplot(gs[0, 0])
    axP = fig.add_subplot(gs[1, 0], sharex=axA)  # total power (thin)
    axH = fig.add_subplot(gs[2, 0], sharex=axA)  # heatmap
    axC = fig.add_subplot(gs[3, 0], sharex=axA)  # state band

    # Title - only show if not empty (device ID removed to avoid overlap)
    if cfg.title:
        axA.set_title(cfg.title, pad=6)

    # =========================
    # Panel A: SOC + Tb
    # =========================
    axA.plot(t_hr, soc, linewidth=2.2, label="SOC")
    if cfg.soc_fill_alpha and cfg.soc_fill_alpha > 0:
        axA.fill_between(t_hr, 0.0, soc, alpha=cfg.soc_fill_alpha, linewidth=0)

    axA.set_ylim(*cfg.soc_ylim)
    axA.set_ylabel("SOC")

    # Low SOC band - only apply to data range, not beyond
    if cfg.show_low_soc_band:
        mask = soc <= cfg.low_soc_threshold
        if mask.any():
            idx0 = int(np.argmax(mask))
            # Use the actual data range, not t_edges[-1] which extends beyond data
            x_end = t_hr[-1] if len(t_hr) > 0 else t_edges[-1]
            axA.axvspan(t_edges[idx0], x_end, alpha=0.10, linewidth=0)

    # Temperature on twin axis
    axA2 = axA.twinx()
    axA2.plot(
        t_hr,
        tb_c,
        linewidth=2.0,
        linestyle="--",
        color=cfg.tb_line_color,
        label="Tb (°C)",
    )

    # Fixed temperature range 10-45°C
    axA2.set_ylim(*cfg.tb_ylim)
    axA2.set_ylabel("Tb (°C)", color=cfg.tb_axis_color)
    axA2.tick_params(axis="y", colors=cfg.tb_axis_color)

    # Legend for Panel A - place on right side outside plot
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA2.get_legend_handles_labels()
    if cfg.legend_on_right:
        axA.legend(
            h1 + h2,
            l1 + l2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fancybox=False,
            edgecolor="#888888",
            fontsize=9,
        )
    elif cfg.panelA_legend_outside:
        axA.legend(
            h1 + h2,
            l1 + l2,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=True,
            fancybox=False,
            edgecolor="#444444",
        )
    else:
        axA.legend(
            h1 + h2,
            l1 + l2,
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#444444",
        )

    finish_axes(axA, cfg.style)
    axA2.spines["top"].set_visible(False)
    
    # Set consistent x-axis limits for Panel A
    axA.set_xlim(x_min, x_max)

    # Hide x tick labels for top panels
    plt.setp(axA.get_xticklabels(), visible=False)

    # =========================
    # Panel P: Total power (separate thin row)
    # =========================
    axP.plot(t_hr, p_total, linewidth=cfg.total_power_linewidth, color="#2E86AB", label="Power")
    if cfg.total_power_ylabel:
        axP.set_ylabel(cfg.total_power_ylabel)
    else:
        axP.set_ylabel("")
        axP.set_yticks([])

    if cfg.legend_on_right:
        axP.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fancybox=False,
            edgecolor="#888888",
            fontsize=9,
        )
    else:
        axP.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#444444",
        )
    finish_axes(axP, cfg.style)
    axP.set_xlim(x_min, x_max)
    plt.setp(axP.get_xticklabels(), visible=False)

    # =========================
    # Panel H: Module heatmap
    # =========================
    axH.set_ylabel("")

    # Determine colormap
    if cfg.heatmap_cmap == "red_gradient":
        cmap = _create_red_gradient_cmap()
    else:
        cmap = cfg.heatmap_cmap

    if have_modules:
        M = np.vstack([p_screen, p_cpu, p_radio, p_bg])  # (4, T)
        finite = M[np.isfinite(M)]
        if finite.size == 0:
            M_norm = np.zeros_like(M)
            p_ref = 1.0
        else:
            p_ref = float(np.percentile(finite, cfg.power_norm_percentile))
            if p_ref <= 1e-9:
                p_ref = float(np.nanmax(finite)) if np.nanmax(finite) > 0 else 1.0
            M_norm = np.clip(M / p_ref, 0.0, 1.0)

        im = axH.imshow(
            M_norm,
            aspect=cfg.heatmap_aspect,
            interpolation=cfg.heatmap_interpolation,
            extent=[x_min, x_max, 0, 4],  # use actual data range
            origin="lower",
            cmap=cmap,
            vmin=cfg.heatmap_vmin,
            vmax=cfg.heatmap_vmax,
        )

        if cfg.show_power_yticks:
            axH.set_yticks([0.5, 1.5, 2.5, 3.5])
            axH.set_yticklabels(_module_labels())
        else:
            axH.set_yticks([])

        for y in [1, 2, 3]:
            axH.axhline(y, linewidth=1.0, alpha=0.18)

        # Colorbar - place in a dedicated axes on the far right
        # This prevents the colorbar from taking space from the heatmap
        if cfg.legend_on_right:
            pos = axH.get_position()

            cbar_w = 0.015   # colorbar 宽度
            gap = 0.010      # 和热力图之间的间距（关键调这个）
            cax_left = pos.x1 + gap

            cax = fig.add_axes([cax_left, pos.y0, cbar_w, pos.height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(cfg.heatmap_cbar_label, fontsize=9)
        else:
            cbar = fig.colorbar(im, ax=axH, pad=0.012, fraction=0.045)
            cbar.set_label(cfg.heatmap_cbar_label, fontsize=9)
    else:
        axH.text(
            0.5,
            0.5,
            "Module power breakdown not available",
            transform=axH.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.85,
        )
        axH.set_yticks([])
        axH.set_xlim(t_edges[0], t_edges[-1])

    finish_axes(axH, cfg.style)
    axH.set_xlim(x_min, x_max)
    plt.setp(axH.get_xticklabels(), visible=False)

    # =========================
    # Panel C: State band (thin, aligned with edges)
    # =========================
    palette = _default_state_palette()

    axC.set_yticks([])
    axC.set_ylabel("")
    axC.set_xlabel(cfg.xlabel)
    axC.set_ylim(0.0, 1.0)

    if cfg.state_band_use_rle:
        segs = _rle_segments([str(s) for s in states])
        for a, b, st in segs:
            # Clamp segment bounds to actual data range to avoid overflow
            x0 = max(t_edges[a] if a < len(t_edges) else x_min, x_min)
            x1 = min(t_edges[b] if b < len(t_edges) else x_max, x_max)
            color = palette.get(st, "#BBBBBB")
            axC.axvspan(x0, x1, ymin=0.0, ymax=1.0, color=color, linewidth=0)
    else:
        uniq = sorted(set(str(s) for s in states))
        mapping = {u: i for i, u in enumerate(uniq)}
        arr = np.array([mapping[str(s)] for s in states], dtype=int)[None, :]
        from matplotlib.colors import ListedColormap

        cmap_state = ListedColormap([palette.get(u, "#BBBBBB") for u in uniq])
        axC.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            extent=[x_min, x_max, 0, 1],  # use actual data range
            origin="lower",
            cmap=cmap_state,
        )

    # state legend (small color patches) - place on right side
    legend_order = ["IDLE", "SOCIAL", "VIDEO", "GAME", "DeepIdle", "Social", "Video", "Gaming", "Navigation"]
    present = [s for s in legend_order if s in set(str(x) for x in states)]
    extras = [s for s in sorted(set(str(x) for x in states)) if s not in present]
    legend_states = present + extras

    handles = [Patch(facecolor=palette.get(s, "#BBBBBB"), edgecolor="none", label=s) for s in legend_states]
    if handles:
        if cfg.legend_on_right:
            axC.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                ncol=1,
                frameon=True,
                fancybox=False,
                edgecolor="#888888",
                fontsize=9,
                handlelength=1.0,
                handletextpad=0.4,
            )
        else:
            axC.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.85),
                ncol=min(5, len(handles)),
                frameon=True,
                fancybox=False,
                edgecolor="#444444",
                columnspacing=0.8,
                handlelength=1.0,
                handletextpad=0.4,
            )

    # clean look: no spines
    axC.spines["left"].set_visible(False)
    axC.spines["right"].set_visible(False)
    axC.spines["top"].set_visible(False)
    axC.tick_params(direction="out", length=3, width=0.8)

    # Set consistent x-axis limits for Panel C
    axC.set_xlim(x_min, x_max)

    if not cfg.legend_on_right:
        plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig