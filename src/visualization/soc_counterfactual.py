# src/visualization/soc_counterfactual.py
"""
SOC counterfactual (ablation) visualization for a single device timeseries JSON.

Design rules (per your request):
- SOC curves: solid lines
- Error curves: dashed lines
- SOC and its corresponding error share the SAME color
- Uncorrected is baseline: only SOC(uncorrected) solid; no paired error

X-axis:
- Use the longest trajectory among all curves as x-axis
- Other curves disappear once reaching SOC ~ 0

Fields read from JSON:
  SOC (fully corrected), SOC_uncorrected, SOC_voltage_only, SOC_temperature_only, SOC_aging_only
  time is optional; if missing, infer from meta.dt or assume dt=1s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import numpy as np
import matplotlib.pyplot as plt

from .style import set_paper_style, finish_axes, save_figure, PaperStyle


@dataclass
class SOCCounterfactualConfig:
    # Wider figure to avoid main panel being squeezed by right legend
    figsize: Tuple[float, float] = (8.8, 4.2)

    title: str = "Counterfactual SOC Trajectories"
    xlabel: str = "Time (hours)"
    ylabel: str = "SOC"
    soc_ylim: Tuple[float, float] = (0.0, 1.0)

    # baseline used for all error curves
    baseline: Literal["uncorrected", "full"] = "uncorrected"

    # Grey fill: baseline vs FULL corrected
    show_error_fill: bool = True
    error_fill_alpha: float = 0.18
    error_fill_label: str = "Cumulative error (baseline vs full)"

    # Right axis: signed errors (allow negative)
    show_error_axis: bool = True
    error_ylabel: str = "SOC error (signed)"

    # Curves stop after reaching SOC~0
    stop_at_zero: bool = True
    soc_zero_eps: float = 1e-6

    # Legend placement
    legend_outside: bool = True

    # Styling
    style: Optional[PaperStyle] = None


# ----------------------------
# Helpers
# ----------------------------
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _infer_dt_sec(payload: Dict[str, Any]) -> float:
    meta = payload.get("meta", {}) or {}
    dt = meta.get("dt", None)
    if dt is not None:
        try:
            dt = float(dt)
            if np.isfinite(dt) and dt > 0:
                return dt
        except Exception:
            pass

    t = _to_np(payload.get("time", []))
    if t.size >= 2:
        d = np.diff(t)
        d = d[np.isfinite(d)]
        if d.size:
            dt2 = float(np.median(d))
            if np.isfinite(dt2) and dt2 > 0:
                return dt2

    return 1.0


def _time_from_dt(n: int, dt_sec: float) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=float)
    return (np.arange(n, dtype=float) * dt_sec) / 3600.0


def _trim_at_zero(t_hr: np.ndarray, soc: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    if t_hr.size == 0 or soc.size == 0:
        return t_hr[:0], soc[:0]
    L = min(t_hr.size, soc.size)
    t_hr = t_hr[:L]
    soc = soc[:L]
    idx = np.where(soc <= eps)[0]
    if idx.size == 0:
        return t_hr, soc
    cut = int(idx[0]) + 1
    return t_hr[:cut], soc[:cut]


def _min_len(*arrs: np.ndarray) -> int:
    sizes = [a.size for a in arrs if a is not None]
    return int(min(sizes)) if sizes else 0


# ----------------------------
# Main plot
# ----------------------------
def plot_soc_counterfactual(
    json_path: str,
    *,
    cfg: Optional[SOCCounterfactualConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    if cfg is None:
        cfg = SOCCounterfactualConfig()

    set_paper_style(cfg.style)
    payload = _load_json(json_path)

    soc_full = _to_np(payload.get("SOC", []))
    soc_unc = _to_np(payload.get("SOC_uncorrected", []))
    soc_v = _to_np(payload.get("SOC_voltage_only", []))
    soc_t = _to_np(payload.get("SOC_temperature_only", []))
    soc_a = _to_np(payload.get("SOC_aging_only", []))

    if soc_full.size == 0:
        raise ValueError(f"Missing 'SOC' (fully corrected) in {json_path}.")
    if soc_unc.size == 0 or soc_v.size == 0 or soc_t.size == 0 or soc_a.size == 0:
        raise ValueError(
            f"Missing SOC_* arrays in {json_path}. "
            "Need SOC_uncorrected/SOC_voltage_only/SOC_temperature_only/SOC_aging_only."
        )

    dt_sec = _infer_dt_sec(payload)

    t_full = _time_from_dt(soc_full.size, dt_sec)
    t_unc = _time_from_dt(soc_unc.size, dt_sec)
    t_v = _time_from_dt(soc_v.size, dt_sec)
    t_t = _time_from_dt(soc_t.size, dt_sec)
    t_a = _time_from_dt(soc_a.size, dt_sec)

    if cfg.stop_at_zero:
        t_full, soc_full = _trim_at_zero(t_full, soc_full, cfg.soc_zero_eps)
        t_unc, soc_unc = _trim_at_zero(t_unc, soc_unc, cfg.soc_zero_eps)
        t_v, soc_v = _trim_at_zero(t_v, soc_v, cfg.soc_zero_eps)
        t_t, soc_t = _trim_at_zero(t_t, soc_t, cfg.soc_zero_eps)
        t_a, soc_a = _trim_at_zero(t_a, soc_a, cfg.soc_zero_eps)

    # longest x-axis
    x_max = 0.0
    for tt in (t_full, t_unc, t_v, t_t, t_a):
        if tt.size:
            x_max = max(x_max, float(tt[-1]))
    if x_max <= 0:
        raise ValueError(f"No valid samples to plot in {json_path} (all curves empty after trimming).")

    # baseline for all error curves
    if cfg.baseline == "uncorrected":
        base_t, base_soc = t_unc, soc_unc
        base_tag = "unc"
    else:
        base_t, base_soc = t_full, soc_full
        base_tag = "full"

    # --- fixed color mapping (SOC solid + error dashed, same color) ---
    # uncorrected baseline uses black (only SOC)
    col_unc = "black"
    col_full = "#1f77b4"   # tab:blue
    col_v = "#ff7f0e"      # tab:orange
    col_t = "#2ca02c"      # tab:green
    col_a = "#d62728"      # tab:red

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # SOC curves (ALL solid)
    ax.plot(t_full, soc_full, label="SOC (fully corrected)", linewidth=2.4, linestyle="-", color=col_full)
    ax.plot(t_unc, soc_unc, label="SOC (uncorrected)", linewidth=2.2, linestyle="-", color=col_unc)
    ax.plot(t_v, soc_v, label="SOC (voltage only)", linewidth=2.0, linestyle="-", color=col_v)
    ax.plot(t_t, soc_t, label="SOC (temperature only)", linewidth=2.0, linestyle="-", color=col_t)
    ax.plot(t_a, soc_a, label="SOC (aging only)", linewidth=2.0, linestyle="-", color=col_a)

    # Grey fill: baseline vs FULL corrected (aligned on common prefix length)
    if cfg.show_error_fill:
        L_fill = _min_len(base_soc, soc_full)
        if L_fill > 1:
            ax.fill_between(
                base_t[:L_fill],
                base_soc[:L_fill],
                soc_full[:L_fill],
                alpha=cfg.error_fill_alpha,
                linewidth=0.0,
                color="#B0B0B0",
                label=cfg.error_fill_label,
                zorder=0,
            )

    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    ax.set_ylim(*cfg.soc_ylim)
    ax.set_xlim(0.0, x_max)

    # Right axis: 3 signed error curves (ALL dashed; same color as corresponding SOC)
    ax2 = None
    if cfg.show_error_axis:
        ax2 = ax.twinx()

        # error definitions: SOC_baseline - SOC_variant
        L_ev = _min_len(base_soc, soc_v)
        L_et = _min_len(base_soc, soc_t)
        L_ea = _min_len(base_soc, soc_a)

        if L_ev > 1:
            ax2.plot(
                base_t[:L_ev],
                (base_soc[:L_ev] - soc_v[:L_ev]),
                linewidth=1.6,
                linestyle="--",
                color=col_v,
                alpha=0.95,
                label="error (vs voltage-only)",
            )
        if L_et > 1:
            ax2.plot(
                base_t[:L_et],
                (base_soc[:L_et] - soc_t[:L_et]),
                linewidth=1.6,
                linestyle="--",
                color=col_t,
                alpha=0.95,
                label="error (vs temperature-only)",
            )
        if L_ea > 1:
            ax2.plot(
                base_t[:L_ea],
                (base_soc[:L_ea] - soc_a[:L_ea]),
                linewidth=1.6,
                linestyle="--",
                color=col_a,
                alpha=0.95,
                label="error (vs aging-only)",
            )

        ax2.set_ylabel(cfg.error_ylabel)
        ax2.tick_params(axis="y", colors="black")
        ax2.yaxis.label.set_color("black")
        ax2.spines["top"].set_visible(False)
        ax2.axhline(0.0, linewidth=1.0, alpha=0.35, color="black")

    if cfg.title:
        ax.set_title(cfg.title)

    finish_axes(ax, cfg.style)

    # Legend spacing: avoid overlapping ticks
    legend_kw = dict(
        frameon=True,
        fancybox=False,
        edgecolor="#888888",
        fontsize=9,
        labelspacing=0.45,
        borderpad=0.60,
        handletextpad=0.65,
        borderaxespad=0.40,
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))

    if cfg.legend_outside:
        ax.legend(
            h1 + h2,
            l1 + l2,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            **legend_kw,
        )
        # More right margin so main plot doesn't get squeezed
        plt.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    else:
        ax.legend(
            h1 + h2,
            l1 + l2,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            **legend_kw,
        )
        plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()

    return fig
