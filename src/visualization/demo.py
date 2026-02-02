"""
Unified plotting entry for:
01 - Lifetime Distribution

Generates:
- Overall lifetime distribution (KDE / ECDF)
- Stratified by Active Usage Intensity (mainline result)
- Stratified by Device Aging
- Supporting figure: Usage composition within intensity groups
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .lifetime_distribution import (
    plot_population_lifetime_distribution,
    LifetimePlotConfig,
)
from .usage_composition import (
    plot_active_composition_stacked_bar,
    UsageCompositionConfig,
)
from .utils import add_usage_intensity_group


# ------------------------
# Helpers
# ------------------------

def _resolve_ttl_col(df: pd.DataFrame) -> str:
    for col in ("TTL_hours", "ttl_hours"):
        if col in df.columns:
            return col
    raise ValueError(f"No TTL column found. Columns={list(df.columns)}")


def _compute_quantile_thresholds(
    lifetime_series: pd.Series,
    quantile_levels: tuple[float, ...],
) -> list[float]:
    x = pd.to_numeric(lifetime_series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return []
    q = np.asarray(quantile_levels)
    q = q[(q >= 0) & (q <= 1)]
    return [float(v) for v in np.quantile(x, q)]


def _format_threshold_label(quantiles: tuple[float, ...]) -> str:
    p = [int(q * 100) for q in quantiles]
    return "Key Threshold (" + "/".join(f"P{i}" for i in p) + ")"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _pick_representative_timeseries(ts_dir: Path) -> Path | None:
    """
    Pick a representative JSON (deterministic):
    - choose lexicographically smallest file name to keep stable across runs
    """
    if not ts_dir.exists():
        return None
    files = sorted(ts_dir.glob("*.json"))
    return files[0] if files else None

def _list_timeseries_jsons(ts_dir: Path) -> list[Path]:
    if not ts_dir.exists():
        return []
    return sorted(ts_dir.glob("*.json"))

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser("02 - Aging Temperature Interaction Figures")
    parser.add_argument(
        "--summary",
        default=str(_repo_root() / "output/population/summary.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(_repo_root() / "output/04_single_device_visualization"),
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.2, 0.5, 0.8],
    )

    # NEW: timeseries related
    parser.add_argument(
        "--timeseries_dir",
        default=str(_repo_root() / "output/population/timeseries"),
        help="Directory that contains timeseries/<Device_ID>.json files",
    )
    parser.add_argument(
        "--device_json",
        default="",
        help="Optional: specify an exact timeseries json path; if set, it overrides --n_devices.",
    )
    parser.add_argument(
        "--n_devices",
        type=int,
        default=10,
        help="How many devices to render from timeseries_dir (sorted). Default: 10",
    )
    args = parser.parse_args()

    # ------------------------
    # Load data
    # ------------------------
    df = pd.read_csv(args.summary)
    ttl_col = _resolve_ttl_col(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Common reference lines
    # ------------------------
    quantiles = tuple(args.quantiles)
    thresholds = _compute_quantile_thresholds(df[ttl_col], quantiles)
    threshold_label = _format_threshold_label(quantiles)

    # # ============================================================
    # # 0. Overall population
    # # ============================================================
    # cfg_overall = LifetimePlotConfig(
    #     ttl_col=ttl_col,
    #     vlines_hours=thresholds,
    #     vlines_label=threshold_label,
    #     title_prefix="Population Lifetime Distribution",
    # )

    # plot_population_lifetime_distribution(
    #     df, kind="kde", stratify="none",
    #     cfg=cfg_overall,
    #     save_path=str(out / "overall_kde.pdf"),
    #     show=False,
    # )
    # plot_population_lifetime_distribution(
    #     df, kind="ecdf", stratify="none",
    #     cfg=cfg_overall,
    #     save_path=str(out / "overall_ecdf.pdf"),
    #     show=False,
    # )

    # # ============================================================
    # # 1. By Active Usage Intensity (MAIN RESULT)
    # # ============================================================
    # df = add_usage_intensity_group(df)

    # cfg_intensity = LifetimePlotConfig(
    #     ttl_col=ttl_col,
    #     usage_state_col="usage_intensity_group",
    #     legend_title_usage="Active Usage Intensity",
    #     vlines_hours=thresholds,
    #     vlines_label=threshold_label,
    #     title_prefix="Lifetime by Active Usage Intensity",
    # )

    # plot_population_lifetime_distribution(
    #     df, kind="kde", stratify="usage_dominant_state",
    #     cfg=cfg_intensity,
    #     save_path=str(out / "by_intensity_kde.pdf"),
    #     show=False,
    # )
    # plot_population_lifetime_distribution(
    #     df, kind="ecdf", stratify="usage_dominant_state",
    #     cfg=cfg_intensity,
    #     save_path=str(out / "by_intensity_ecdf.pdf"),
    #     show=False,
    # )

    # # ============================================================
    # # 2. By Device Aging (CONTROL / CONTEXT)
    # # ============================================================
    # cfg_age = LifetimePlotConfig(
    #     ttl_col=ttl_col,
    #     device_age_col="device_age_months",
    #     legend_title_age="Device Age",
    #     vlines_hours=thresholds,
    #     vlines_label=threshold_label,
    #     title_prefix="Lifetime by Device Age",
    # )

    # plot_population_lifetime_distribution(
    #     df, kind="kde", stratify="device_age_months",
    #     cfg=cfg_age,
    #     save_path=str(out / "by_age_kde.pdf"),
    #     show=False,
    # )
    # plot_population_lifetime_distribution(
    #     df, kind="ecdf", stratify="device_age_months",
    #     cfg=cfg_age,
    #     save_path=str(out / "by_age_ecdf.pdf"),
    #     show=False,
    # )

    # # ============================================================
    # # 3. Supporting: Usage composition
    # # ============================================================
    # comp_cfg = UsageCompositionConfig(
    #     group_col="usage_intensity_group",
    #     title="Usage Composition within Intensity Groups",
    # )
    # plot_active_composition_stacked_bar(
    #     df,
    #     cfg=comp_cfg,
    #     save_path=str(out / "usage_composition_by_intensity.pdf"),
    #     show=False,
    # )

    # # ============================================================
    # # 4. Aging × Temperature interaction (INSIGHT FIGURE)
    # # ============================================================

    # from .aging_temperature import (
    #     plot_aging_temperature_interaction_single,
    #     AgingTemperatureSingleConfig,
    # )

    # at_cfg = AgingTemperatureSingleConfig(
    #     age_col="device_age_months",
    #     ttl_col=ttl_col,
    #     temp_col="avg_battery_temp_celsius",
    #     max_points=500,
    #     figsize=(6.8, 4.8),
    # )

    # plot_aging_temperature_interaction_single(
    #     df,
    #     cfg=at_cfg,
    #     save_path=str(out / "aging_temperature_interaction_single.pdf"),
    #     show=False,
    # )

    # # ============================================================
    # # 5. Who consumes energy? (Power structure boxplot)
    # # ============================================================

    # from .power_structure_boxplot import (
    #     plot_power_structure_boxplot,
    #     PowerStructureBoxplotConfig,
    # )

    # ps_cfg = PowerStructureBoxplotConfig(
    #     figsize=(6.6, 4.2),
    # )

    # plot_power_structure_boxplot(
    #     df,
    #     cfg=ps_cfg,
    #     save_path=str(out / "who_consumes_energy_boxplot.pdf"),
    #     show=False,
    # )

    # ============================================================
    # Single-device compact dynamics figures (batch)
    # ============================================================
    from .device_timeseries_compact import (
        plot_device_timeseries_compact,
        DeviceTimeseriesCompactConfig,
    )

    ts_dir = Path(args.timeseries_dir)

    if args.device_json.strip():
        ts_paths = [Path(args.device_json.strip())]
    else:
        all_ts = _list_timeseries_jsons(ts_dir)
        if not all_ts:
            print("[WARN] No timeseries json found; skip single-device dynamics figures.")
            print(f"       timeseries_dir={ts_dir}")
            return
        n = int(args.n_devices)
        if n <= 0:
            print("[WARN] --n_devices <= 0, nothing to render.")
            return
        ts_paths = all_ts[:n]

    # Output folder for batch
    out_ts = out / "single_device_dynamics"
    out_ts.mkdir(parents=True, exist_ok=True)

    cfg = DeviceTimeseriesCompactConfig(
        figsize=(7.2, 4.8),
        height_ratios=(2.2, 0.38, 1.05, 0.22),
        hspace=0.08,
        show_low_soc_band=True,
        low_soc_threshold=0.20,
        state_band_use_rle=True,
        total_power_ylabel=None,  # keep middle panel clean; set to "Total Power (W)" if you want
    )

    # Optional: write an index file so you can quickly browse
    index_lines = []
    for i, ts_path in enumerate(ts_paths, start=1):
        if not ts_path.exists():
            print(f"[WARN] Missing: {ts_path}")
            continue

        # Use filename stem as device id (your JSON is <Device_ID>.json)
        device_id = ts_path.stem
        save_pdf = out_ts / f"{i:02d}_{device_id}.pdf"

        plot_device_timeseries_compact(
            str(ts_path),
            cfg=cfg,
            save_path=str(save_pdf),
            show=False,
        )

        index_lines.append(f"{i:02d}\t{device_id}\t{save_pdf.name}")
        print(f"[OK] ({i}/{len(ts_paths)}) Saved: {save_pdf}")

    if index_lines:
        (out_ts / "index.tsv").write_text(
            "idx\tDevice_ID\tfile\n" + "\n".join(index_lines) + "\n",
            encoding="utf-8",
        )


    print(f"[OK] Saved lifetime figures to: {out.resolve()}")


if __name__ == "__main__":
    main()
