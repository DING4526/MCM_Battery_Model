"""
Example usage / quick sanity check.

By default this script reads summary.csv (runner output) and generates figures
with data-driven key thresholds. If summary.csv is missing, it falls back to
synthetic data.

Run:
    python -m visualization.demo --summary summary.csv
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


def _make_synthetic(n: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # device age in months: mixture of new / mid / old
    age = np.concatenate([
        rng.integers(0, 12, n // 3),
        rng.integers(12, 24, n // 3),
        rng.integers(24, 48, n - 2 * (n // 3)),
    ])

    # usage dominant state
    states = np.array(["idle", "social", "video", "gaming", "navigation"])
    usage = rng.choice(states, size=n, p=[0.20, 0.25, 0.25, 0.15, 0.15])

    # baseline lifetime by usage (hours)
    mu = {
        "idle": 8.0,
        "social": 7.0,
        "video": 6.2,
        "gaming": 5.2,
        "navigation": 6.5,
    }
    sigma = {
        "idle": 0.9,
        "social": 0.95,
        "video": 0.85,
        "gaming": 0.80,
        "navigation": 0.90,
    }

    base = np.array([rng.normal(mu[u], sigma[u]) for u in usage])

    # aging effect: older => shorter
    age_penalty = 0.02 * (age / 12)  # ~2% per year
    ttl = base * (1 - age_penalty) + rng.normal(0, 0.15, size=n)

    # keep positive
    ttl = np.clip(ttl, 1.0, None)

    return pd.DataFrame({
        "ttl_hours": ttl,
        "device_age_months": age,
        "usage_dominant_state": usage
    })


def _resolve_ttl_col(df: pd.DataFrame) -> str:
    for col in ("ttl_hours", "TTL_hours"):
        if col in df.columns:
            return col
    raise ValueError(
        "No TTL column found. Expected 'ttl_hours' or 'TTL_hours'. "
        f"Available columns: {list(df.columns)}."
    )


def _infer_usage_from_ratios(df: pd.DataFrame) -> pd.DataFrame:
    ratio_cols = {
        "ratio_idle": "idle",
        "ratio_social": "social",
        "ratio_video": "video",
        "ratio_game": "gaming",
    }
    available = [col for col in ratio_cols if col in df.columns]
    if len(available) < 2:
        return df

    ratios = df[available].apply(pd.to_numeric, errors="coerce")
    has_any = ratios.notna().any(axis=1)
    dominant = ratios.idxmax(axis=1)
    usage = dominant.map(ratio_cols).where(has_any)

    df = df.copy()
    df["usage_dominant_state"] = usage
    return df


def _has_valid_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _compute_quantile_thresholds(lifetime_series: pd.Series, quantile_levels: tuple[float, ...]) -> list[float]:
    values = pd.to_numeric(lifetime_series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return []
    q = np.asarray(quantile_levels, dtype=float)
    q = q[(q >= 0.0) & (q <= 1.0)]
    if q.size == 0:
        return []
    thresholds = np.quantile(values, q)
    return [float(v) for v in thresholds]


def _format_threshold_label(quantile_levels: tuple[float, ...]) -> str:
    perc = [int(q * 100) for q in quantile_levels]
    if not perc:
        return "Key Threshold"
    if len(perc) == 1:
        return f"Key Threshold (P{perc[0]})"
    return "Key Threshold (" + "/".join(f"P{p}" for p in perc) + ")"

def _repo_root() -> Path:
    # demo.py 位于 D:\src\visualization\demo.py
    # parent: visualization
    # parent: src
    # parent: D:\
    return Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot lifetime distribution from summary.csv.")
    parser.add_argument(
        "--summary",
        default=str(_repo_root() / "output" / "population" / "summary.csv"),
        help="Path to runner output summary.csv. Default: <repo>/output/population/summary.csv",
    )
    parser.add_argument("--out", default=str(_repo_root() / "output" / "figures_summary"),
                        help="Output directory for figures.")
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.2, 0.5, 0.8],
        help="Quantiles used for key threshold lines (0-1).",
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic demo data (ONLY if explicitly set).")
    args = parser.parse_args()

    summary_path = Path(args.summary)

    if args.synthetic:
        df = _make_synthetic()
        ttl_col = "ttl_hours"
    else:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"summary.csv not found at: {summary_path.resolve()}\n"
                f"Tip: run with --summary <path-to-summary.csv> or place it at "
                f"{(_repo_root() / 'output' / 'population' / 'summary.csv').resolve()}"
            )
        df = pd.read_csv(summary_path)
        ttl_col = _resolve_ttl_col(df)
        if "usage_dominant_state" not in df.columns:
            df = _infer_usage_from_ratios(df)

    device_age_col = "device_age_months"
    usage_state_col = "usage_dominant_state"
    has_device_age = _has_valid_column(df, device_age_col)
    has_usage_state = _has_valid_column(df, usage_state_col)

    quantile_levels = tuple(args.quantiles)
    thresholds = _compute_quantile_thresholds(df[ttl_col], quantile_levels)
    threshold_label = _format_threshold_label(quantile_levels)

    cfg = LifetimePlotConfig(
        ttl_col=ttl_col,
        device_age_col=device_age_col,
        usage_state_col=usage_state_col,
        figsize=(6.6, 4.4),
        max_levels_usage=8,
        vlines_hours=thresholds if thresholds else None,
        vlines_label=threshold_label,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    r = df[["ratio_idle","ratio_social","ratio_video","ratio_game"]].apply(pd.to_numeric, errors="coerce")
    print(r.describe(percentiles=[.5,.8,.9,.95]))
    print((r.idxmax(axis=1)).value_counts())

    plot_population_lifetime_distribution(
        df,
        kind="kde",
        stratify="none",
        cfg=cfg,
        save_path=str(out / "lifetime_kde_overall.pdf"),
        show=False,
    )
    plot_population_lifetime_distribution(
        df,
        kind="ecdf",
        stratify="none",
        cfg=cfg,
        save_path=str(out / "lifetime_ecdf_overall.pdf"),
        show=False,
    )

    if has_device_age:
        plot_population_lifetime_distribution(
            df,
            kind="kde",
            stratify=cfg.device_age_col,
            cfg=cfg,
            save_path=str(out / "lifetime_kde_by_age.pdf"),
            show=False,
        )
        plot_population_lifetime_distribution(
            df,
            kind="ecdf",
            stratify=cfg.device_age_col,
            cfg=cfg,
            save_path=str(out / "lifetime_ecdf_by_age.pdf"),
            show=False,
        )

    if has_usage_state:
        plot_population_lifetime_distribution(
            df,
            kind="kde",
            stratify=cfg.usage_state_col,
            cfg=cfg,
            save_path=str(out / "lifetime_kde_by_usage.pdf"),
            show=False,
        )
        plot_population_lifetime_distribution(
            df,
            kind="ecdf",
            stratify=cfg.usage_state_col,
            cfg=cfg,
            save_path=str(out / "lifetime_ecdf_by_usage.pdf"),
            show=False,
        )

    print(f"Saved figures to: {out.resolve()}")


if __name__ == "__main__":
    
    main()
