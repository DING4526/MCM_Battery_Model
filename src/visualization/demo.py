"""
Example usage / quick sanity check.

This script generates demo figures with synthetic data.
Run:
    python -m visualization.demo
"""

from __future__ import annotations

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


def main():
    df = _make_synthetic()

    cfg = LifetimePlotConfig(
        figsize=(6.6, 4.4),
        max_levels_usage=8,
        vlines_hours=[5.0, 7.0],
        vlines_label="Key Threshold",
    )

    out = Path("figures_demo")
    out.mkdir(exist_ok=True)

    plot_population_lifetime_distribution(df, kind="kde", stratify="none", cfg=cfg,
                                         save_path=str(out / "lifetime_kde_overall.pdf"), show=False)
    plot_population_lifetime_distribution(df, kind="ecdf", stratify="none", cfg=cfg,
                                         save_path=str(out / "lifetime_ecdf_overall.pdf"), show=False)

    plot_population_lifetime_distribution(df, kind="kde", stratify="device_age_months", cfg=cfg,
                                         save_path=str(out / "lifetime_kde_by_age.pdf"), show=False)
    plot_population_lifetime_distribution(df, kind="ecdf", stratify="device_age_months", cfg=cfg,
                                         save_path=str(out / "lifetime_ecdf_by_age.pdf"), show=False)

    plot_population_lifetime_distribution(df, kind="kde", stratify="usage_dominant_state", cfg=cfg,
                                         save_path=str(out / "lifetime_kde_by_usage.pdf"), show=False)
    plot_population_lifetime_distribution(df, kind="ecdf", stratify="usage_dominant_state", cfg=cfg,
                                         save_path=str(out / "lifetime_ecdf_by_usage.pdf"), show=False)

    print(f"Saved demo figures to: {out.resolve()}")


if __name__ == "__main__":
    main()
