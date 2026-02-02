"""
Visualization package (paper-grade static figures).

This module provides "Population Lifetime Distribution" plots:
- Kernel Density Estimate (KDE)
- Empirical CDF (ECDF)

Supports:
- No stratification
- Stratify by device_age_months (auto-binning or user-specified bins)
- Stratify by usage_dominant_state (categorical)

All text elements default to English for paper compatibility.
"""

from .lifetime_distribution import (
    LifetimePlotConfig,
    plot_population_lifetime_distribution,
    plot_population_kde,
    plot_population_ecdf,
    plot_kde_stratified,
    plot_ecdf_stratified,
    add_age_group_column,
)

__all__ = [
    "LifetimePlotConfig",
    "plot_population_lifetime_distribution",
    "plot_population_kde",
    "plot_population_ecdf",
    "plot_kde_stratified",
    "plot_ecdf_stratified",
    "add_age_group_column",
]
