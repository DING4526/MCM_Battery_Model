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
from .utils import add_age_group_column, add_usage_intensity_group
from .usage_composition import plot_active_composition_stacked_bar

__all__ = [
    "LifetimePlotConfig",
    "plot_population_lifetime_distribution",
    "plot_population_kde",
    "plot_population_ecdf",
    "plot_kde_stratified",
    "plot_ecdf_stratified",
    "add_age_group_column",
    "add_usage_intensity_group",
    "plot_active_composition_stacked_bar",
]

from .power_structure_boxplot import (
    PowerStructureBoxplotConfig,
    plot_power_structure_boxplot,
)

__all__ += [
    "PowerStructureBoxplotConfig",
    "plot_power_structure_boxplot",
]

from .device_timeseries_compact import (
    DeviceTimeseriesCompactConfig,
    plot_device_timeseries_compact,
)

__all__ += [
    "DeviceTimeseriesCompactConfig",
    "plot_device_timeseries_compact",
]