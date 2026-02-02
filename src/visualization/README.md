# visualization module

Paper-grade static plots for the battery lifetime project.

## Features

Population Lifetime Distribution plots:
- KDE (Kernel Density Estimate)
- ECDF (Empirical CDF)

Stratification modes:
- none
- by `device_age_months` (auto-binning into `<1 year`, `1–2 years`, `≥2 years`)
- by `usage_dominant_state`

All labels/titles are in **English** by default for LaTeX papers.

## Install requirements

- matplotlib
- numpy
- pandas
- scipy (required for KDE)

## Quick start

```python
import pandas as pd
from visualization import plot_population_lifetime_distribution, LifetimePlotConfig

df = pd.read_parquet("your_simulation.parquet")  # or pd.read_csv(...)

cfg = LifetimePlotConfig(
    ttl_col="ttl_hours",
    device_age_col="device_age_months",
    usage_state_col="usage_dominant_state",
    figsize=(6.6, 4.4),
    vlines_hours=[5.0, 7.0],   # optional reference lines
)

# Overall KDE / ECDF
plot_population_lifetime_distribution(df, kind="kde", stratify="none", cfg=cfg,
                                      save_path="lifetime_kde_overall.pdf", show=False)

plot_population_lifetime_distribution(df, kind="ecdf", stratify="none", cfg=cfg,
                                      save_path="lifetime_ecdf_overall.pdf", show=False)

# Stratified by device age
plot_population_lifetime_distribution(df, kind="kde", stratify="device_age_months", cfg=cfg,
                                      save_path="lifetime_kde_by_age.pdf", show=False)

# Stratified by dominant usage state
plot_population_lifetime_distribution(df, kind="ecdf", stratify="usage_dominant_state", cfg=cfg,
                                      save_path="lifetime_ecdf_by_usage.pdf", show=False)
```

## Demo / summary.csv

Generate figures from runner output (summary.csv) with data-driven key thresholds:

```bash
python -m visualization.demo --summary summary.csv --out output/figures_summary
```

If summary.csv is not available, you can still generate synthetic demo figures:

```bash
python -m visualization.demo --synthetic --out figures_demo
```
