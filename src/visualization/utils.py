"""
Small utilities used across plotting functions.
"""

from __future__ import annotations

from typing import Tuple, Iterable, Optional, Dict, Any, Sequence, List, Union

import numpy as np
import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    *,
    ttl_col: str,
    strata_cols: Optional[Sequence[str]] = None,
) -> None:
    missing = [ttl_col] if ttl_col not in df.columns else []
    if strata_cols:
        missing += [c for c in strata_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ECDF for a 1D array.

    Returns:
        x_sorted, y (in [0,1]).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1) / x_sorted.size
    return x_sorted, y


def robust_xlim(values: np.ndarray, pad_ratio: float = 0.04) -> Tuple[float, float]:
    """
    Robust x-limits using percentiles to avoid extreme outliers dominating.

    Args:
        values: 1D numeric array.
        pad_ratio: relative padding on each side.

    Returns:
        (xmin, xmax)
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, 99.5)
    if lo == hi:
        lo = x.min()
        hi = x.max() if x.max() > x.min() else x.min() + 1.0
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_age_group_column(
    df: pd.DataFrame,
    *,
    age_col: str = "device_age_months",
    out_col: str = "age_group",
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    right: bool = False,
) -> pd.DataFrame:
    """
    Add a categorical age_group column based on device_age_months.

    Default bins:
        [0, 12), [12, 24), [24, inf)  (in months)
    """
    if age_col not in df.columns:
        raise ValueError(f"Column '{age_col}' not found.")

    age = to_numeric_series(df[age_col])

    if bins is None:
        bins = [0, 12, 24, np.inf]
    if labels is None:
        labels = ["<1 year", "1–2 years", "≥2 years"]

    if len(labels) != (len(bins) - 1):
        raise ValueError("labels must have length len(bins)-1")

    df = df.copy()
    df[out_col] = pd.cut(age, bins=bins, labels=labels, right=right, include_lowest=True)
    return df


def order_categories(
    df: pd.DataFrame,
    col: str,
    *,
    order: Optional[Sequence[str]] = None,
    max_levels: Optional[int] = None,
) -> List[Any]:
    """
    Determine plotting order for categories.

    - If 'order' is provided, use it (filtering to existing levels).
    - Else use frequency order (descending).
    - Optionally limit to max_levels (keep top-N).
    """
    levels = df[col].dropna()
    if order is not None:
        out = [o for o in order if o in set(levels)]
    else:
        out = list(levels.value_counts().index)

    if max_levels is not None:
        out = out[:max_levels]
    return out


def add_usage_intensity_group(
    df: pd.DataFrame,
    *,
    social_col: str = "ratio_social",
    video_col: str = "ratio_video",
    game_col: str = "ratio_game",
    out_col: str = "usage_intensity_group",
    active_ratio_col: str = "active_usage_ratio",
    bins: Sequence[float] = (0.0, 0.35, 0.65, 1.0),
    labels: Sequence[str] = ("low-active", "mid-active", "high-active"),
    right: bool = False,
) -> pd.DataFrame:
    """
    Add usage intensity group based on active usage ratio:
        active = ratio_social + ratio_video + ratio_game

    Creates:
      - active_usage_ratio (continuous)
      - usage_intensity_group (categorical)

    Raises:
      - ValueError if ratio columns missing or grouping collapses to <= 1 level.
    """
    for c in (social_col, video_col, game_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found; required for intensity grouping.")

    df = df.copy()

    active = (
        0.8 * to_numeric_series(df[social_col])
        + 1.7 * to_numeric_series(df[video_col])
        + 2.5 * to_numeric_series(df[game_col])
    )

    df[active_ratio_col] = active

    df[out_col] = pd.cut(
        active,
        bins=list(bins),
        labels=list(labels),
        include_lowest=True,
        right=right,
    )

    vc = df[out_col].value_counts(dropna=True)
    if vc.size < 2:
        raise ValueError(
            f"{out_col} collapsed (<=1 group). Check ratio columns / bins.\n"
            f"value_counts:\n{vc}"
        )

    return df