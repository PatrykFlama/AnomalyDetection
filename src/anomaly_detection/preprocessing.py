from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-8


@dataclass(frozen=True)
class ScaleStats:
    center: float
    scale: float


def train_prefix_length(n_points: int, fraction: float, min_length: int = 2) -> int:
    """Return the prefix length used as normal training data."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    return max(min_length, min(n_points, int(np.ceil(n_points * fraction))))


def robust_scale_stats(values: np.ndarray) -> ScaleStats:
    """Median and MAD-based scale, with safe fallbacks for nearly constant data."""
    values = np.asarray(values, dtype=np.float32)
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return ScaleStats(center=center, scale=scale)


def standard_scale_stats(values: np.ndarray) -> ScaleStats:
    values = np.asarray(values, dtype=np.float32)
    center = float(np.nanmean(values))
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return ScaleStats(center=center, scale=scale)


def apply_scale(values: np.ndarray, stats: ScaleStats) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values - stats.center) / stats.scale).astype(np.float32)


def finite_array(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(
        np.float32
    )


def shifted_rolling(values: np.ndarray, window: int, agg: str) -> np.ndarray:
    """Causal rolling statistic using only points before the current timestamp."""
    if window <= 0:
        raise ValueError("window must be positive")
    series = pd.Series(np.asarray(values, dtype=np.float32))
    rolling = series.rolling(window=window, min_periods=1)
    if agg == "mean":
        result = rolling.mean()
    elif agg == "std":
        result = rolling.std(ddof=0)
    elif agg == "median":
        result = rolling.median()
    elif agg == "min":
        result = rolling.min()
    elif agg == "max":
        result = rolling.max()
    else:
        raise ValueError(f"Unsupported rolling aggregation: {agg}")
    result = result.shift(1)
    fallback = float(series.iloc[0]) if len(series) else 0.0
    return result.bfill().fillna(fallback).to_numpy(dtype=np.float32)


def rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling MAD around the causal rolling median."""
    values = np.asarray(values, dtype=np.float32)
    median = shifted_rolling(values, window, "median")
    abs_deviation = np.abs(values - median)
    mad = shifted_rolling(abs_deviation, window, "median")
    return finite_array(1.4826 * mad, fill_value=1.0)


def point_features(values: np.ndarray, rolling_window: int) -> np.ndarray:
    """Build simple causal point-level features for tree-based detectors."""
    values = np.asarray(values, dtype=np.float32)
    prev = np.roll(values, 1)
    prev[0] = values[0]
    diff = values - prev
    mean = shifted_rolling(values, rolling_window, "mean")
    std = shifted_rolling(values, rolling_window, "std")
    median = shifted_rolling(values, rolling_window, "median")
    min_value = shifted_rolling(values, rolling_window, "min")
    max_value = shifted_rolling(values, rolling_window, "max")
    residual_mean = values - mean
    residual_median = values - median
    features = np.column_stack(
        [
            values,
            diff,
            np.abs(diff),
            mean,
            std,
            median,
            min_value,
            max_value,
            residual_mean,
            residual_median,
        ]
    )
    return finite_array(features)


def sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) < window:
        raise ValueError("series is shorter than the requested window")
    return np.lib.stride_tricks.sliding_window_view(values, window).astype(np.float32)


def window_scores_to_points(n_points: int, window: int, scores: np.ndarray) -> np.ndarray:
    """Align scores for windows ending at t back to point timestamps."""
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return np.zeros(n_points, dtype=np.float32)
    output = np.empty(n_points, dtype=np.float32)
    prefix_len = min(window - 1, n_points)
    output[:prefix_len] = float(scores[0])
    output[prefix_len:] = scores[: n_points - prefix_len]
    return finite_array(output)


def prediction_errors_to_points(
    n_points: int, context_window: int, errors: np.ndarray
) -> np.ndarray:
    """Align one-step-ahead errors for point t after a context window."""
    errors = np.asarray(errors, dtype=np.float32)
    if len(errors) == 0:
        return np.zeros(n_points, dtype=np.float32)
    output = np.empty(n_points, dtype=np.float32)
    prefix_len = min(context_window, n_points)
    output[:prefix_len] = float(errors[0])
    output[prefix_len:] = errors[: n_points - prefix_len]
    return finite_array(output)
