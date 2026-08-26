"""
Provides functions to compare model output against a reference (reference simulation or observations)
using common metrics, and utilities to apply them.

Metrics
-------
- rmse              : Root Mean Square Error
- bias              : Mean bias (run - reference)
- pearson_r         : Pearson correlation coefficient
- mss               : Model Skill Score 

(each metrics has a duplicate for the circular version for variables such at velocity direction or tidal phase)

Application helpers
-------------------
- compare_vs_reference  : Compare model output to a reference by a metric function.
- compare_multi_metrics : Compare model output to a reference by several metric functions.

Author: Karolina Anurova-Prykhodko
"""

import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.stats import circmean

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rmse(a, b):
    """Root Mean Square Error,
    where `a` = model, `b` = observations (reference). """
    return np.sqrt(((a - b) ** 2).mean(axis=0))

def bias(a, b):
    """
    Mean bias, where `a` = model, `b` = observations (reference). """
    return (a - b).mean(axis=0)

def pearson_r(a, b):
    
    return pearsonr(a, b)[0]

def mss(a, b):
    """
    Willmott's Model Skill Score (column-wise):

        MSS = 1 - sum((a - b)^2) /
                  sum((|a - mean(b)| + |b - mean(b)|)^2)

    where `a` = model, `b` = observations (reference).

    Ranges from 0 (no agreement) to 1 (perfect agreement).
    """
    b_mean = b.mean(axis=0, skipna=True)

    num = ((a - b) ** 2).sum(axis=0, skipna=True)
    denom = (((a - b_mean).abs() + (b - b_mean).abs()) ** 2).sum(axis=0, skipna=True)
    
    if denom == 0:
        raise ValueError
    #denom = denom.replace(0, np.nan)

    return 1 - num / denom


def circular_diff(a, b):
    """
    Shortest signed circular difference assuming a 360-degree cycle.
    """
    return (a - b + 180) % 360 - 180



def circular_corr(a, b, axis=0):
    """
    Circular-circular correlation coefficient, assuming a 360-degree cycle.

    This is appropriate when both `a` and `b` are circular variables,
    for example model wind direction vs observed wind direction.

    Parameters
    ----------
    a, b : array-like
        Circular variables in degrees.

    axis : int
        Axis along which to calculate correlation.

    Returns
    -------
    float or np.ndarray
        Circular correlation coefficient.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")

    # Move target axis to the front
    a = np.moveaxis(a, axis, 0)
    b = np.moveaxis(b, axis, 0)

    out_shape = a.shape[1:]
    result = np.full(out_shape, np.nan, dtype=float)

    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)

    for i in range(a_flat.shape[1]):
        ai = a_flat[:, i]
        bi = b_flat[:, i]

        valid = np.isfinite(ai) & np.isfinite(bi)

        if valid.sum() < 2:
            continue

        ai = ai[valid]
        bi = bi[valid]

        alpha = ai * 2 * np.pi / 360
        beta = bi * 2 * np.pi / 360

        alpha_bar = np.arctan2(
            np.mean(np.sin(alpha)),
            np.mean(np.cos(alpha)),
        )

        beta_bar = np.arctan2(
            np.mean(np.sin(beta)),
            np.mean(np.cos(beta)),
        )

        sin_alpha = np.sin(alpha - alpha_bar)
        sin_beta = np.sin(beta - beta_bar)

        num = np.sum(sin_alpha * sin_beta)
        denom = np.sqrt(
            np.sum(sin_alpha ** 2) * np.sum(sin_beta ** 2)
        )

        if denom != 0:
            result.reshape(-1)[i] = num / denom

    if result.shape == ():
        return float(result)

    return result


def rmse_circ(a, b):
    """
    RMSE for circular variables in degrees (direction, phase). 
    """
    err = circular_diff(a, b)

    return np.sqrt(np.nanmean(err ** 2, axis=0))


def bias_circ(a, b):
    """
    Circular mean bias in degrees.
    """
    err = circular_diff(a, b)

    return circmean(err, high=180, low=-180, nan_policy="omit", axis=0)


def pearson_r_circ(a, b):
    """
    Circular-circular correlation coefficient, assuming a 360-degree cycle.

    This replaces standard Pearson correlation for circular variables.

    Parameters
    ----------
    a : array-like
        Model circular values in degrees.

    b : array-like
        Reference circular values in degrees.

    Returns
    -------
    float or np.ndarray
        Circular correlation coefficient.
    """
    return circular_corr(a, b, axis=0)


def mss_circ(a, b):
    """
    Circular Willmott's Model Skill Score, assuming a 360-degree cycle.

    Uses circular distances instead of linear distances.

    Parameters
    ----------
    a : array-like
        Model circular values in degrees.

    b : array-like
        Reference circular values in degrees.

    Returns
    -------
    float or np.ndarray
        Circular Model Skill Score.
    """
    b_mean = circmean(b, high=180, low=-180, nan_policy="omit", axis=0)
    num = np.nansum(
        circular_diff(a, b) ** 2,
        axis=0,
    )

    denom = np.nansum(
        (
            np.abs(circular_diff(a, b_mean))
            + np.abs(circular_diff(b, b_mean))
        ) ** 2,
        axis=0,
    )
    return 1 - num / denom

# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------


def compare_vs_reference(run, reference, metric_fn, columns=None):
    """
    Compare model output against a reference, computing the function(s)
    metric_fn per column.

    Parameters
    ----------
    run : pd.DataFrame
        Indexed by timestamp.
    reference : pd.DataFrame
        Indexed by timestamp.
    metric_fn : callable
        Function of the form f(a, b) -> float, where `a` and `b` are
        row-aligned 1-D Series for a single column.
    columns : str | list[str], optional
        Column(s) to compare. If None, all common columns are used.

    Returns
    -------
    pd.Series
        Index = column name, values = metric.
    """
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        cols = [c for c in run.columns if c in reference.columns]
    else:
        cols = [
            c for c in columns
            if c in run.columns and c in reference.columns
        ]

    if not cols:
        raise ValueError("No common columns to compare.")

    a, b = run[cols].align(reference[cols], join="inner", axis=0)

    if a.empty:
        raise ValueError("No overlapping timestamps between model output and reference.")

    # Drop rows where either side has NaN, per column, before applying metric
    results = {}

    for c in cols:
        pair = pd.concat(
            [a[c], b[c]],
            axis=1,
            keys=["model_output", "reference"]
        ).dropna()

        if pair.empty:
            results[c] = float("nan")
        else:
            results[c] = metric_fn(pair["model_output"], pair["reference"])

    return pd.Series(results, name=metric_fn.__name__)

def compare_multi_metrics(run, reference, metric_fns, columns=None):
    """
    Compare model output against a reference across multiple metrics.

    Parameters
    ----------
    run : pd.DataFrame
    reference : pd.DataFrame
    metric_fns : list[callable]
        Each callable f(a, b) -> float. Its __name__ is used as the metric column.
    columns : str | list[str], optional
        Columns to compare. If None, all common columns are used.
    run_name : str, optional
        Label for this run, stored in a 'run' column. Defaults to 'run'.

    Returns
    -------
    pd.DataFrame
        One row per compared column. Columns: ['run', 'variable', <metric names...>].
    """
    series_list = []
    for fn in metric_fns:
        s = compare_vs_reference(run, reference, fn, columns=columns)
        # Use the callable's __name__ (fallback for things like pearsonr wrappers)
        s.name = getattr(fn, "__name__", str(fn))
        series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df.index.name = "variable"
    df = df.reset_index()
    return df