"""
Metrics module.

Provides functions to compare model output against a reference (reference simulation or observations),
using common metrics, and utilities to apply them across multiple runs outputs.

Metrics
-------
- rmse              : Root Mean Square Error
- bias              : Mean bias (run - reference)
- rrmse             : Relative RMSE (RMSE / mean of reference)
- sensitivity_index : RMSE normalized by the reference range (max - min)
- pearson_r         : Pearson correlation coefficient
- mss               : Model Skill Score (Murphy 1988) / Nash-Sutcliffe efficiency

Application helpers
-------------------
- compare_vs_reference : Apply a metric between one run and the reference,
                        across all matching DataFrames (keyed by index).
- compare_all_runs     : Apply a metric between every run and the reference
                        in a dictionary of runs.
- by_layer             : Reorganize a per-run dict of metric DataFrames into a per-layer dict.

All metric functions share the signature ``f(a, b) -> pd.Series``, where
``a`` and ``b`` are row-aligned DataFrames restricted to common columns,
so new metrics can be added and passed directly to the application helpers.

Author: Karolina Anurova-Prykhodko
"""

import pandas as pd
import numpy as np
import pandas as pd


def rmse(a, b):
    """Root Mean Square Error (column-wise)."""
    return np.sqrt(((a - b) ** 2).mean(axis=0, skipna=True))


def bias(a, b):
    """
    Mean bias (column-wise): mean(a - b).
    Positive => run overestimates the reference.
    """
    return (a - b).mean(axis=0, skipna=True)


def rrmse(a, b):
    """
    Relative RMSE (column-wise), expressed as a percentage:

        RRMSE = RMSE(a, b) / mean(a) * 100

    where `a` = model data and `b` = measurement data.
    Returns NaN where the model mean is zero.
    """
    r = rmse(a, b)
    model_mean = a.mean(axis=0, skipna=True).replace(0, np.nan)
    return r / model_mean * 100


def sensitivity_index(a, b):
    """Sensitivity Index (column-wise): RMSE / (b_max - b_min)."""
    r = rmse(a, b)
    ref_range = b.max(axis=0, skipna=True) - b.min(axis=0, skipna=True)
    ref_range = ref_range.replace(0, np.nan)
    return r / ref_range


def pearson_r(a, b):
    """
    Pearson correlation coefficient (column-wise) between `a` and `b`.
    NaNs are handled pairwise per column.
    """
    out = {}
    for c in a.columns:
        s1, s2 = a[c], b[c]
        mask = s1.notna() & s2.notna()
        if mask.sum() < 2:
            out[c] = np.nan
            continue
        x, y = s1[mask], s2[mask]
        sx, sy = x.std(ddof=0), y.std(ddof=0)
        if sx == 0 or sy == 0:
            out[c] = np.nan
        else:
            out[c] = ((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)
    return pd.Series(out)


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
    denom = denom.replace(0, np.nan)

    return 1 - num / denom


def compare_vs_reference(run, reference, metric_fn, columns=None):
    """
    Apply `metric_fn(a, b)` column-wise between each DataFrame in `run`
    and the matching DataFrame in `reference`, keyed by index.

    Parameters
    ----------
    run : dict[int, pd.DataFrame]
    reference : dict[int, pd.DataFrame]
    metric_fn : callable
        Function of the form f(a, b) -> pd.Series, where `a` and `b` are
        row-aligned DataFrames restricted to the columns of interest,
        and the returned Series is indexed by column name.
    columns : list[str], optional
        Restrict the comparison to these columns. If None, all common
        columns are used.

    Returns
    -------
    pd.DataFrame
        Rows = suffix index, columns = data columns, values = metric.
    """
    rows = {}

    for idx, df_run in run.items():
        if idx not in reference:
            print(f"Warning: index {idx} not in reference, skipping.")
            continue

        df_ref = reference[idx]

        if columns is None:
            cols = [c for c in df_run.columns if c in df_ref.columns]
        else:
            cols = [c for c in columns
                    if c in df_run.columns and c in df_ref.columns]

        if not cols:
            print(f"Warning: no common columns for index {idx}, skipping.")
            continue

        a, b = df_run[cols].align(df_ref[cols], join='inner', axis=0)
        rows[idx] = metric_fn(a, b)

    return pd.DataFrame.from_dict(rows, orient='index').sort_index()


def compare_all_runs(all_results, metric_fn, reference_key='Run1', columns=None):
    """
    Apply `metric_fn` between every run in `all_results` and the reference run.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by run name (excluding the reference).
    """
    ref = all_results[reference_key]
    return {
        run_key: compare_vs_reference(run, ref, metric_fn, columns=columns)
        for run_key, run in all_results.items()
        if run_key != reference_key
    }

def by_layer(results_per_run):
    """
    Reorganize a per-run dict of metric DataFrames into a per-layer dict.

    Input
    -----
    results_per_run : dict[str, pd.DataFrame]
        Output of `compare_all_runs`. Keys are run names (e.g. 'Run2', 'Run3'),
        values are DataFrames with rows = layer index, columns = data columns.

    Output
    ------
    dict[int, pd.DataFrame]
        Keys are layer indices. Each DataFrame has rows = run name,
        columns = data columns.
    """
    # Stack all run DataFrames into one with a (run, layer) MultiIndex
    combined = pd.concat(results_per_run, names=['run', 'layer'])

    # For each layer, pull out its cross-section (rows indexed by run)
    layers = combined.index.get_level_values('layer').unique()
    return {
        layer: combined.xs(layer, level='layer').sort_index()
        for layer in sorted(layers)
    }