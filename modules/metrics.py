"""
Provides functions to compare model output against a reference (reference simulation or observations),
using common metrics, and utilities to apply them across multiple runs outputs.

Metrics
-------
- rmse              : Root Mean Square Error
- bias              : Mean bias (run - reference)
- rrmse             : Relative RMSE (RMSE / mean of reference)
- sensitivity_index : RMSE normalized by the reference range (max - min)
- pearson_r         : Pearson correlation coefficient
- mss               : Model Skill Score 

Application helpers
-------------------
- compare_vs_reference : Apply a metric between one run and the reference.
- compare_all_runs     : Apply a metric between every run and the reference
                        in a dictionary of runs.
- by_layer             : Reorganize a per-run dict of metric DataFrames into a per-layer dict.
- layers_to_dataarray  : Convert dataframe column to dataarray with dimesions.

Author: Karolina Anurova-Prykhodko
"""

import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

def rmse(a, b):
    """Root Mean Square Error (column-wise)."""
    return np.sqrt(((a - b) ** 2).mean(axis=0))


def bias(a, b):
    """
    Mean bias (column-wise): mean(a - b).
    Positive => run overestimates the reference.
    """
    return (a - b).mean(axis=0)


def rrmse(a, b):
    """
    Relative RMSE (column-wise), expressed as a percentage:

        RRMSE = RMSE(a, b) / mean(a) * 100

    where `a` = model data and `b` = measurement data.
    Returns NaN where the model mean is zero.
    """
    r = rmse(a, b)
    model_mean = a.mean(axis=0).replace(0, np.nan)
    return r / model_mean * 100


def sensitivity_index(a, b):
    """Sensitivity Index (column-wise): RMSE / (b_max - b_min)."""
    r = rmse(a, b)
    ref_range = b.max(axis=0) - b.min(axis=0)
    ref_range = ref_range.replace(0, np.nan)
    return r / ref_range


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


def layers_to_dataarray(run_dict, column='velocity_U', range_name='range'):
    """
    Convert a dict of per-layer DataFrames into a DataArray (range, time),
    preserving the datetime index from the source DataFrames.

    Parameters
    ----------
    run_dict : dict[int, pd.DataFrame]
        e.g. all_results['Run1']; each value is a DataFrame indexed by time
        and containing `column`.
    column : str
        Column to extract.
    range_name : str
        Name of the layer/range coordinate.

    Returns
    -------
    xarray.DataArray with dims (range, time)
    """
    layers = sorted(run_dict.keys())

    # Keep the original (datetime) index; concat aligns on it automatically
    series_list = [run_dict[k][column].rename(k) for k in layers]
    df = pd.concat(series_list, axis=1)      # index = time, columns = layers
    df = df.sort_index()

    data = df.values.T                       # (n_layers, n_time)

    da = xr.DataArray(
        data,
        dims=(range_name, 'time'),
        coords={
            range_name: np.array(layers, dtype=float),
            'time': df.index.values,         # preserved datetimes
        },
        name=column,
    )
    return da

def compare_vs_reference_da(run_da, ref_da, metric_fn,
                            range_dim='range', time_dim='time'):
    """
    Compare two DataArrays layer-by-layer along `range_dim`.

    Parameters
    ----------
    run_da, ref_da : xarray.DataArray
        Must share `range_dim` and `time_dim`.
    metric_fn : callable(a, b) -> float
        Takes two 1D numpy arrays (already NaN-aligned) and returns a scalar.

    Returns
    -------
    pd.Series indexed by range values.
    """
    # Align on time
    a, b = xr.align(run_da, ref_da, join='inner')

    results = {}
    for r in a[range_dim].values:
        x = a.sel({range_dim: r}).values
        y = b.sel({range_dim: r}).values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() == 0:
            results[r] = np.nan
        else:
            results[r] = metric_fn(x[mask], y[mask])

    return pd.Series(results).sort_index()

def compare_vs_reference_df(run, reference, metric_fn, columns=None):
    """
    Compare a run DataFrame against a reference DataFrame on shared
    timestamps, computing `metric_fn` per column.

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
        cols = [c for c in columns
                if c in run.columns and c in reference.columns]

    if not cols:
        raise ValueError("No common columns to compare.")

    a, b = run[cols].align(reference[cols], join='inner', axis=0)

    if a.empty:
        raise ValueError("No overlapping timestamps between run and reference.")

    # Drop rows where either side has NaN, per column, before applying metric
    results = {}
    for c in cols:
        pair = pd.concat([a[c], b[c]], axis=1, keys=['run', 'ref']).dropna()
        if pair.empty:
            results[c] = float('nan')
        else:
            results[c] = metric_fn(pair['run'], pair['ref'])

    return pd.Series(results, name=metric_fn.__name__)

def compare_multi_metrics(run, reference, metric_fns, columns=None, run_name=None):
    """
    Compare a run DataFrame against a reference across multiple metrics.

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
        s = compare_vs_reference_df(run, reference, fn, columns=columns)
        # Use the callable's __name__ (fallback for things like pearsonr wrappers)
        s.name = getattr(fn, "__name__", str(fn))
        series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df.index.name = "variable"
    df = df.reset_index()
    df.insert(0, "run", run_name if run_name is not None else "run")
    return df


def compare_runs(runs, reference, metric_fns, columns=None):
    """
    Compare several runs against the same reference.

    Parameters
    ----------
    runs : dict[str, pd.DataFrame]
        Mapping of run label -> run DataFrame.
    reference : pd.DataFrame
    metric_fns : list[callable]
    columns : str | list[str], optional

    Returns
    -------
    pd.DataFrame
        Stacked results with columns ['run', 'variable', <metric names...>].
    """
    frames = [
        compare_multi_metrics(run_df, reference, metric_fns,
                              columns=columns, run_name=label)
        for label, run_df in runs.items()
    ]
    return pd.concat(frames, ignore_index=True)