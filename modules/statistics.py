# ###########################################################################
#
# Author  : Karolina Anurova-Prykhodko
#
# Description : Contains the functions related to statistics (RMSE, sensitivity index, bias, etc.)
#
# ###########################################################################

import pandas as pd
import numpy as np
import pandas as pd


def rmse_vs_reference(run, reference, columns=None):
    """
    Compute RMSE between each DataFrame in `run` and the matching DataFrame
    in `reference`, keyed by index.

    Parameters
    ----------
    run : dict[int, pd.DataFrame]
        Output of `read_files` for the run being evaluated (e.g. Run2).
    reference : dict[int, pd.DataFrame]
        Output of `read_files` for the reference run (e.g. Run1).
    columns : list[str], optional
        Restrict the comparison to these columns. If None, all columns common
        to both DataFrames are used.

    Returns
    -------
    pd.DataFrame
        Rows = suffix index, columns = data columns, values = RMSE.
    """
    rmse_rows = {}

    for idx, df_run in run.items():
        if idx not in reference:
            print(f"Warning: index {idx} not in reference, skipping.")
            continue

        df_ref = reference[idx]

        # Decide which columns to compare
        if columns is None:
            cols = [c for c in df_run.columns if c in df_ref.columns]
        else:
            cols = [c for c in columns
                    if c in df_run.columns and c in df_ref.columns]

        if not cols:
            print(f"Warning: no common columns for index {idx}, skipping.")
            continue

        # Align on the (time) index so rows correspond
        a, b = df_run[cols].align(df_ref[cols], join='inner', axis=0)

        # Column-wise RMSE, ignoring NaNs
        diff = (a - b) ** 2
        rmse_rows[idx] = np.sqrt(diff.mean(axis=0, skipna=True))

    return pd.DataFrame.from_dict(rmse_rows, orient='index').sort_index()

def rmse_all_runs(all_results, reference_key='Run1', columns=None):
    ref = all_results[reference_key]
    return {
        run_key: rmse_vs_reference(run, ref, columns=columns)
        for run_key, run in all_results.items()
        if run_key != reference_key
    }

def sensitivity_index(run, reference, columns=None):
    """
    Compute the Sensitivity Index (SI) between each DataFrame in `run` and the
    matching DataFrame in `reference`, keyed by index.

        SI = RMSE / (Q_ref_max - Q_ref_min)

    where Q_ref_max and Q_ref_min are the max and min of the reference series
    for each column, at each index.

    Parameters
    ----------
    run : dict[int, pd.DataFrame]
    reference : dict[int, pd.DataFrame]
    columns : list[str], optional
        Restrict the comparison to these columns. If None, all common columns
        are used.

    Returns
    -------
    pd.DataFrame
        Rows = suffix index, columns = data columns, values = SI.
    """
    si_rows = {}

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

        # Align on the (time) index so rows correspond
        a, b = df_run[cols].align(df_ref[cols], join='inner', axis=0)

        # Column-wise RMSE
        rmse = np.sqrt(((a - b) ** 2).mean(axis=0, skipna=True))

        # Reference range per column
        ref_range = b.max(axis=0, skipna=True) - b.min(axis=0, skipna=True)

        # Avoid division by zero -> NaN where reference is constant
        ref_range = ref_range.replace(0, np.nan)

        si_rows[idx] = rmse / ref_range

    return pd.DataFrame.from_dict(si_rows, orient='index').sort_index()

def sensitivity_index_all_runs(all_results, reference_key='Run1', columns=None):
    ref = all_results[reference_key]
    return {
        run_key: sensitivity_index(run, ref, columns=columns)
        for run_key, run in all_results.items()
        if run_key != reference_key
    }