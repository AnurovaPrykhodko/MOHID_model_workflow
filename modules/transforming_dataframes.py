# ###########################################################################
#
# Author  : Karolina Anurova-Prykhodko
#
# Description : Contains the functions related to transforming dataframes.
#
# ###########################################################################

import pandas as pd
import statistics 

def si_by_layer(all_results, reference_key='Run1', columns=None):
    """
    Reshape sensitivity indices into one DataFrame per layer (suffix index).

    Parameters
    ----------
    all_results : dict[str, dict[int, pd.DataFrame]]
        e.g. {'Run1': {1: df, 2: df, ...}, 'Run2': {...}, ...}
    reference_key : str
        Key in `all_results` to use as the reference run.
    columns : list[str], optional
        Restrict the comparison to these columns.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from layer index -> DataFrame whose rows are runs and whose
        columns are ['Run', 'si_<column1>', 'si_<column2>', ...].
    """
    ref = all_results[reference_key]

    # {layer_idx: list of rows, one per run}
    per_layer_rows = {}

    for run_key, run in all_results.items():
        if run_key == reference_key:
            continue

        si_df = statistics.sensitivity_index(run, ref, columns=columns)
        # si_df: rows = layer index, columns = data columns

        for layer_idx, row in si_df.iterrows():
            record = {'Run': run_key}
            record.update({f'si_{col}': row[col] for col in si_df.columns})
            per_layer_rows.setdefault(layer_idx, []).append(record)

    return {
        layer_idx: pd.DataFrame(rows).reset_index(drop=True)
        for layer_idx, rows in per_layer_rows.items()
    }