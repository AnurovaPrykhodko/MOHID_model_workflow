"""
General helpers

Author: Karolina Anurova-Prykhodko
"""


import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def assign_bin(depth, bins):
    """Return the bin name whose (lo, hi] interval contains `depth`."""
    for name, lo, hi in bins:
        if lo < depth <= hi:
            return name
    return None


def compute_layer_thickness(depths, bottom):
    """Thickness of each layer, using midpoints between adjacent layer centers."""
    sorted_layers = sorted(depths.items(), key=lambda kv: kv[1])
    ids = [l for l, _ in sorted_layers]
    ctr = np.array([d for _, d in sorted_layers])

    faces = np.concatenate(([0.0], (ctr[:-1] + ctr[1:]) / 2, [bottom]))
    thickness = np.diff(faces)
    return dict(zip(ids, thickness.tolist()))


def group_layers_by_bin(depths, bins, available_layers):
    """Map bin name -> list of layer ids present in `available_layers`."""
    available = set(available_layers)
    groups = {name: [] for name, _, _ in bins}
    for layer, depth in depths.items():
        if layer not in available:
            continue
        b = assign_bin(depth, bins)
        if b is not None:
            groups[b].append(layer)
    return groups


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _stack_layers(run_dict, layers):
    """Stack per-layer DataFrames into an array of shape (n_layers, n_times, n_cols)."""
    ref = run_dict[layers[0]]
    idx, cols = ref.index, ref.columns
    arr = np.stack([
        run_dict[l].reindex(index=idx, columns=cols).values
        for l in layers
    ])
    return arr, idx, cols


def _normalized_weights(layers, thickness_map):
    w = np.array([thickness_map[l] for l in layers], dtype=float)
    return w / w.sum()


def _weighted_mean(arr, w):
    """Weighted mean along the first (layer) axis."""
    return np.tensordot(w, arr, axes=(0, 0))


def _circular_mean_deg(arr_deg, w):
    """Weighted circular mean of angles in degrees, returned in [0, 360)."""
    theta = np.deg2rad(arr_deg)
    s = np.tensordot(w, np.sin(theta), axes=(0, 0))
    c = np.tensordot(w, np.cos(theta), axes=(0, 0))
    return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0


def _aggregate_bin(run_dict, layers, thickness_map, circular_cols, uv_cols):
    """Compute the weighted-mean DataFrame for a single bin."""
    arr, idx, cols = _stack_layers(run_dict, layers)
    w = _normalized_weights(layers, thickness_map)

    result = pd.DataFrame(_weighted_mean(arr, w), index=idx, columns=cols)

    # Circular quantities need special treatment
    for c in circular_cols:
        if c in cols:
            j = cols.get_loc(c)
            result[c] = _circular_mean_deg(arr[:, :, j], w)

    # Keep modulus consistent with the averaged components
    if uv_cols and set(uv_cols).issubset(cols):
        u, v = uv_cols
        result["velocity_modulus"] = np.hypot(result[u], result[v])

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_run(
    run_dict,
    layer_depths,
    bins,
    bottom_depth,
    circular_cols=(),
    uv_cols=("velocity_U", "velocity_V"),
):
    """
    Aggregate a single run's per-layer DataFrames into depth bins.

    Parameters
    ----------
    run_dict : dict
        Keys are layer indices, values are DataFrames indexed by date.
    layer_depths : dict
        Map of layer id -> depth (center of layer).
    bins : list of (name, lo, hi)
        Bin definitions, intervals interpreted as (lo, hi].
    bottom_depth : float
        Depth of the bottom face, used for the deepest layer's thickness.
    circular_cols : iterable of str
        Columns (in degrees) to average with circular statistics.
    uv_cols : tuple of (str, str) or None
        If given and both present, recompute `velocity_modulus` from them.

    Returns
    -------
    dict
        One DataFrame per non-empty bin, keyed by bin name.
    """
    thickness_map = compute_layer_thickness(layer_depths, bottom_depth)
    groups = group_layers_by_bin(layer_depths, bins, run_dict.keys())

    return {
        bname: _aggregate_bin(run_dict, layers, thickness_map, circular_cols, uv_cols)
        for bname, layers in groups.items()
        if layers
    }