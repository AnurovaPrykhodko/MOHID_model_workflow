# ###########################################################################
#
# Author  : Karolina Anurova-Prykhodko
#
# Description : Contains the functions to quality flag ADCP data.
#
# ###########################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# ###########################################################################
# functions related to quality control
# ###########################################################################

def detect_outliers(ds, bottom_threshold, surface_threshold, set_to_NaN=True):
     """
     Detect signal outliers based on linear decay of signal. Signal counts as outlier if less than 3 beams are valid.
     
     Parameters
     ----------
     ds : xarray.Dataset
     bottom_threshold : float
          Amplitude threshold at the bottom (first bin). Bins with amplitude above
          this value are considered outliers.
     surface_threshold : float
          Amplitude threshold at the surface (last bin). Threshold is linearly
          interpolated between bottom and surface.
     set_to_NaN : Boolean
          Adjust to return copy of dataset with velocity set to NaN according to outliers (True)
          or return mask with true for valid values (False)

     Returns:
     -------
     ds_cleaned : xarray.Dataset
          Copy of the input dataset with:
          - 'amp' : beam amplitudes above threshold set to NaN
          - 'vel' : velocities set to NaN where fewer than 3 beams are valid.
     or 
     beam_mask : xarray.Dataset
          Mask with True for valid values.
     """
     n_bins = ds.sizes['range']
     threshold = np.linspace(bottom_threshold, surface_threshold, n_bins)
     threshold_da = xr.DataArray(threshold, dims=['range'], coords={'range': ds.range})

     ds_cleaned = ds.copy()

     mask_per_beam = ds['amp'] < threshold_da
     beam_mask = mask_per_beam.sum(dim='beam') >= 3 # keep only if 3 or more beams valid

     ds_cleaned['amp'] = ds['amp'].where(mask_per_beam)

     if 'vel' in ds and set_to_NaN == True:
           ds_cleaned['vel'] = ds['vel'].where(beam_mask)
           return ds_cleaned
     elif set_to_NaN == False:
           return beam_mask
     else:
           print('dataset does not contain variable vel')

def detect_const_pressure(ds, pressure_diff_thresh=0.001):
    """
    Detect three consecutive constant pressure values in a dataset. Constant is defined as a 
    value +- pressure_diff_thresh to allow for rounded values of the last digit.
    
    Parameters:
        ds: xarray Dataset containing 'pressure' variable
        pressure_diff_thresh: float, threshold for detecting constant values (default 0.001)
    
    Returns:
        mask_pressure_constant: xarray DataArray, True where pressure is constant
    """
    # Calculate difference between consecutive values
    pressure_diff = np.abs(ds['pressure'] - ds['pressure'].shift(time=1)).fillna(0)
    
    # Check where difference is below threshold
    pressure_constant = pressure_diff <= pressure_diff_thresh
    
    # Find where we have more than 3 consecutive constant values
    consecutive_constant = pressure_constant.rolling(time=3, min_periods=3).sum() >= 3
    
    # Shift back to flag all points of constant runs
    mask_pressure_constant = (
        consecutive_constant | 
        consecutive_constant.shift(time=-1, fill_value=False) | 
        consecutive_constant.shift(time=-2, fill_value=False)
    )
    
    return mask_pressure_constant

def detect_spikes(variable, window, dim, min_periods=None, threshold=3.0):
    """
    Detect spikes using z-scores computed over a rolling window.
    
    Parameters
    ----------
    variable : xr.DataArray
        Input data array.
    window : int
        Size of the rolling window.
    dim : str
        Dimension along which to roll.
    threshold : float, optional
        Z-score threshold (default: 3.0).
    min_periods : int or None, optional
        Minimum observations required. If None, defaults to window size
        (stricter, more NaNs at edges).
    
    Returns
    -------
    xr.DataArray
        Boolean array where True indicates a spike.
    """
    if min_periods is None:
        min_periods = window
    
    rolling = variable.rolling({dim: window}, center=True, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    
    std = xr.where(std == 0, np.nan, std)
    z_scores = (variable - mean) / std
    
    spikes = np.abs(z_scores) > threshold
    spikes.name = "spikes"
    
    return spikes

def cusum_test(x, dim='time'):
    """
    Cumulative sum (CUSUM) test for change point detection.
    Returns the most likely change point index and the test statistic.
    """
    mean = x.mean(dim=dim, skipna=True)
    std = x.std(dim=dim, skipna=True)
    
    # Avoid division by zero
    std = xr.where(std == 0, np.nan, std)
    
    # Cumulative sum of deviations
    S = (x - mean).cumsum(dim=dim, skipna=False)
    SS = S / std
    
    SS_abs = np.abs(SS)
    change_point = SS_abs.argmax(dim=dim, skipna=True).values
    
    R = SS_abs.max(dim=dim, skipna=True) - SS_abs.min(dim=dim, skipna=True)
    
    n = x.sizes[dim]
    test_statistic = R / np.sqrt(n)
    
    return change_point, test_statistic, S

def rolling_gradient(ds, dim, window):
    """
    Calculate gradient over rolling windows using covariance/variance.

    Parameters
    ----------
    ds : xarray.DataArray
        Input data array.
    dim : str
        Dimension along which to compute the rolling gradient.
    window : int
        Size of the rolling window (must be odd for centered windows).

    Returns
    -------
    xarray.DataArray
        DataArray of gradient with same shape as input (NaNs at edges).
    """

    # Construct rolling windows
    rolled = ds.rolling({dim: window}, center=True).construct("window")

    # x values (indices)
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    # Compute mean of y (amplitude) within each window
    y_mean = rolled.mean(dim="window")

    # Compute covariance and slope
    covariance = ((x - x_mean) * (rolled - y_mean)).sum(dim="window")
    gradient = covariance / x_var

    return gradient


def detect_decay_error(ds):
     """
     Detect signal outliers based on linear decay of signal.

     Compute mean value and standard deviation of amplitude for bin 1 and bin 15 for each time profile.
     Set start and mid thresholds to mean values + 3 standard deviations.
     Compute slope between start and mid thresholds, then extrapolate to get end expected value.
     Set end threshold as end expected value*1.2.
     Compute threshold linear decay based on start and end threshold.
     Everything above is masked as unrelistic decay.

     Parameters
     ----------
     ds : xarray.Dataset

     Returns:
     -------
     mask_per beam : xarray.Dataset
          Mask with True for nonvalid values.
     """

     # Finding mean values of amplitude for the beams, each time profile
     amp_start = ds['amp'].isel(range=slice(0, 1)).mean(dim=['beam','range'])    # bin 1
     amp_mid = ds['amp'].isel(range=slice(14, 15)).mean(dim=['beam','range'])    # bins 15 
     
     # compute standard deviations
     amp_start_std = ds['amp'].isel(range=slice(0, 1)).std(dim=['beam','range'])    # bin 1
     amp_mid_std = ds['amp'].isel(range=slice(14, 15)).std(dim=['beam','range'])    # bins 15

     start_threshold = amp_start + 3*amp_start_std  # allow for marginal
     mid_threshold = amp_mid + 3*amp_mid_std

     slope = (mid_threshold - start_threshold) / 14  # range bin differenece is 14
     n_bins = len(ds['range'])
     end_threshold = (start_threshold + slope * n_bins) *1.2  # allow for slightly higher threshold

     thresholds = np.linspace(start_threshold.values, end_threshold.values, n_bins)

     thresholds = xr.DataArray(
     thresholds,
     dims=['range', 'time'],
     coords={'range': ds['range'].values, 'time': ds['time'].values}
     )

     mask_per_beam = ds['amp'] > thresholds

     return mask_per_beam

# ###########################################################################
# functions related to application
# ###########################################################################

def apply_qc_mask(mask, true_values, qc1, qc2):
    """
    Apply a mask to update primary and secondary quality flag values.
    
    Parameters:
        mask: xarray DataArray boolean mask
        true_values: tuple (qc1_value, qc2_value) where mask is True
        qc1: existing qc1 array (kept where mask is False)
        qc2: existing qc2 array (kept where mask is False)
    
    Returns:
        qc1, qc2: xarray DataArrays with updated values
    """
    qc1 = xr.where(mask, true_values[0], qc1)
    qc2 = xr.where(mask, true_values[1], qc2)
    
    return qc1, qc2

def summarize_qc(ds, qc_var):
    qc = ds[qc_var]
    
    # Read attributes
    flag_values = qc.attrs.get('flag_values', [])
    flag_meanings = qc.attrs.get('flag_meanings', '').split()
    
    # Build mapping
    mapping = dict(zip(flag_values, flag_meanings))
    
    # Count occurrences
    counts = qc.to_series().value_counts()
    total_count = counts.sum()
    
    # Print summary
    print(f"QC Summary for '{qc_var}':")
    for flag in flag_values:
        count = (qc == flag).sum().item()
        label = mapping.get(flag, "unknown")
        pct = (count / total_count) * 100
        print(f"{label}: {count} ({pct:.2f}%)")
    
    print("\n")


# ###########################################################################
# functions for visualization
# ###########################################################################

def plot_masked(mask, array, mask_name):
    """
    Plot masked array values.
    
    Parameters:
        mask: xarray DataArray boolean mask
        array: xarray DataArray to plot
        mask_name: str, description of the mask for the title
    """
    array[0].where(mask).plot(x='time')
    plt.title(f'Values flagged as {mask_name} (dir=E)')
    plt.show()

def plot_qc_primary(ds, direction=0):
    """
    Plot primary QC flags for chosen direction.
    
    Parameters:
        ds: xarray Dataset containing 'vel_qc_primary'
        direction: int, index for direction (0=E, 1=N, 2=U1, 3=U2)
    """
    
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}
    
    # Define discrete colormap for QC values
    colors = [
        "#79D97A",  # 1: Good
        "#8E8E8E",  # 2: Unknown
        "#EDB24A",  # 3: Questionable
        "#DD6B60",  # 4: Bad
        "#4F4F4F",  # 9: Missing
    ]
    
    qc_labels = ['Good', 'Unknown', 'Potentially correctable bad data', 'Bad', 'Missing']
    
    cmap = ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 9.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Select the variable to plot
    if 'vel_qc_primary' in ds:
        data = ds['vel_qc_primary'][direction]
    
    # Plot without colorbar - wider figure
    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    
    # Create legend patches
    legend_patches = [Patch(facecolor=c, edgecolor='black', label=lbl) 
                      for c, lbl in zip(colors, qc_labels)]
    
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.85), title='Primary Flag')
    ax.set_title(f'Primary Flags for velocity (dir={dir_labels.get(direction, direction)})')
    plt.ylabel('Range (m)')
    ax.set_xlabel('Time (months)')
    plt.tight_layout()
    plt.show()

def plot_qc_secondary(ds, direction=0):
    """
    Plot secondary QC flags with legend box instead of colorbar.
    """
    dir_labels = {0: 'E', 1: 'N', 2: 'U1', 3: 'U2'}

    colors = [
        "#79D97A",  # passed_all_tests
        "#8E8E8E",  # unknown
        "#4F4F4F",  # missing_data
        "#F2CE5B",  # pressure_error
        "#6FDEE0",  # compass_heading_error
        "#EDB24A",  # time_error
        "#5F86E6",  # velocity_spike
        "#C06AD0",  # below_correlation_threshold
        "#DD6B60",  # signal_amplitude_outliers
        "#E78AC3",  # above_surface
    ]

    labels = [
        'Passed all tests',
        'Unknown',
        'Missing data',
        'Pressure error',
        'Compass heading error',
        'Time error',
        'Velocity spike',
        'Below correlation threshold',
        'Signal amplitude error',
        'Above surface',
    ]

    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(9.7, 5))

    # Plot WITHOUT colorbar
    ds['vel_qc_secondary'][direction].plot(
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=False
    )

    # Create legend patches (same idea as your primary function)
    legend_patches = [
        Patch(facecolor=c, edgecolor='black', label=lbl)
        for c, lbl in zip(colors, labels)
    ]

    ax.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.02, 0.7),
        title='Secondary Flag'
    )

    ax.set_title(f'Secondary Flags for velocity (dir={dir_labels.get(direction, direction)})')
    ax.set_ylabel('Range (m)')
    ax.set_xlabel('Time (months)')

    plt.tight_layout()
    plt.show()

# ###########################################################################
# could potentially be removed
# ###########################################################################

def print_data_removed(ds_before, ds_after, variable):
    """
    Prints variable data removed and the percentage. 
    """
    before = np.isfinite(ds_before[variable].values).sum()  
    after = np.isfinite(ds_after[variable].values).sum()
    removed = before - after
    percent_removed = (removed / before) * 100

    print(f"Data removed for {variable}: {removed} ({percent_removed:.2f}%)")

def interp_profile(vel, coord, coord_new):
    """Interpolate a single velocity profile from time varying coordinate to constant coordinate."""
    f = interp1d(coord, vel, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(coord_new)


def plot_profiles(vel, range_coord='range', range_slice=(5, 25),
                  mean_color='green', alpha_grey = 0.4,
                  xlabel='V (m/s)', ylabel='Range (m)', title='ADCP V',
                  figsize=(4, 4), ax=None):
    """
    Plot individual velocity profiles (gray) and the time-mean profile.

    Parameters
    ----------
    vel : xarray.DataArray
        Velocity with dims (range, time) — e.g. vel_h[1] or ds['vel'][0].
    range_coord : str
        Name of the range coordinate on `vel`.
    range_slice : tuple(float, float) or None
        (min, max) range to select. Use None to keep full range.
    mean_color : {'green', 'blue'}
        Color of the mean profile line.
    xlabel, ylabel, title : str
        Axis labels and title.
    figsize : tuple
        Figure size, used only if `ax` is None.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot into. If None, a new figure/axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    color_map = {'green': 'limegreen', 'blue': 'dodgerblue'}
    if mean_color not in color_map:
        raise ValueError(f"mean_color must be one of {list(color_map)}, got {mean_color!r}")
    mean_c = color_map[mean_color]

    # Optional range subsetting
    if range_slice is not None:
        vel = vel.sel({range_coord: slice(*range_slice)})

    v = vel.values                       # (range, time)
    r = vel[range_coord].values          # (range,)

    # Ensure orientation is (range, time)
    if v.shape[0] != r.size and v.shape[1] == r.size:
        v = v.T

    v_mean = np.nanmean(v, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(v, r, color='gray', linewidth=0.5, alpha=alpha_grey)
    ax.plot(v_mean, r, color=mean_c, linewidth=3)

    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    return ax

def speed(df):
    """Speed."""
    return np.sqrt(df[0]**2 + df[1]**2)

def direction(df):
    """Direction."""
    return (np.degrees(np.arctan2(df[0], df[1]))) % 360