"""
Provides functions related to reading MOHID output files.

Functions
---------
MOHIDHdf5toNetcdf
    Adjusted from MARETEC. Reads a MOHID Hdf5 file and converts it into Netcdf.

get_mohid_timeseries_1_file
    Copied from MARETEC.
    Reads a MOHID timeseries file

read_files
    Reads all MOHID timeseries files in a folder.

Author: Karolina Anurova-Prykhodko
"""

from datetime import datetime
from pathlib import Path
import re
import pandas as pd
import h5py
import numpy as np
import xarray as xr

def MOHIDHdf5toNetcdf(filename, dates= [['0']], in_t=0, file_stride=0, outdir=''):
    f = h5py.File(filename,'r')
    
    dims = ['lat','lon']
    coords={'lat': (('lat'),f['Grid']['Latitude'][0,:-1]),
            'lon': (('lon'),f['Grid']['Longitude'][:-1,0])}
    
    TimeList = list(f['Time'].keys())
    
    dates_in = dates
    
#    #Writing variables from 'Results'
    var_to_write=['temperature']
    var_to_read=['temperature']    
    t=in_t
    #stride = file_stride
    timeindex=-1
    for timestep in TimeList:
        readtime = 1
        timeindex=timeindex+1
        date=f['Time'][TimeList[timeindex]][:].transpose()
        date=''.join(str(e) for e in date)
        for sublist in dates:
            if sublist == date:
                readtime = 0
        if readtime == 1:
            dates=np.vstack((dates,date))
            k=0
            #if stride == file_stride:
            for var in var_to_read:
                TimeVar = var+timestep[-6:]
                if f['Results'][var][TimeVar].ndim == 2:
                    data = f['Results'][var][TimeVar][:,:].transpose()
                    temp = xr.DataArray(data, coords=coords, dims=dims)
                    temp.encoding['_FillValue'] = float(data.min())
                elif f['Results'][var][TimeVar].ndim > 2:
                    temp = xr.DataArray(f['Results'][var][TimeVar][-1,:,:].transpose(), coords=coords, dims=dims)
                ds=xr.Dataset({var_to_write[k]:temp})
                if k == 0:
                    ds.to_netcdf(outdir+'WaterProperties_'+str(t).zfill(4)+'.nc',mode='w')
                elif k > 0 :
                    ds.to_netcdf(outdir+'WaterProperties_'+str(t).zfill(4)+'.nc',mode='a')
                k=k+1        
            t=t+1
             #   stride = 0
            #stride=stride+1
        
    #Writing variables from 'Grid'
    var_to_write=['Bathymetry']
    var_to_read=['Bathymetry']
    t=in_t
    dates = dates_in
    #stride = file_stride
    timeindex=-1
    for timestep in TimeList:
        readtime = 1
        timeindex=timeindex+1
        date=f['Time'][TimeList[timeindex]][:].transpose()
        date=''.join(str(e) for e in date)
        for sublist in dates:
            if sublist == date:
                readtime = 0
        if readtime == 1:
            dates=np.vstack((dates,date))
            k=0
            #if stride == file_stride:
            for var in var_to_read:
                TimeVar = var+timestep[-6:]
                if f['Grid'][var].ndim == 2:
                    data = f['Grid'][var][:,:].transpose()
                    temp = xr.DataArray(data, coords=coords, dims=dims)
                    temp.encoding['_FillValue'] = float(data.min())
                elif f['Grid'][var].ndim > 2:
                    temp = xr.DataArray(f['Grid'][var][-1,:,:].transpose(), coords=coords, dims=dims)
                ds=xr.Dataset({var_to_write[k]:temp})        
                ds.to_netcdf(outdir+'WaterProperties_'+str(t).zfill(4)+'.nc',mode='a')
                k=k+1
            t=t+1
             #   stride = 0
            #stride=stride+1
    
    #Writing computed variables
    #var_to_compute=['Uavg','Vavg']
    return t, dates


def get_mohid_timeseries_1_file(file, remove_time_0=True):
    """
    Copied from MARETEC.

    Function to read a single MOHID timeseries file.
    """
    f = open(file)
    prev_l = ''
    while True:
        l = f.readline()
        if l == '':
            break
        if l.find('<BeginTimeSerie>') != -1:
            header = prev_l
            break
        prev_l = l
    data = []
    while True:
        l = f.readline()
        if l == '':
            break
        if l.find('<EndTimeSerie>') != -1:
            break
        data.append(l)
    f.close()

    header = header.strip(' \n')
    header = header.split(' ')
    header = list(filter(None, header))
    header = ['date'] + header[7:]
    data = [x.strip(' \n') for x in data]
    if remove_time_0:
        data.pop(0)
    data = [x.split(' ') for x in data]
    data = [list(filter(None, x)) for x in data]
    data = [[float(x) for x in y] for y in data]
    data = [[datetime(int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(float(x[6])))] + x[7:] for x in data]
    df = pd.DataFrame.from_records(data, columns=header, index=header[0])

    return df


def read_files(folder, name, base_dir):
    """
    Read all `.srh` and `.srw` files in `base_dir / folder` whose names are
    `name_1`, `name_2`, ... For each index, columns from the `.srw` file are
    appended to the `.srh` file.

    Parameters
    ----------
    folder : str
        Subfolder inside `base_dir` (e.g. 'Run1').
    name : str
        Base file name without the `_<index>` suffix (e.g. '36_17').
    base_dir : Path

    Returns
    -------
    dict
        Mapping from the suffix index (int) to the merged DataFrame.
    """
    folder_path = Path(base_dir) / folder

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder_path.resolve()}")

    # Match "<name>_<number>.srh" or "<name>_<number>.srw"
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.(srh|srw)$", re.IGNORECASE)

    # Collect files grouped by index: {idx: {'srh': path, 'srw': path}}
    grouped = {}
    for entry in folder_path.iterdir():
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            idx = int(m.group(1))
            ext = m.group(2).lower()
            grouped.setdefault(idx, {})[ext] = entry

    if not grouped:
        raise FileNotFoundError(
            f"No files matching '{name}_<index>.srh/.srw' found in {folder_path.resolve()}"
        )

    results = {}
    for idx in sorted(grouped):
        files = grouped[idx]
        srh_path = files.get('srh')
        srw_path = files.get('srw')

        try:
            if srh_path is not None:
                df_srh = get_mohid_timeseries_1_file(str(srh_path))
            else:
                df_srh = None
                print(f"Warning: no .srh file for index {idx}")

            if srw_path is not None:
                df_srw = get_mohid_timeseries_1_file(str(srw_path))
            else:
                df_srw = None

            if df_srh is not None and df_srw is not None:
                # Drop columns from srw that already exist in srh to avoid duplicates
                new_cols = [c for c in df_srw.columns if c not in df_srh.columns]
                # Align on index (usually the time index) and append the new columns
                merged = df_srh.join(df_srw[new_cols], how='left')
                results[idx] = merged
            elif df_srh is not None:
                results[idx] = df_srh
            elif df_srw is not None:
                results[idx] = df_srw

        except Exception as e:
            print(f"Warning: failed to read files for index {idx}: {e}")

    return results