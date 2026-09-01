"""
Provides functions related to reading and converting MOHID output files.

Functions (original)
---------
    read_files
        Scans a folder for MOHID .srh timeseries files and reads them.

Author: Karolina Anurova-Prykhodko



Functions (not original)
---------
From MARETEC (https://github.com/maretec/MOHID_python_tools):
    MOHIDHdf5toNetcdf
        Reads a MOHID Hdf5 file and converts it into Netcdf.

    get_mohid_timeseries_1_file
        Reads a MOHID timeseries file

From SMS-Coastal (https://github.com/fmmendonca/SMS-Coastal):
    convert2hdf5
        Converts a Netcdf to Hdf5
    chklog
        Check MOHID log if convert2hdf5 was succecful
"""

from datetime import datetime
from pathlib import Path
import re
import pandas as pd
import h5py
import numpy as np
import xarray as xr
from os import chdir, getcwd, path
from shutil import copyfile
from subprocess import run

def read_files(base_dir, folder, name):
    """
    Read all `.srh` files in `base_dir / folder` whose names are
    `name_1`, `name_2`, ...

    Parameters
    ----------
    base_dir : Path
    folder : str
        Subfolder inside `base_dir` (e.g. 'Run1').
    name : str
        Base file name without the `_<index>` suffix (e.g. 'layer_17').

    Returns
    -------
    dict
        Mapping from the suffix index to a each dataframe.
    """
    folder_path = Path(base_dir) / folder

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder_path.resolve()}")

    pattern = re.compile(
        rf"^{re.escape(name)}_(\d+)\.srh$",
        re.IGNORECASE
    )

    results = {}

    for srh_path in folder_path.glob("*.srh"):
        match = pattern.match(srh_path.name)

        if not match:
            continue

        idx = int(match.group(1))

        try:
            results[idx] = get_mohid_timeseries_1_file(str(srh_path))
        except Exception as e:
            print(f"Warning: failed to read {srh_path.name}: {e}")

    if not results:
        raise FileNotFoundError(
            f"No .srh files matching '{name}_<index>.srh' found in {folder_path.resolve()}"
        )

    return dict(sorted(results.items()))

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

    return t, dates

def MOHIDHdf5toNetcdf_wind(filename, dates= [['0']], in_t=0, file_stride=0, outdir=''):
    f = h5py.File(filename,'r')
    
    dims = ['lat','lon']
    coords={'lat': (('lat'),f['Grid']['Latitude'][0,:-1]),
            'lon': (('lon'),f['Grid']['Longitude'][:-1,0])}
    
    TimeList = list(f['Time'].keys())
    
    dates_in = dates
    
#    #Writing variables from 'Results'
    var_to_write=['wind velocity X', 'wind velocity Y']
    var_to_read=['wind velocity X', 'wind velocity Y']  
    t=in_t

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
                    ds.to_netcdf(outdir+'Atmosphere_'+str(t).zfill(4)+'.nc',mode='w')
                elif k > 0 :
                    ds.to_netcdf(outdir+'Atmosphere_'+str(t).zfill(4)+'.nc',mode='a')
                k=k+1        
            t=t+1
        
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
                ds.to_netcdf(outdir+'Atmosphere_'+str(t).zfill(4)+'.nc',mode='a')
                k=k+1
            t=t+1

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

def chklog(flog) -> int:
    """Check for the successful message in a MOHID log file."""

    status = 1
    dat = open(flog, "r")
    line = dat.readline()

    while line:
        if ("Program" and "successfully terminated") in line:
            line = None
            status = 0
            continue
        line = next(dat, None)

    dat.close()
    return status

def convert2hdf5(mohidir: str, mdat: str, outdir: str) -> int:
    """Runs the program Convert2Hdf5.exe from MOHID.
    Relative paths are allowed.
    
    Keyword arguments
    - mohidir: path to the directory with the MOHID tool and its libraries;
    - mdat: name and path to the ConvertToHDF5Action.dat file;
    - outdir: output file directory, where the log will also be written.
    """

    # Set paths:
    curdir = getcwd()
    mohidir = path.abspath(mohidir)
    mdat = path.abspath(mdat)
    
    # Check MOHID files:
    if not path.isdir(mohidir):
        print("[ERROR] m_supp_mohid: MOHID directory not found")
        return 1
    elif not path.isfile(mdat):
        print("[ERROR] m_supp_mohid: MOHID conversion .dat file not found")
        return 1
    
    mlog = path.join(path.abspath(outdir), "mohidlog.dat")
    copyfile(mdat, path.join(mohidir, "ConvertToHDF5Action.dat"))

    mfiles = (
        "Convert2Hdf5.exe", "hdf5.dll", "hdf5dll.dll", "hdf5_f90cstub.dll",
        "hdf5_fortran.dll", "hdf5_hl.dll", "hdf5_tools.dll", "libcurl.dll",
        "libiomp5md.dll", "msvcp140.dll", "msvcr120.dll", "netcdf.dll",
        "szip.dll", "szlibdll.dll", "vcruntime140.dll", "zlib.dll",
        "zlib1.dll",
    )
    mfiles = [path.join(mohidir, file) for file in mfiles]

    for file in mfiles:
        if path.isfile(file): continue
        print("[ERROR] m_supp_mohid: MOHID file missing:", file)
        return 1

    # Change to MOHID working directory and run program:
    print("Running MOHID Convert2Hdf5...")
    chdir(mohidir)

    with open("nomfich.dat", "w") as dat:
        dat.write("ROOT_SRT : .\\\n")
        dat.write("IN_MODEL : ConvertToHDF5Action.dat\n")

    cmd = "Convert2Hdf5.exe > " + mlog
    print(cmd)
    run(cmd, shell=True)
    chdir(curdir)

    return chklog(mlog)

