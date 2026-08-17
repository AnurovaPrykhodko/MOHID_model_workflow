# MOHID_model_workflow:

This repository contains a scientific Python workflow developed during my internship in numerical ocean modelling. The purpose is to:
- process environmental forcing data into MOHID model-ready format.
- process model output.
- evaluate model performance against observations of ADCP, tide gauge and satellite data.

The repository demostrates API data acquisition, oceanographic data processing, model validation metrics and visualisation, and reproducible analysis using Python modules and Jupyter Notebooks. 

# Repository structure:

## Modules/

#### ADCP.py
Functions for processing ADCP data, applying quality control, and visualising quality flags and data. 

#### read_files.py
Utilites for reading and converting MOHID output files.

#### metrics.py
Functions for comparing model output against a reference using common metrics, and utilities to apply them.

#### helpers.py
Helpers related to interpolating model output to match the ADCP data in order to compare them.

## Notebooks/

#### preprocessing/
- ADCP.ipynb: processes raw ADCP data to Netcdf and csv, applies quality control. 
- CERRA.ipynb: API download of CERRA data, processes and convert atmospheric forcing from Netcdf to Hdf5.
- discharge_conversion.ipynb: convert submarine discharge flow from xlsx time-series to .dat format.

#### validation/
- validation_ADCP.ipynb: reads and interpolate model output .srh files to the geometry of the ADCP observations, and compares them.
- validation_gauge.ipynb: Compares tide gauge waterlevel from Puertos del Estado and model output.
- validation_SST.ipynb: API download of ODYSSEA satellite data, reads  model output Hdf5 files, and compares between.

## mohid/
Tools to convert Netcdf data to Hdf5 from MARETEC (https://github.com/Mohid-Water-Modelling-System, GNU General Public License)