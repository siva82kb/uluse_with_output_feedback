"""
Other miscellaneous functions for GMAC analysis on daily data.
"""

import sys
import os
import pathlib
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def read_raw_data(data_path: str, colnames: list[str]) -> pd.DataFrame:
    """
    Read the raw data from the given path.
    """
    # Read the raw data
    _data = pd.read_csv(data_path, sep=',', header=None, index_col=False)
    _data.columns = colnames
    _data['datetime'] = pd.to_datetime(_data['datetime'])
    # Add file name for sanity check
    _data['filename'] = data_path.split(os.sep)[-1]
    return _data


def read_raw_date_data(data_path: str) -> pd.DataFrame:
    """
    Read the raw data.
    """
    # Read the raw data
    _data = pd.read_csv(data_path, sep=',', index_col=False)
    _data['datetime'] = pd.to_datetime(_data['datetime'])
    return _data


def organize_rawdata_datewise(arm, subj, base_data_dir, details, rd_cols, prcd_files):
    """
    Organize the raw data into date-wise files.
    """
    # Check if the processed folder exits.
    _prcddir = base_data_dir / "processed" / subj
    pathlib.Path(_prcddir).mkdir(exist_ok=True)

    # Remove all CSV files.
    for f in glob.glob(pathlib.Path(_prcddir / f"*{arm}*raw*.csv").as_posix()):
        os.remove(f)
    
    # Get the dates and their corresponding files
    arm_files = [_f for _f in details['files'][subj] if arm in _f]
    n_af = len(arm_files)
    for i, _f in enumerate(arm_files):
        if _f in prcd_files:
            continue
        # File not processed. Process.
        _fname = _f.split(os.sep)[-1]
        _str = f"\r{i:3d} / {n_af:3d} {subj:>4} {_fname:>25} {' ':>15}"
        sys.stdout.write(_str)
        # Read data
        _data = read_raw_data(data_path=_f, colnames=rd_cols)
        # Find the dates in the data.
        _dates = np.unique([_d.date() for _d in _data['datetime'] if _d.year >= 2018])
        
        # Check if a file for this date exits.
        for _d in _dates:
            # Find index from the current data with the date.
            _dinx = np.array([_dc.date() == _d for _dc in _data['datetime']])
            # If file exists, then open, append and save.
            _fname = _prcddir / f"{arm}_raw_{str(_d)}.csv"
            if pathlib.Path(_fname).is_file():
                sys.stdout.write(_str + f" {_d} Old")
                _data.loc[_dinx, :].to_csv(_fname, sep=",", mode='a',
                                           header=False, index=False)
            else:
                sys.stdout.write(_str + f" {_d} New")
                _data.loc[_dinx, :].to_csv(_fname, sep=",", index=False)
            sys.stdout.flush()
    return arm_files


def interp1d_rawdata(yvals, xold, xnew):
    """
    Interpolates the given data
    """
    return interp1d(xold, yvals, kind='linear', axis=0, fill_value='extrapolate')(xnew)
