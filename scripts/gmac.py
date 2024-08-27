
"""Module for with functions for computing gmac and analysing data from 
daily life.

Author: Sivakumar Balasubramanian
"""

import numpy as np
import pandas as pd
from scipy import signal


def estimate_pitch(accl: np.array, farm_inx: int, nwin: int) -> np.array:
    """
    Estimates the pitch angle of the forearm from the accelerometer data.
    """
    # Moving averaging using the causal filter
    acclf = signal.lfilter(np.ones(nwin) / nwin, 1, accl, axis=0) if nwin > 1 else accl
    # Compute the norm of the acceleration vector
    acclfn = acclf / np.linalg.norm(acclf, axis=1, keepdims=True)
    return -np.rad2deg(np.arccos(acclfn[:, farm_inx])) + 90


def estimate_accl_mag(accl: np.array, fs: float, fc: float, nc: int,
                      n_am: int) -> np.array:
    """
    Compute the magnitude of the accelerometer signal.
    """
    # Highpass filter the acceleration data.
    sos = signal.butter(nc, fc, btype='highpass', fs=fs, output='sos')
    accl_filt = np.array([signal.sosfilt(sos, accl[:, 0]),
                          signal.sosfilt(sos, accl[:, 1]),
                          signal.sosfilt(sos, accl[:, 2])]).T
    
    # Acceleration magnitude    
    amag = np.linalg.norm(accl_filt, axis=1)
    
    # Moving average filter
    # _input = np.append(np.ones(n_am - 1) * amag[0], amag)
    # _impresp = np.ones(n_am) / n_am
    # return np.convolve(_input, _impresp, mode='valid')
    return signal.lfilter(np.ones(n_am) / n_am, 1, amag, axis=0)


def estimate_gmac(accl: np.array, accl_farm_inx: int, Fs: float, params: dict) -> np.array:
    """
    Estimate GMAC for the given acceleration data and parameters.
    """
    # Estimate pitch and acceleration magnitude
    pitch = estimate_pitch(accl, accl_farm_inx, params["np"])
    accl_mag = estimate_accl_mag(accl, Fs, fc=params["fc"], nc=params["nc"],
                                 n_am=params["nam"])
    
    # Compute GMAC
    _pout = 1.0 * (pitch >= params["p_th"])
    _amout = 1.0 * (accl_mag > params["am_th"])
    return _pout * _amout


def estimate_gmac2(accl: np.array, accl_farm_inx: int, Fs: float, params: dict, full_output: bool = False) -> np.array:
    """
    Estimate GMAC for the given acceleration data and parameters.
    """
    # Estimate pitch and acceleration magnitude
    pitch = estimate_pitch(accl, accl_farm_inx, params["np"])
    accl_mag = estimate_accl_mag(accl, Fs, fc=params["fc"], nc=params["nc"],
                                 n_am=params["nam"])
    
    # Compute GMAC
    _pout = detector_with_hystersis(pitch, params["p_th"], params["p_th_band"])
    _amout = detector_with_hystersis(accl_mag, params["am_th"], params["am_th_band"])
    return (pitch, accl_mag, _pout * _amout) if full_output else (_pout * _amout) 


def detector_with_hystersis(x: np.array, th: float, th_band: float) -> np.array:
    """
    Implements a detector with hystersis.
    """
    y= np.zeros(len(x))
    for i in range(1, len(y)):
        if y[i-1] == 0:
            y[i] = 1 * (x[i] > th)
        else:
            y[i] = 1 * (x[i] >= (th - th_band))
    return y