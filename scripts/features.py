"""
Module to compute the differen features for the UL use detection problem.

The features are: UL use, mean and variance acceleration components, mean and 
variance of acceleration norm, minimum and maximum of acceleration norm, and 
the Shannon entropy of the acceleration norm.

Author: Sivakumar Balasubramanian
Date: 31 August 2024
"""

import numpy as np
import scipy as sp

FEATURE_COLS = ["uluse", "mean_x", "mean_y", "mean_z",
                "var_x", "var_w", "var_z",
                "mean_x2", "mean_y2", "mean_z2",
                "var_x2", "var_w2", "var_z2",
                "min_x2", "min_y2", "min_z2",
                "max_x2", "max_y2", "max_z2",
                "ent_x", "ent_y", "ent_z",
                "outfb"]


# Compute the features for given data frame.
def get_uluse_from_raters(ratings: np.array) -> np.array:
    _mean = np.mean(ratings, axis=1)
    _uluse = _mean > 0.5
    _uluse[_mean == 0.5] = np.random.rand(np.sum(_mean == 0.5)) > 0.5
    return _uluse