"""
This module provides functions for calculating local correlation between two surface between two input.

Reference:
- https://github.com/elifesciences-publications/project_msm/blob/master/scripts/surflocalcorr.m
- Eichert N, Robinson EC, Bryant KL, Jbabdi S, Jenkinson M, Li L, Krug K, Watkins KE & Mars R. 2020. Cross-species cortical alignment identifies different types of anatomical reorganization in the primate temporal lobe. eLife https://doi.org/10.7554/eLife.53232

Functions:
- local_corr(x, y, coor, a, method="spearmanr"): Calculate the local correlation.
- surflocalcorr(x, y, sph, a=30, method="spearmanr", return_gifti=False): Calculate the surface local correlation between two input.
"""

import nibabel as nib
import numpy as np
from ..utils._spearman import spearman_r
from pathlib import PosixPath
import numba


def local_corr(x, y, coor, a, method="spearmanr"):
    """
    Calculate the local correlation between two arrays.

    Parameters:
    - x (ndarray): First input array.
    - y (ndarray): Second input array.
    - coor (ndarray): Array of coordinates.
    - a (float): Angle threshold in degrees.
    - method (str): Correlation method to use. Default is "spearmanr".

    Returns:
    - r (ndarray): Array of local correlation values.
    """
    v = coor / np.sqrt((coor**2).sum(axis=1))[:, np.newaxis]
    r = np.zeros(v.shape[0])
    # no values # It seems that the value 0 in the surface data is always considered as nan value, especially when the you plot it. But the value 0 is not a nan value, so I remove the condition x != 0 and y != 0.
    vals = np.logical_and(
        ~np.isnan(x), ~np.isnan(y)
    )

    for i in numba.prange(v.shape[0]):
        cos_angle = v @ v[i]
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        j = (np.degrees(angle) < a) & vals
        # Ensure enough values for stable correlation
        assert np.sum(j) > 30, "Not enough data points for correlation calculation"
        if method == "pearsonr":
            r[i] = np.corrcoef(x[j], y[j])[0, 1]
        elif method == "spearmanr":
            r[i] = spearman_r(x[j], y[j])

    r[~vals] = np.nan
    return r


def surflocalcorr(x, y, sph, a=30, method="spearmanr", return_gifti=False):
    """
    Calculate the surface local correlation between two input datasets.

    Parameters:
    x (str or nibabel.gifti.GiftiImage): Input dataset x. If a string is provided, it is assumed to be the path to the dataset and will be loaded using nibabel.
    y (str or nibabel.gifti.GiftiImage): Input dataset y. If a string is provided, it is assumed to be the path to the dataset and will be loaded using nibabel.
    sph (str or nibabel.gifti.GiftiImage): Spherical coordinates. If a string is provided, it is assumed to be the path to the dataset and will be loaded using nibabel.
    a (int, optional): Parameter a. Defaults to 30.
    method (str, optional): Correlation method. Can be pearsonr or spearmanr. Defaults to 'spearmanr'.
    return_gifti (bool, optional): Whether to return the result as a nibabel.gifti.GiftiImage object. Defaults to False.

    Returns:
    numpy.ndarray or nibabel.gifti.GiftiImage: Array of local correlation values. If return_gifti is True, the result is returned as a nibabel.gifti.GiftiImage object.

    """
    if isinstance(x, PosixPath):
        x = nib.load(x)
    if isinstance(x, str):
        x = nib.load(x)
    x = x.agg_data()
    if isinstance(y, PosixPath):
        y = nib.load(y)
    if isinstance(y, str):
        y = nib.load(y)
    y = y.agg_data()
    if isinstance(sph, PosixPath):
        sph = nib.load(sph)
    if isinstance(sph, str):
        sph = nib.load(sph)
    coordinates, _ = sph.agg_data()
    assert method in [
        "pearsonr",
        "spearmanr",
    ], "Invalid method. Must be 'pearsonr' or 'spearmanr'."
    corr = local_corr(x, y, coordinates, a, method)
    # 保存到Gifti对象
    if return_gifti:
        corr = corr.astype(np.float32)
        corr_gifti = nib.gifti.GiftiDataArray(corr)
        corr_gifti = nib.gifti.GiftiImage(darrays=[corr_gifti])
        return corr_gifti

    return corr
