"""
spin_test_utils.py

Author: Yichun Huang
Date: 26/07/2024

This module provides functions to generate spin permutations of imaging data based on different methods.

"""

import warnings
import numpy as np
import pandas as pd
from neuromaps import images, nulls
from brainsmash.mapgen.base import Base


def generate_spin_permutation(
    data,
    parcellationLR,
    atlas="fsLR",
    density="32k",
    perm=1000,
    method="vasa",
    seed=1234,
    surfaces=None,
    **kwargs
):
    """
    Generate spin permutations of imaging data based on different methods.

    Parameters:
    - data (numpy.ndarray): The imaging data.
    - parcellationLR (str): The parcellation file path.
    - atlas (str, optional): The atlas used for parcellation. Defaults to "fsLR".
    - density (str, optional): The density of the parcellation. Defaults to "32k".
    - perm (int, optional): The number of permutations to generate. Defaults to 1000.
    - method (str, optional): The method used for spin permutation. Must be one of "vasa", "alexander_bloch",
      "vazquez_rodriguez", "baum", or "burt2020". Defaults to "vasa".
    - seed (int, optional): The random seed for reproducibility. Defaults to 1234.
    - **kwargs (optional): Additional keyword arguments specific to certain methods. If use 'burt2020' method,
      surfaces, left_dist_mat_file and right_dist_mat_file are required. Delta default is [0.1].

    Returns:
    - perm_imaging (numpy.ndarray): The generated spin permutations of the imaging data, with shape (n_regions, perm).
    """
    if isinstance(data, pd.Series):
        data = data.values
    data = np.squeeze(data)
    parcellation = images.relabel_gifti(parcellationLR)

    # Check if the number of parcels in parcellation is consistent with the number of regions in imaging data
    one_hemi_perc_num = len(np.unique(parcellation[0].darrays[0].data)) - 1
    if one_hemi_perc_num * 2 != data.shape[0]:
        warnings.warn(
            "The number of parcels in parcellation is not consistent with the number of regions in imaging data, \
             \nwe assume that only left hemisphere is provided..."
        )
        assert (
            one_hemi_perc_num == data.shape[0]
        ), "Even the left hemisphere is not consistent, please check"
        only_left = True
        fake_right = np.zeros_like(data)
        data = np.concatenate((data, fake_right), axis=0)
    else:
        only_left = False

    # Generate spin permutation based on the selected method
    assert method in [
        "vasa",
        "alexander_bloch",
        "vazquez_rodriguez",
        "baum",
        "burt2020",
    ], "method must be one of vasa, alexander_bloch, vazquez_rodriguez, baum, burt2020"
    if method == "vasa":
        perm_imaging = nulls.vasa(
            data,
            atlas=atlas,
            density=density,
            n_perm=perm,
            parcellation=parcellation,
            surfaces=surfaces,
            seed=seed,
        )
    elif method == "alexander_bloch":
        perm_imaging = nulls.alexander_bloch(
            data,
            atlas=atlas,
            density=density,
            n_perm=perm,
            parcellation=parcellation,
            surfaces=surfaces,
            seed=seed,
        )
    elif method == "vazquez_rodriguez":
        perm_imaging = nulls.vazquez_rodriguez(
            data,
            atlas=atlas,
            density=density,
            n_perm=perm,
            parcellation=parcellation,
            surfaces=surfaces,
            seed=seed,
        )
    elif method == "baum":
        perm_imaging = nulls.baum(
            data,
            atlas=atlas,
            density=density,
            n_perm=perm,
            parcellation=parcellation,
            surfaces=surfaces,
            seed=seed,
        )
    elif method == "burt2020":
        # https://doi.org/10.1016/j.neuroimage.2020.117038
        if "left_dist_mat_file" not in kwargs:
            raise ValueError(
                "left_dist_mat_file is required for burt2020 method, \
                if u dont have one, \
                go to https://drive.google.com/drive/folders/1HZxh7aOral_blIQHQkT7IX525RaMyjPp, \
                or https://brainsmash.readthedocs.io/en/latest/gettingstarted.html#computing-a-cortical-distance-matrix"
            )
        if "right_dist_mat_file" not in kwargs:
            raise ValueError("right_dist_mat_file is required for burt2020 method")
        left_dist_mat_file = kwargs["left_dist_mat_file"]
        right_dist_mat_file = kwargs["right_dist_mat_file"]
        # split data into left and right hemisphere
        data = np.squeeze(data)
        lh_data = data[: len(data) // 2]
        rh_data = data[len(data) // 2 :]

        # delta
        deltas = kwargs.get(
            "deltas",
            [
                0.1,
            ],
        )
        deltas = np.array(deltas)

        # left
        base = Base(x=lh_data, D=left_dist_mat_file, deltas=deltas, seed=seed)
        left_surrogates = base(n=perm)
        left_surrogates = left_surrogates.T
        # right
        base = Base(x=rh_data, D=right_dist_mat_file, deltas=deltas, seed=seed)
        right_surrogates = base(n=perm)
        right_surrogates = right_surrogates.T
        perm_imaging = np.concatenate((left_surrogates, right_surrogates), axis=0)

    if only_left:
        perm_imaging = perm_imaging[:one_hemi_perc_num, :]

    return perm_imaging
