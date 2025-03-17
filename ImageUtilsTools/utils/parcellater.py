import numpy as np
import nibabel as nib
from pathlib import Path


def parcellate_surface_data(array, parcel, method="mean"):
    """
    Parcellate surface data according to a parcellation file, ignoring regions labeled as 0 (medial wall).

    Parameters
    ----------
    array : numpy.ndarray
        Surface data with shape (number_of_vertices, feature_dimension)
    parcel : str, Path, numpy.ndarray, or nibabel.gifti.GiftiImage
        Parcellation file or array containing parcellation information

    Returns
    -------
    parcel_data : numpy.ndarray
        Parcellated data with shape (number_of_valid_parcels, feature_dimension)
    labels : numpy.ndarray
        Valid parcel labels (excluding 0)
    """
    if isinstance(parcel, str) | isinstance(parcel, Path):
        parcel = nib.load(parcel).darrays[0].data
    elif isinstance(parcel, np.ndarray):
        pass
    elif isinstance(parcel, nib.gifti.gifti.GiftiImage):
        parcel = parcel.darrays[0].data
    else:
        raise ValueError(
            "parcel must be a path to a gifti file, a gifti object, or a numpy array"
        )

    assert (
        array.shape[0] == parcel.shape[0]
    ), "array and parcel must have the same number of vertices"

    parcel = parcel.astype(int)

    labels = np.unique(parcel)
    labels = labels[labels != 0]
    
    if array.ndim == 1:
        array = array[:, np.newaxis]
    parcel_data = np.zeros((len(labels), array.shape[1]))

    # Loop through each parcel and average the data within that parcel
    for i, label in enumerate(labels):
        mask = parcel == label
        if method == "mean":
            parcel_data[i] = np.mean(array[mask], axis=0)
        elif method == "median":
            parcel_data[i] = np.median(array[mask], axis=0)
        elif method == "sum":
            parcel_data[i] = np.sum(array[mask], axis=0)
        elif method == "max":
            parcel_data[i] = np.max(array[mask], axis=0)
        else:
            raise ValueError("method must be one of 'mean', 'median', 'sum', 'max'")

    return parcel_data


def reverse_parcellate_surface_data(array, parcel):
    """
    Reverse parcellation of surface data according to a parcellation file, ignoring regions labeled as 0 (medial wall).

    Parameters
    ----------
    array : numpy.ndarray
        Parcellated data with shape (number_of_valid_parcels, feature_dimension)
    parcel : str, Path, numpy.ndarray, or nibabel.gifti.GiftiImage
        Parcellation file or array containing parcellation information

    Returns
    -------
    data : numpy.ndarray
        Surface data with shape (number_of_vertices, feature_dimension)
    """
    if isinstance(parcel, str) | isinstance(parcel, Path):
        parcel = nib.load(parcel).darrays[0].data
    elif isinstance(parcel, np.ndarray):
        pass
    elif isinstance(parcel, nib.gifti.gifti.GiftiImage):
        parcel = parcel.darrays[0].data
    else:
        raise ValueError(
            "parcel must be a path to a gifti file, a gifti object, or a numpy array"
        )

    parcel = parcel.astype(int)
    labels = np.unique(parcel)
    labels = labels[labels != 0]

    data = np.zeros((parcel.shape[0], array.shape[1]))

    # Loop through each parcel and assign the data to the corresponding vertices
    for i, label in enumerate(labels):
        mask = parcel == label
        data[mask] = array[i]

    return data