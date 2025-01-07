import numpy as np
import gzip
import nibabel as nib
from nibabel.filebasedimages import ImageFileError


def load_gii(img):
    """
    Copy from abagen.utils.load_gifti
    Loads gifti file `img`

    Will try to gunzip `img` if gzip is detected, and will pass pre-loaded
    GiftiImage object

    Parameters
    ----------
    img : os.PathLike or nib.GiftiImage object
        Image to be loaded

    Returns
    -------
    img : nib.GiftiImage
        Loaded GIFTI images
    """
    try:
        img = nib.load(img)
    except (ImageFileError, TypeError) as err:
        # it's gzipped, so read the gzip and pipe it in
        if isinstance(err, ImageFileError) and str(err).endswith('.gii.gz"'):
            with gzip.GzipFile(img) as gz:
                img = nib.GiftiImage.from_bytes(gz.read())
        # it's not a pre-loaded GiftiImage so error out
        elif isinstance(err, TypeError) and not isinstance(
            img, nib.gifti.gifti.GiftiImage
        ):
            raise err

    return img


def save_gii(array, file=None):
    agg_data = array.astype(np.float32)
    img = nib.gifti.GiftiDataArray(agg_data)
    img = nib.gifti.GiftiImage(darrays=[img])
    if file is not None:
        img.to_filename(file)
    else:
        return img


def load_cifti(img):
    """
    Load a CIFTI image file using nibabel.

    Parameters:
    img (str or nibabel image): The file path to the CIFTI image or a pre-loaded nibabel image object.

    Returns:
    nibabel image: The loaded CIFTI image object.

    Raises:
    ImageFileError: If the file cannot be loaded as a CIFTI image.
    TypeError: If the input is not a valid CIFTI image or file path.
    """
    try:
        img = nib.load(img)
    except (ImageFileError, TypeError) as err:
        # it's gzipped, so read the gzip and pipe it in
        if isinstance(err, ImageFileError) and str(err).endswith('.nii.gz"'):
            with gzip.GzipFile(img) as gz:
                img = nib.load(gz)
        # it's not a pre-loaded GiftiImage so error out
        elif isinstance(err, TypeError) and not isinstance(
            img, nib.cifti2.cifti2.Cifti2Image
        ):
            raise err

    return img


def volume_from_cifti(data, axis):
    """
    Convert CIFTI-2 data to a NIfTI-1 volume.

    Parameters:
    data (numpy.ndarray): The CIFTI-2 data array.
    axis (nibabel.cifti2.BrainModelAxis): The BrainModelAxis object containing
        information about the brain model axis, including volume mask, voxel
        indices, volume shape, and affine transformation.

    Returns:
    nibabel.Nifti1Image: A NIfTI-1 image containing the volumetric data.
    """
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]  # Assume brainmodels axis is last, move it to front
    volmask = axis.volume_mask  # Which indices on this axis are for voxels?
    vox_indices = tuple(axis.voxel[volmask].T)  # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(
        axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
        dtype=data.dtype,
    )
    vol_data[vox_indices] = data  # "Fancy indexing"
    return nib.Nifti1Image(
        vol_data, axis.affine
    )  # Add affine for spatial interpretation


def surf_data_from_cifti(data, axis, surf_name):
    """
    Extracts surface data from a CIFTI file based on the specified surface name.

    Parameters:
    data (numpy.ndarray): The data array from which surface data is to be extracted.
    axis (nib.cifti2.BrainModelAxis): The BrainModelAxis object that contains information about the brain structures.
    surf_name (str): The name of the surface structure to extract data for.

    Returns:
    numpy.ndarray: The extracted surface data corresponding to the specified surface name.

    Raises:
    ValueError: If no structure with the specified surface name is found in the BrainModelAxis.
    """
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for (
        name,
        data_indices,
        model,
    ) in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:  # Just looking for a surface
            data = data.T[
                data_indices
            ]  # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex  # Generally 1-N, except medial wall vertices
            surf_data = np.zeros(
                (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype
            )
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")
