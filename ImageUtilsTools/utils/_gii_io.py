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
