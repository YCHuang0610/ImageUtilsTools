import numpy as np
import gzip
import nibabel as nib
import seaborn as sns
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


class Save_Gii:
    """
    A utility class to create and save GIFTI (.gii) files in different modes:
    - array: Regular floating-point data. Often *.darray.gii files.
    - label: Integer label data with optional color table. Often *.dlabel.gii files.
    - surf: Surface data, requiring (vertices, faces). Often *.surf.gii files.
    """

    def __init__(self, array, file=None, gii_type="metric", colortable=None):
        """
        Initialize the Save_Gii class.

        Args:
            array: Numpy array for data, or a tuple (vertices, faces) for 'surf'.
            file: Output filename. If None, returns a GiftiImage object instead of writing a file.
            gii_type: One of 'array', 'label', or 'surf'. Determines which save_* method is called.
            colortable: Optional dict for label colors. Contains label to RGBA color mappings.
        """
        self.array = array
        self.file = file
        self.colortable = colortable
        self.img = None
        if gii_type not in ["metric", "label", "surf"]:
            raise ValueError("Invalid GIFTI type. Must be 'metric', 'label', or 'surf'.")
        elif gii_type == "metric":
            self.img = self.save_darray()
        elif gii_type == "label":
            self.img = self.save_dlabel()
        elif gii_type == "surf":
            self.img = self.save_surf()

    def save_darray(self):
        """
        Create a GiftiImage from a float32 array and optionally save to file.

        Returns:
            A nibabel.gifti.GiftiImage object.
        """
        agg_data = self.array.astype(np.float32)
        agg_data = list(nib.gifti.GiftiDataArray(data=data) for data in agg_data.T)
        img = nib.gifti.GiftiImage(darrays=agg_data)

        if self.file is not None:
            img.to_filename(self.file)

        return img

    def save_dlabel(self):
        """
        Create a label-type GiftiImage (int32) with an optional color table.

        Returns:
            A nibabel.gifti.GiftiImage object.
        """
        agg_data = self.array.astype(np.int32)
        agg_data = nib.gifti.GiftiDataArray(
            data=agg_data,
            intent="NIFTI_INTENT_LABEL",
            datatype="NIFTI_TYPE_INT32",
        )
        gifti_label_table = nib.gifti.GiftiLabelTable()
        unique_labels = np.unique(agg_data.data)

        if self.colortable is not None:
            label_colors = self.colortable
        else:
            label_colors = self.assign_colors(unique_labels)

        for label in unique_labels:
            if label <= 0:
                continue
            gifti_label = nib.gifti.GiftiLabel()
            gifti_label.key = int(label)
            gifti_label.label = f"Region_{label}"

            # Add color to label
            r, g, b, a = label_colors[label]
            gifti_label.red = r
            gifti_label.green = g
            gifti_label.blue = b
            gifti_label.alpha = a

            gifti_label_table.labels.append(gifti_label)

        img = nib.gifti.GiftiImage(darrays=[agg_data], labeltable=gifti_label_table)

        if self.file is not None:
            img.to_filename(self.file)

        return img

    def save_surf(self):
        """
        Create a surface-type GiftiImage from (vertices, faces).

        Raises:
            ValueError: If the input array is not a tuple of (vertices, faces).

        Returns:
            A nibabel.gifti.GiftiImage object.
        """
        # 如果是表面数据，则array为元祖，第一个元素为顶点坐标，第二个元素为面索引
        try:
            vertices, faces = self.array
        except ValueError:
            raise ValueError("Surface data must be a tuple of (vertices, faces)")
        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.int32)
        vertices = nib.gifti.GiftiDataArray(
            data=vertices,
            intent="NIFTI_INTENT_POINTSET",
        )
        faces = nib.gifti.GiftiDataArray(
            data=faces,
            intent="NIFTI_INTENT_TRIANGLE",
        )
        img = nib.gifti.GiftiImage(darrays=[vertices, faces])

        if self.file is not None:
            img.to_filename(self.file)

        return img

    @staticmethod
    def assign_colors(
        unique_labels, palette_name="bright", as_rgb_float=True, alpha=1.0
    ):
        """
        Assign colors to label values. Requires seaborn.

        Args:
            unique_labels: An array of label values (integers).
            palette_name: A seaborn color palette (e.g. 'husl', 'hls', 'bright').
            as_rgb_float: Return RGB in [0, 1] if True, else in [0, 255].
            alpha: Alpha (transparency) value for all labels.

        Returns:
            A dict mapping label value to an RGBA tuple.

        为有效标签分配颜色（支持超过20种颜色，需安装seaborn）

        参数：
            unique_labels : array-like
                包含所有唯一标签的数组（通常为整数）
            palette_name : str, 默认'husl'
                seaborn支持的调色板名称，如'husl', 'hls', 'bright', 'dark'等
            as_rgb_float : bool, 默认True
                返回RGB值是否为0-1浮点数（Matplotlib兼容格式）

        返回：
            label_to_color : dict
                标签到颜色的映射字典（无效标签<=0不会包含）
        """
        valid_labels = unique_labels[unique_labels > 0]
        n = len(valid_labels)

        # 生成颜色（使用seaborn的husl/hls等调色板）
        colors = sns.color_palette(palette_name, n_colors=n)

        # 转换为Matplotlib兼容的RGB浮点数组（若需要）
        if not as_rgb_float:
            colors = np.array(colors) * 255  # 转换为0-255整数格式

        # 添加alpha通道转换为RGBA
        colors = [(*rgb, alpha) for rgb in colors]

        # 创建映射字典
        label_to_color = dict(zip(valid_labels, colors))
        return label_to_color

    def __repr__(self):
        if self.file is not None:
            return f"Save gifti file to {self.file}"
        else:
            return f"Gifti image object is created"


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
    numpy.ndarray: The vertex indices for the extracted surface data.

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
            return surf_data, vtx_indices
    raise ValueError(f"No structure named {surf_name}")
