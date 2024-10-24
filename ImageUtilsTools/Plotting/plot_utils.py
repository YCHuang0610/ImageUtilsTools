import os
import nibabel as nib
import numpy as np
import pandas as pd

from surfplot import Plot
from surfplot.utils import threshold

from neuromaps.transforms import mni152_to_fslr
from neuromaps.datasets import fetch_fslr


def map_array_to_label(array, label):
    """
    Maps an array of values to a label file.

    Parameters:
    array (ndarray): Array of values, shape = (num_labels,)
    label (str): Path to the label file, shape = (num_vertices,)

    Returns:
    ndarray: Array of values mapped to the label file, shape = (num_vertices,)

    Raises:
    AssertionError: If the length of the array is not equal to the number of region labels.

    """
    if isinstance(array, pd.Series):
        array = array.values
    if isinstance(label, str):
        label = nib.load(label).agg_data()
    else:
        assert isinstance(
            label, nib.gifti.gifti.GiftiImage
        ), "label should be a path or a GiftiImage object"
        label = label.agg_data()
    # 0 is the medial wall, so we need to exclude it
    unique_label = np.unique(label)
    unique_label = unique_label[unique_label != 0]
    assert len(array) == len(
        unique_label
    ), "The length of array should be equal to the number of region labels"
    # 将array按照label匹配到的顺序排列，除0之外
    new_array = np.zeros(len(label))
    for i in range(len(unique_label)):
        new_array[label == unique_label[i]] = array[i]
    return new_array


def map_array_LR_to_label(array_LR, lh_parc, rh_parc):
    """
    Maps the left and right hemisphere arrays to their respective labels.

    Args:
        array_LR (numpy.ndarray): The input array containing both left and right hemisphere data.
        lh_parc (str): The file path to the left hemisphere parcellation file.
        rh_parc (str): The file path to the right hemisphere parcellation file.

    Returns:
        tuple: A tuple containing the mapped left hemisphere array and the mapped right hemisphere array.
    """
    if isinstance(lh_parc, str):
        label_num = len(np.unique(nib.load(lh_parc).agg_data())) - 1
    else:
        assert isinstance(
            lh_parc, nib.gifti.gifti.GiftiImage
        ), "lh_parc should be a path or a GiftiImage object"
        label_num = len(np.unique(lh_parc.agg_data())) - 1
    array_L = array_LR[:label_num]
    array_R = array_LR[label_num:]
    map_array_L = map_array_to_label(array_L, lh_parc)
    map_array_R = map_array_to_label(array_R, rh_parc)
    return map_array_L, map_array_R


def Plot_MySurf_VertexWise(
    left_data,
    right_data,
    lh,
    rh,
    cmap="viridis",
    color_range=None,
    cbar=True,
    title=None,
    size=(500, 400),
    layout="grid",
    views=None,
    brightness=0.5,
):
    p = Plot(
        lh,
        rh,
        size=size,
        layout=layout,
        views=views,
        mirror_views=True,
        brightness=brightness,
    )
    if color_range is not None:
        p.add_layer(
            {"left": left_data, "right": right_data},
            cbar=cbar,
            cmap=cmap,
            color_range=color_range,
        )
    else:
        p.add_layer({"left": left_data, "right": right_data}, cbar=cbar, cmap=cmap)
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_MySurf_mni152Volume(
    img,
    two_side=True,  # 'two_side', 'one_side'
    suface_type="inflated",
    cutoff=None,
    **kwargs,
):
    gii_lh, gii_rh = mni152_to_fslr(img)

    if cutoff is not None:
        data_lh = threshold(gii_lh.agg_data(), cutoff, two_sided=two_side)
        data_rh = threshold(gii_rh.agg_data(), cutoff, two_sided=two_side)
    else:
        data_lh = gii_lh.agg_data()
        data_rh = gii_rh.agg_data()

    # mask medial wall
    medwall = os.path.abspath(os.path.join(os.path.dirname(__file__), 'medwall.tsv'))
    medwall = np.loadtxt(medwall).astype(int)
    data_rl = np.concatenate([data_lh, data_rh], axis=0)
    data_rl[medwall==1] = np.nan
    data_lh, data_rh = np.split(data_rl, 2)

    surfaces = fetch_fslr()
    lh, rh = surfaces[suface_type]
    figure = Plot_MySurf_VertexWise(
        data_lh,
        data_rh,
        lh,
        rh,
        **kwargs,
    )
    return figure


def Plot_MySurf_RegionWise(
    array_LR,
    lh_parc,
    rh_parc,
    lh,
    rh,
    cmap="viridis",
    color_range=None,
    cbar=True,
    title=None,
    size=(500, 400),
    layout="grid",
    views=None,
    brightness=0.5,
):
    """
    Plot a surface region-wise.

    Args:
        array_LR (numpy.ndarray): Array to be plotted on the surface.
        lh_parc (numpy.ndarray): Left hemisphere parcellation array.
        rh_parc (numpy.ndarray): Right hemisphere parcellation array.
        lh (nibabel.nifti1.Nifti1Image): Left hemisphere surface image.
        rh (nibabel.nifti1.Nifti1Image): Right hemisphere surface image.
        cmap (str, optional): Colormap to be used for plotting. Defaults to 'viridis'.
        color_range (tuple, optional): Range of values to be mapped to colors. Defaults to None.
        cbar (bool, optional): Whether to show the colorbar. Defaults to True.
        title (str, optional): Title of the plot. Defaults to None.
        size (tuple, optional): Size of the plot figure. Defaults to (500, 400).

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    # map array to label
    array_LR = np.array(array_LR)
    map_array_L, map_array_R = map_array_LR_to_label(array_LR, lh_parc, rh_parc)
    p = Plot(
        lh,
        rh,
        size=size,
        layout=layout,
        views=views,
        mirror_views=True,
        brightness=brightness,
    )
    if color_range is not None:
        p.add_layer(
            {"left": map_array_L, "right": map_array_R},
            cbar=cbar,
            cmap=cmap,
            color_range=color_range,
        )
    else:
        p.add_layer({"left": map_array_L, "right": map_array_R}, cbar=cbar, cmap=cmap)
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_MySurf_RegionWise_OneHemi(
    array_single_hemi,
    parc,
    surf,
    cmap="viridis",
    color_range=None,
    cbar=True,
    title=None,
    size=(500, 400),
    layout="grid",
    views=None,
    brightness=0.5,
):
    array_single_hemi = np.array(array_single_hemi)
    map_array = map_array_to_label(array_single_hemi, parc)
    p = Plot(surf_lh=surf, size=size, layout=layout, views=views, brightness=brightness)
    if color_range is not None:
        p.add_layer(map_array, cbar=cbar, cmap=cmap, color_range=color_range)
    else:
        p.add_layer(map_array, cbar=cbar, cmap=cmap)
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_Each_Region_Num(region_num, parc_hemi, surf_hemi, size=(500, 200)):
    """
    Plot each region number on a surface.

    Args:
        region_num (int): The region number to plot.
        parc_hemi (str): The path to the parcellation hemisphere file.
        surf_hemi (str): The path to the surface hemisphere file.
        size (tuple, optional): The size of the plot. Defaults to (500, 200).

    Returns:
        figure: The generated plot figure.
    """
    label = nib.load(parc_hemi).agg_data()
    # 使label除0之外的值从1开始
    unique_label = np.unique(label)
    unique_label = unique_label[unique_label != 0]
    min_label = np.min(unique_label)
    label = label - min_label + 1
    # 画图
    regions = np.where(np.isin(label, region_num), label, 0)
    p = Plot(surf_hemi, size=size)
    p.add_layer(regions, cmap="tab20", cbar=False)
    p.add_layer(regions, cmap="gray", as_outline=True, cbar=False)
    figure = p.build()
    return figure
