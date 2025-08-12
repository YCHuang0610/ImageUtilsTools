"""
Author: [Yichun Huang]

This module contains functions and classes for plotting data on a brain surface.
Functions for surface plotting are mainly depend on the surfplot package.

"""

import os
import subprocess
import nibabel as nib
import numpy as np
import pandas as pd

from ..utils._gii_io import load_gii

from surfplot import Plot
from surfplot.utils import threshold

from neuromaps.transforms import mni152_to_fslr
from neuromaps.datasets import fetch_fslr


def plot_base(
    left_data=None,
    right_data=None,
    surf_lh=None,
    surf_rh=None,
    cmap="viridis",
    color_range=None,
    cbar=True,
    layout="grid",
    size=(500, 400),
    views=None,
    brightness=0.5,
    hemi="both",
    outline_parc_lh=None,
    outline_parc_rh=None,
    outline_cmap="gray",
    outline_alpha=1,
):
    if hemi == "both":
        p = Plot(
            surf_lh=surf_lh,
            surf_rh=surf_rh,
            size=size,
            layout=layout,
            views=views,
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

        if outline_parc_lh is not None and outline_parc_rh is not None:
            p.add_layer(
                {"left": outline_parc_lh, "right": outline_parc_rh},
                as_outline=True,
                cbar=False,
                cmap=outline_cmap,
                alpha=outline_alpha,
            )

    elif hemi == "left":
        p = Plot(
            surf_lh=surf_lh,
            size=size,
            layout=layout,
            views=views,
            brightness=brightness,
        )
        if color_range is not None:
            p.add_layer(left_data, cbar=cbar, cmap=cmap, color_range=color_range)
        else:
            p.add_layer(left_data, cbar=cbar, cmap=cmap)

        if outline_parc_lh is not None:
            p.add_layer(
                {"left": outline_parc_lh,},
                as_outline=True,
                cbar=False,
                cmap=outline_cmap,
                alpha=outline_alpha,
            )

    elif hemi == "right":
        p = Plot(
            surf_rh=surf_rh,
            size=size,
            layout=layout,
            views=views,
            brightness=brightness,
        )
        if color_range is not None:
            p.add_layer(right_data, cbar=cbar, cmap=cmap, color_range=color_range)
        else:
            p.add_layer(right_data, cbar=cbar, cmap=cmap)

        if outline_parc_rh is not None:
            p.add_layer(
                {"left": outline_parc_rh,},
                as_outline=True,
                cbar=False,
                cmap=outline_cmap,
                alpha=outline_alpha,
            )
    else:
        raise ValueError("hemi should be 'both', 'left' or 'right'")
    return p


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
    label = load_gii(label).agg_data()
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
    parc_data_L = load_gii(lh_parc).agg_data()
    parc_data_R = load_gii(rh_parc).agg_data()
    assert len(parc_data_L) == len(
        parc_data_R
    ), "The length of left and right hemisphere parcellation should be equal"
    label_num = len(np.unique(parc_data_L)) - 1
    array_L = array_LR[:label_num]
    array_R = array_LR[label_num:]
    map_array_L = map_array_to_label(array_L, lh_parc)
    map_array_R = map_array_to_label(array_R, rh_parc)
    return map_array_L, map_array_R


def remove_medial_wall(data_lh, data_rh, species="human"):
    if species == "human":
        medwall = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "medwall.tsv")
        )
    elif species == "monkey":
        medwall = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "medwall_monkey.tsv")
        )

    medwall = np.loadtxt(medwall).astype(int)
    data_rl = np.concatenate([data_lh, data_rh], axis=0)
    data_rl[medwall == 1] = np.nan
    data_lh, data_rh = np.split(data_rl, 2)
    return data_lh, data_rh


def Plot_MySurf_VertexWise(
    left_data,
    right_data,
    lh,
    rh,
    title=None,
    **kwargs,
):
    p = plot_base(
        left_data=left_data,
        right_data=right_data,
        surf_lh=lh,
        surf_rh=rh,
        **kwargs,
    )
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_MySurf_VertexWise_OneHemi(
    single_data,
    surf,
    title=None,
    hemi="left",
    size=(500, 200),
    **kwargs,
):
    array_single_hemi = np.array(single_data)
    p = plot_base(
        left_data=array_single_hemi if hemi == "left" else None,
        right_data=array_single_hemi if hemi == "right" else None,
        surf_lh=surf if hemi == "left" else None,
        surf_rh=surf if hemi == "right" else None,
        hemi=hemi,
        size=size,
        **kwargs,
    )
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
    data_lh, data_rh = remove_medial_wall(data_lh, data_rh)

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
    as_outline=False,
    outline_alpha=1,
    outline_cmap="gray",
    title=None,
    **kwargs,
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
    p = plot_base(
        left_data=map_array_L,
        right_data=map_array_R,
        surf_lh=lh,
        surf_rh=rh,
        **kwargs,
    )
    if as_outline:
        p.add_layer(
            {"left": lh_parc, "right": rh_parc},
            as_outline=True,
            cbar=False,
            cmap=outline_cmap,
            alpha=outline_alpha,
        )
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_MySurf_RegionWise_OneHemi(
    array_single_hemi,
    parc,
    surf,
    as_outline=False,
    outline_alpha=1,
    outline_cmap="gray",
    title=None,
    hemi="left",
    size=(500, 200),
    **kwargs,
):
    array_single_hemi = np.array(array_single_hemi)
    map_array = map_array_to_label(array_single_hemi, parc)
    p = plot_base(
        left_data=map_array if hemi == "left" else None,
        right_data=map_array if hemi == "right" else None,
        surf_lh=surf if hemi == "left" else None,
        surf_rh=surf if hemi == "right" else None,
        hemi=hemi,
        size=size,
        **kwargs,
    )
    if as_outline:
        p.add_layer(
            parc, as_outline=True, cbar=False, cmap=outline_cmap, alpha=outline_alpha
        )
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


def Plot_Each_Region_Num(
    region_num,
    parc_hemi,
    surf_hemi,
    as_outline=True,
    outline_alpha=1,
    outline_cmap="gray",
    cmap="tab20",
    size=(500, 200),
    hemi="left",
    title=None,
    **kwargs,
):
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
    p = plot_base(
        left_data=regions if hemi == "left" or hemi == "both" else None,
        right_data=regions if hemi == "right" or hemi == "both" else None,
        surf_lh=surf_hemi if hemi == "left" or hemi == "both" else None,
        surf_rh=surf_hemi if hemi == "right" or hemi == "both" else None,
        cmap=cmap,
        cbar=False,
        size=size,
        hemi=hemi,
        **kwargs,
    )
    if as_outline:
        p.add_layer(
            regions, cmap=outline_cmap, as_outline=True, cbar=False, alpha=outline_alpha
        )
    figure = p.build()
    if title is not None:
        figure.axes[0].set_title(title)
    return figure


# xvfb warper
xvfb_process = None # global variable to hold the Xvfb process
def start_xvfb():
    global xvfb_process
    # 设置 DISPLAY 环境变量
    os.environ["DISPLAY"] = ":99.0"
    # 启动 xvfb
    xvfb_process = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def stop_xvfb():
    global xvfb_process
    if xvfb_process is not None:
        xvfb_process.terminate()  # 终止子进程
        xvfb_process.wait()       # 等待子进程完全退出
        xvfb_process = None
