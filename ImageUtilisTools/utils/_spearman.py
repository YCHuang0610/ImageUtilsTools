import numba
import numpy as np


@numba.njit
def rank_array(array):
    """
    Ranks the elements in an array.

    Args:
        array: Input array.

    Returns:
        ranked: Array with elements ranked in ascending order.

    """
    _args = array.argsort()
    ranked = np.empty_like(array)
    ranked[_args] = np.arange(array.size)
    return ranked


@numba.njit
def spearman_r(x, y):
    x_rank = rank_array(x)
    y_rank = rank_array(y)
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return corr


@numba.njit(parallel=True)
def spearman_opt(imaging, genes):
    """
    Calculate the Spearman rank correlation between two arrays.

    Args:
        imaging: Array representing imaging data.
        genes: Array representing gene expression data.

    Returns:
        corr: Array of correlation coefficients.

    """
    corr = np.zeros(genes.shape[1])
    ranked_img = rank_array(imaging)
    for i in numba.prange(genes.shape[1]):
        ranked_gen = rank_array(genes[:, i])
        corr[i] = np.corrcoef(ranked_img, ranked_gen)[0, 1]
    return corr