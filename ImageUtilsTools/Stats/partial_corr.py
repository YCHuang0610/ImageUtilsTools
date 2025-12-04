import numpy as np
from scipy import stats, linalg


"""
Partial Correlation functions in Python 
reference: 
    https://gist.github.com/fabianp/9396204419c7b638d38f

The algorithm is detailed here:
    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    
Date: April 2023
Author: YCH, ychunhuang@foxmal.com
"""


def standardization(C, axis=0):
    mean = C.mean(axis=axis)
    devia = C.std(axis=axis)
    C = (C - mean) / devia
    return C


def pairPartial_corr(C, names, covas, standard=True):
    """
    (Matlab's partialcorr(X, Z) function)
    Returns the sample linear partial correlation coefficients between pairs of variables of names in C,
    controlling for the variables of covas in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    names : a list of column names in C to compute the pairs of correlation coefficient
    covas : a list of column names in C used as covariance to regressed out
    standard : whether to standardize the matrix first (subtract column means and divided standard deviation)

    Returns
    -------
    (R_corr, P_value)
    R_corr : array-like, shape (p, p)
        Correlation coefficient array
    P_value : array-like, shape (p, p)
        P values array for the correlation array
    """
    C_names = C[names]
    C_covas = C[covas]
    C_names = np.asarray(C_names)
    C_covas = np.asarray(C_covas)

    if standard:
        C_names = standardization(C_names)
        C_covas = standardization(C_covas)

    p = C_names.shape[1]
    R_corr = np.zeros((p, p), dtype=float)
    P_value = np.zeros((p, p), dtype=float)
    for i in range(p):
        R_corr[i, i] = 1
        P_value[i, i] = 0
        for j in range(i + 1, p):
            # liner regression
            beta_i = linalg.lstsq(C_covas, C_names[:, i])[0]
            beta_j = linalg.lstsq(C_covas, C_names[:, j])[0]
            # caculate residuals
            res_i = C_names[:, i] - C_covas.dot(beta_i)
            res_j = C_names[:, j] - C_covas.dot(beta_j)
            # caculate correlation between residuals
            corr, pvalue = stats.pearsonr(res_i, res_j)
            R_corr[i, j] = corr
            R_corr[j, i] = corr
            P_value[i, j] = pvalue
            P_value[j, i] = pvalue

    return R_corr, P_value

def pairPartial_corr_matrix(C, names, covas, standard=True):
    """
    (Matlab's partialcorr(X, Z) function)
    Returns the sample linear partial correlation coefficients between pairs of variables of names in C,
    controlling for the variables of covas in C.
    """
    # 使用 DataFrame 到 numpy 的零拷贝/少拷贝转换
    try:
        C_names = C[names].to_numpy(dtype=float, copy=False)
        C_covas = C[covas].to_numpy(dtype=float, copy=False)
    except AttributeError:
        # 兼容传入已是 ndarray 的情况
        C_names = np.asarray(C[names], dtype=float)
        C_covas = np.asarray(C[covas], dtype=float)

    if standard:
        C_names = standardization(C_names)
        C_covas = standardization(C_covas)

    n, p = C_names.shape
    k = C_covas.shape[1]

    # 一次性多响应最小二乘：C_covas * B = C_names
    # B 形状 (k, p)
    B, _, _, _ = linalg.lstsq(C_covas, C_names)
    # 残差矩阵 (n, p)
    resid = C_names - C_covas.dot(B)

    # 向量化相关矩阵
    # 防止全零列导致除零，先做标准差裁剪
    resid_std = resid.std(axis=0, ddof=1)
    safe_std = np.where(resid_std == 0, 1.0, resid_std)
    resid_z = (resid - resid.mean(axis=0)) / safe_std
    R_corr = np.corrcoef(resid_z, rowvar=False)
    np.fill_diagonal(R_corr, 1.0)
    R_corr = np.clip(R_corr, -1.0, 1.0)

    # p 值（部分相关的自由度 df = n - k - 2）
    df = n - k - 2
    if df <= 0:
        # 样本太少时无法给出有效 p 值
        P_value = np.full((p, p), np.nan, dtype=float)
        np.fill_diagonal(P_value, 0.0)
        return R_corr, P_value

    r = np.clip(R_corr, -0.9999999, 0.9999999)
    t_stat = r * np.sqrt(df / (1.0 - r * r))
    # 双侧检验
    P_value = 2.0 * stats.t.sf(np.abs(t_stat), df)
    np.fill_diagonal(P_value, 0.0)

    return R_corr, P_value

def ppcor(data):
    """
    A function to perform Partial Correlation just like the ppcor library in R
     (clone of R's ppcor)
    reference: https://gist.github.com/fabianp/9396204419c7b638d38f
    """
    X = -np.linalg.inv(np.cov(data.T))
    stdev = np.sqrt(np.abs(np.diag(X)))
    X /= stdev[:, None]
    X /= stdev[None, :]
    np.fill_diagonal(X, 1.0)
    return X
