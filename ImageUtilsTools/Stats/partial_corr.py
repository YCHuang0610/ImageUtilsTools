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
