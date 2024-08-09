"""
Author: [Yichun Huang]
Date: [26/07/2024]

This module contains functions and classes for performing correlation analysis between gene expression data and imaging data.

Functions:
- rank_array(array): Ranks the elements in an array.
- spearman_opt(imaging, genes): Calculates the Spearman rank correlation between two arrays.
- compute_spearman_pval(corr, boot_corr): Computes the p-values and their FDR correction for the correlation.

Classes:
- TransImgCorr: Performs correlation analysis between gene expression data and imaging data.

"""

import numba
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from ..Stats.spin_test_utils import generate_spin_permutation
from ..utils._spearman import spearman_opt


def compute_spearman_pval(corr, boot_corr):
    """
    Compute the p-values, and its fdr correction, of the correlation,
    from the list of bootstrapped correlations.

    Args:
        corr: Array of correlation coefficients.
        boot_corr: Array of bootstrapped correlations.

    Returns:
        pval: Array of p-values.
        pval_fdr: Array of FDR-corrected p-values.

    """
    # This calculation assumes that the order of the genes is the same
    # in both the original and the bootstrapped list. IF one is ordered,
    # make sure the order of the other is the same.
    n_genes = boot_corr.shape[0]
    n_perm = boot_corr.shape[1]
    pval = np.zeros(n_genes)
    pval_fdr = np.zeros(n_genes)
    for i in range(n_genes):
        if corr[i] >= 0:
            pval[i] = np.sum(boot_corr[i, :] > corr[i]) / n_perm
        else:
            pval[i] = np.sum(boot_corr[i, :] < corr[i]) / n_perm
    # FDR correction
    _, pval_fdr, _, _ = multipletests(pval, method="fdr_bh", is_sorted=False)
    return pval, pval_fdr


class TransImgCorr:
    """
    Performs correlation analysis between gene expression data and imaging data.

    Args:
        gene_expression: Array representing gene expression data.
        imaging_data: Array representing imaging data.
        gene_label: Array of gene labels.
        spin_test_method: Method for spin test (optional).
        parcellationLR: Parcellation file (optional).
        atlas: Atlas type (default: "fsLR").
        density: Density (default: "32k").
        perm: Number of permutations (default: 1000).
        seed: Random seed (default: 1234).
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        gene_expression,
        imaging_data,
        gene_label,
        spin_test_method=None,
        parcellationLR=None,
        atlas="fsLR",
        density="32k",
        perm=1000,
        seed=1234,
        **kwargs
    ):
        self.Y = imaging_data.reshape(-1, 1)
        self.Y = zscore(self.Y, axis=0, ddof=1)
        self.X = zscore(gene_expression, axis=0, ddof=1)
        self.n_regions = self.X.shape[0]
        self.gene_labels = np.array(gene_label)

        self.perm = perm
        self.seed = seed

        self.corr = spearman_opt(np.squeeze(self.Y), self.X)

        # initialize the spin permutation of zscored imaging data
        if spin_test_method is not None:
            assert parcellationLR is not None, "Please provide the parcellation file"
            self.use_spin_test = True
            self.permutation_imaging = generate_spin_permutation(
                self.Y,
                parcellationLR,
                atlas=atlas,
                density=density,
                perm=self.perm,
                method=spin_test_method,
                seed=self.seed,
                **kwargs
            )
        else:
            self.use_spin_test = False

    def permutative_P_statistic(self):
        """
        Perform bootstrapping on the correlation.

        The function first calculates the correlation between the imaging
        vector and each of the genes. Then, it performs 1000 bootstrapping
        iterations of the same correaltion only using the permuted imaging
        data.

        Returns:
            p_val: Array of p-values.
            p_val_fdr: Array of FDR-corrected p-values.

        """
        boot_corr = np.zeros((self.X.shape[1], self.perm))

        if self.use_spin_test:
            for i in tqdm(range(self.perm)):
                boot_corr[:, i] = spearman_opt(
                    np.squeeze(self.permutation_imaging[:, i]), self.X
                )
        else:
            rng = np.random.default_rng(seed=self.seed)
            for i in tqdm(range(self.perm)):
                order = rng.permutation(self.n_regions)  # TODO: spin test
                boot_corr[:, i] = spearman_opt(np.squeeze(self.Y[order, :]), self.X)

        p_val, p_val_fdr = compute_spearman_pval(self.corr, boot_corr)
        return p_val, p_val_fdr

    def run(self):
        """
        Run the correlation analysis and return the results.

        Returns:
            df_corr: DataFrame containing gene labels, correlation coefficients, p-values, and FDR-corrected p-values.

        """
        p_val, p_val_fdr = self.permutative_P_statistic()
        # Create a dataframe with the results
        df_corr = pd.DataFrame(
            {
                "gene": self.gene_labels,
                "corr": self.corr,
                "p_val": p_val,
                "p_val_fdr": p_val_fdr,
            }
        )
        df_corr_positve = df_corr[df_corr["corr"] > 0]
        df_corr_negative = df_corr[df_corr["corr"] < 0]
        df_corr_positve = df_corr_positve.sort_values(by="p_val_fdr")
        df_corr_negative = df_corr_negative.sort_values(by="p_val_fdr")
        return df_corr_positve, df_corr_negative
