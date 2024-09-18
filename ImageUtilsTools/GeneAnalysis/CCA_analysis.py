"""
Author: [Yichun Huang]
Date: [09/08/2024]

This module contains functions and classes for performing Canonical Correlation Analysis (CCA) and permutation analysis.

Functions:
    CCA_function_align_weights: Aligns the independent variable data and calculates the canonical correlation coefficient and weights.
    
Classes:
    CCA_permutation: Performs CCA permutation analysis and computes p-values and Z-scores of the weights.

"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from ..Stats.spin_test_utils import generate_spin_permutation


def CCA_function_align_weights(X, y):
    # 因为y是一维的，所以这里的n_components=1，并且y_c等于y
    cca = CCA(n_components=1)
    cca.fit(X, y)
    X_c, y_c = cca.transform(X, y)
    r = np.corrcoef(X_c.T, y.T)[0, 1]
    X_weight = cca.x_weights_
    negative = False
    if r < 0:  # 匹配方向
        negative = True
        X_c = -X_c
        r = -r
        X_weight = -X_weight
    return X_c, r, X_weight, negative


class CCA_permutation:
    """
    Class for performing CCA (Canonical Correlation Analysis) permutation analysis.

    Args:
        X (array-like): The independent variable data matrix.
        y (array-like): The dependent variable data matrix.
        x_label (array-like): Labels for the independent variables.
        spin_test_method (str, optional): The spin test method to use for permutation. Defaults to None.
        parcellationLR (array-like, optional): The parcellation data for spin permutation. Required if `spin_test_method` is not None. Defaults to None.
        atlas (str, optional): The atlas used for spin permutation. Defaults to "fsLR".
        density (str, optional): The density used for spin permutation. Defaults to "32k".
        perm (int, optional): The number of permutations to perform. Defaults to 1000.
        seed (int, optional): The random seed for reproducibility. Defaults to 1234.
        **kwargs: Additional keyword arguments to be passed to the spin permutation function.

    Attributes:
        X (array-like): The z-scored independent variable data matrix.
        y (array-like): The z-scored dependent variable data matrix.
        x_label (array-like): Labels for the independent variables.
        n_regions (int): The number of regions in the independent variable data.
        n_features (int): The number of features in the independent variable data.
        perm (int): The number of permutations to perform.
        seed (int): The random seed for reproducibility.
        X_c (array-like): The aligned independent variable data after CCA.
        r (float): The canonical correlation coefficient.
        weights (array-like): The weights of the independent variables in the CCA.
        negative (bool): Flag indicating if the correlation is negative.
        use_spin_test (bool): Flag indicating if spin permutation is used.
        y_perm (array-like): The permuted dependent variable data for spin permutation.

    """

    def __init__(
        self,
        X,
        y,
        x_label,
        spin_test_method=None,
        parcellationLR=None,
        atlas="fsLR",
        density="32k",
        perm=1000,
        seed=1234,
        **kwargs,
    ):
        self.X = zscore(X, axis=0, ddof=1)
        self.y = zscore(y, ddof=1)
        self.x_label = np.array(x_label)
        self.n_regions = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.perm = perm
        self.seed = seed

        # run the first CCA
        self.X_c, self.r, self.weights, self.negative = CCA_function_align_weights(
            self.X, self.y
        )
        if self.negative:
            print("Negative correlation,")
            print(f"Canonical Correlation Coefficient: {-self.r}")
        print(f"Canonical Correlation Coefficient: {self.r}")

        # initialize spin permutation of zscored imaging data (Y)
        if spin_test_method is not None:
            assert parcellationLR is not None, "parcellationLR is required"
            self.use_spin_test = True
            self.y_perm = generate_spin_permutation(
                self.y,
                parcellationLR,
                atlas=atlas,
                density=density,
                perm=perm,
                method=spin_test_method,
                seed=self.seed,
                **kwargs,
            )
        else:
            self.use_spin_test = False

    @property
    def x_score(self):
        """
        Get the aligned independent variable data after CCA.
        The X_c is positive aligned with the dependent variable.
        So, if the correlation is negative, the X_c will be negative aligned with the dependent variable.

        Returns:
            array-like: The aligned independent variable data after CCA.

        """
        if self.negative:
            return -self.X_c
        return self.X_c

    @property
    def x_weight(self):
        """
        Get the weights of the independent variables in the CCA.

        Returns:
            array-like: The weights of the independent variables in the CCA.

        """
        return self.weights

    def permutative_P_statistic(self):
        """
        Perform permutation analysis and compute the p-value of CCA.

        Returns:
            float: The p-value of CCA.

        """
        r_perm_list = np.zeros(self.perm)
        if self.use_spin_test:
            for i in tqdm(range(self.perm)):
                y_permutation = self.y_perm[:, i]
                _, r_perm, _, _ = CCA_function_align_weights(self.X, y_permutation)
                r_perm_list[i] = r_perm
        else:
            rng = np.random.default_rng(self.seed)
            for i in tqdm(range(self.perm)):
                y_permutation = rng.permutation(self.y)
                _, r_perm, _, _ = CCA_function_align_weights(self.X, y_permutation)
                r_perm_list[i] = r_perm

        p = (r_perm_list > self.r).sum() / self.perm

        print(f"p-value of CCA: {p}")
        return p

    def compute_bootstrap(self):
        """
        Perform bootstrap analysis and compute the Z-scores of the weights.

        Returns:
            DataFrame: A DataFrame containing the gene list and corresponding Z-scores.

        """
        rng = np.random.default_rng(self.seed)
        weight_perm = np.zeros((self.perm, self.n_features))
        for i in tqdm(range(self.perm)):
            bootindex = rng.choice(
                np.arange(0, self.n_regions), size=self.n_regions, replace=True
            )
            X_boot = self.X[bootindex, :]
            y_boot = self.y[bootindex]
            _, _, x_weight, _ = CCA_function_align_weights(X_boot, y_boot)
            weight_perm[i, :] = x_weight.flatten()

        std = np.std(weight_perm, axis=0)

        temp_w = self.weights.flatten() / std

        Z = np.sort(temp_w)[::-1]
        ind = np.argsort(temp_w)[::-1]
        gene_list = self.x_label[ind]

        df = pd.DataFrame({"Gene": gene_list, "Z-score": Z})
        return df
