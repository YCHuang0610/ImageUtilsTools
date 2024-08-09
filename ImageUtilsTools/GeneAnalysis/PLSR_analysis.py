"""
PLSR_analysis.py

Author: Yichun Huang
Date: 26/07/2024

This file contains functions and a class for performing Partial Least Squares Regression (PLSR) analysis on gene expression and imaging data.

# pip install -e git+https://github.com/netneurolab/pypyls.git/#egg=pyls

Reference:
- https://github.com/alegiac95/Imaging-transcriptomics
- https://github.com/netneurolab/pypyls
- [Whitaker and Vértes, PNAS 2016](http://www.pnas.org/content/113/32/9105)
- https://github.com/SarahMorgan/Morphometric_Similarity_SZ/blob/master/Gene_analyses.md

Functions:
- correlation(c1, c2): Calculate the correlation between two sets of data.

Class:
- TransImgPLS: Performs PLSR analysis on gene expression and imaging data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuromaps import stats
from pyls import pls_regression
from tqdm import tqdm
from scipy.stats import zscore, pearsonr, spearmanr
from ..Stats.spin_test_utils import generate_spin_permutation


def correlation(c1, c2):
    """
    Calculate the correlation between two sets of data.

    Parameters:
    c1 (numpy.ndarray): Matrix with shape (n, 2).
    c2 (numpy.ndarray): Vector with shape (n,).

    Returns:
    numpy.ndarray: Correlation coefficients between c1 and c2.

    The function calculates the correlation coefficients between the columns of c1 and c2.
    It first horizontally stacks c1 and c2, then computes the correlation matrix.
    Finally, it returns the correlation coefficients between the columns of c1 and c2.
    """
    return np.corrcoef(np.hstack((c1, c2)), rowvar=False)[:2, -1]


class TransImgPLS:
    """
    TransImgPLS class performs Partial Least Squares Regression (PLSR) analysis on gene expression and imaging data.

    Args:
        gene_expression (ndarray): The gene expression data.
        imaging_data (ndarray): The imaging data.
        gene_label (list): The labels for the genes.
        n_components (int, optional): The number of components to extract. Defaults to 2.
        spin_test_method (str, optional): The spin test method. Defaults to None.
        parcellationLR (str, optional): The parcellationLR parameter. Required if spin_test_method is not None. Defaults to None.
        perm (int, optional): The number of permutations for spin test or bootstrap. Defaults to 1000.
        atlas (str, optional): The atlas parameter. Defaults to "fsLR".
        density (str, optional): The density parameter. Defaults to "32k".
        seed (int, optional): The random seed. Defaults to 1234.
        **kwargs: Additional keyword arguments.

    Attributes:
        perm (int): The number of permutations for spin test or bootstrap.
        dim (int): The number of components to extract.
        seed (int): The random seed.
        Y (ndarray): The z-scored imaging data.
        X (ndarray): The z-scored gene expression data.
        n_regions (int): The number of regions in the gene expression data.
        gene_labels (list): The labels for the genes.
        XS (ndarray): The scores of the extracted components.
        stats_W (ndarray): The weights of the extracted components.
        x1 (ndarray): The indices of the sorted weights for the first component.
        PLS1w (ndarray): The sorted weights for the first component.
        PLS1_gene_labels (ndarray): The gene labels corresponding to the sorted weights for the first component.
        x2 (ndarray): The indices of the sorted weights for the second component.
        PLS2w (ndarray): The sorted weights for the second component.
        PLS2_gene_labels (ndarray): The gene labels corresponding to the sorted weights for the second component.
        use_spin_test (bool): Indicates whether spin test is used.
        permutation_imaging (ndarray): The spin permutation of z-scored imaging data.

    Methods:
        plot_xscore_correlation: Plots the correlation between imaging data and PLS component.
        plot_expVar: Plots the percentage of variance explained by the components.
        permutative_P_statistic: Performs permutation testing to assess the significance of PLS result.
        compute_bootstrap: Computes bootstrap weights and standard errors.
        align_sign_with_Y: Aligns the sign of the scores and weights with the imaging data.

    """

    def __init__(
        self,
        gene_expression,
        imaging_data,
        gene_label,
        n_components=2,
        spin_test_method=None,
        parcellationLR=None,
        perm=1000,
        atlas="fsLR",
        density="32k",
        seed=1234,
        **kwargs
    ):
        # set up regression parameters
        self.perm = perm
        self.dim = n_components
        self.seed = seed

        # initialize input data and zscore imaging data and gene expression data
        self.Y = imaging_data.reshape(-1, 1)
        self.Y = zscore(self.Y, axis=0, ddof=1)
        self.X = zscore(gene_expression, axis=0, ddof=1)
        self.n_regions = self.X.shape[0]
        self.gene_labels = gene_label

        # run the first PLS regression
        res = pls_regression(
            self.X, self.Y, n_components=self.dim, n_perm=0, n_boot=0, seed=self.seed
        )
        self.XS = res["x_scores"]
        self.stats_W = res["x_weights"]
        self.XS, self.stats_W = self.align_sign_with_Y(self.XS, self.Y, self.stats_W)

        # sort the weights and gene labels
        self.x1 = np.argsort(-self.stats_W[:, 0])
        self.PLS1w = self.stats_W[self.x1, 0]
        self.PLS1_gene_labels = np.array(self.gene_labels[self.x1])
        self.x2 = np.argsort(-self.stats_W[:, 1])
        self.PLS2w = self.stats_W[self.x2, 1]
        self.PLS2_gene_labels = np.array(self.gene_labels[self.x2])

        # initialize spin permutation of zscored imaging data
        if spin_test_method is not None:
            assert parcellationLR is not None, "parcellationLR is required"
            self.use_spin_test = True
            self.permutation_imaging = generate_spin_permutation(
                self.Y,
                parcellationLR,
                atlas=atlas,
                density=density,
                perm=perm,
                method=spin_test_method,
                seed=self.seed,
                **kwargs
            )
        else:
            self.use_spin_test = False

    def plot_xscore_correlation(
        self, plot=False, PC=1, metric="spearmanr", savefig=None
    ):
        """
        Plots the correlation between imaging data and PLS component.

        Args:
            plot (bool, optional): Whether to plot the correlation. Defaults to False.
            PC (int, optional): The PLS component to plot. Defaults to 1.
            metric (str, optional): The correlation metric to use. Defaults to 'spearmanr'.
            savefig (str, optional): The filename to save the plot. Defaults to None.

        Returns:
            ndarray: The scores of the specified PLS component.

        Raises:
            AssertionError: If PC is not 1 or 2.
            AssertionError: If metric is not 'spearmanr' or 'pearsonr'.
        """
        # 计算相关系数
        assert PC in [1, 2], "PC must be 1 or 2"
        assert metric in [
            "spearmanr",
            "pearsonr",
        ], "metric must be spearmanr or pearsonr"
        if self.use_spin_test:
            r, p = stats.compare_images(
                np.squeeze(self.Y),
                self.XS[:, PC - 1],
                nulls=self.permutation_imaging,
                metric=metric,
            )
        else:
            if metric == "spearmanr":
                r, p = spearmanr(np.squeeze(self.Y), self.XS[:, PC - 1])
            elif metric == "pearsonr":
                r, p = pearsonr(np.squeeze(self.Y), self.XS[:, PC - 1])
        # 绘图
        if plot:
            plt.figure()
            plt.scatter(np.squeeze(self.Y), self.XS[:, PC - 1])
            plt.xlabel("Imaging data")
            plt.ylabel("PLS component {}".format(PC))
            plt.text(
                0.1,
                0.9,
                "r={:.2f}, p={:.2f}".format(r, p),
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            if savefig is not None:
                plt.savefig(savefig)
        print(
            "correlation between imaging data and PLS component {}: r={:.4f}, p={:.4f}".format(
                PC, r, p
            )
        )
        return self.XS[:, PC - 1]

    def plot_expVar(self, dim=15, plot=False, savefig=None):
        """
        Plots the percentage of variance explained by the components.

        Args:
            dim (int, optional): The number of components to include. Defaults to 15.
            plot (bool, optional): Whether to plot the percentage of variance explained. Defaults to False.
            savefig (str, optional): The filename to save the plot. Defaults to None.

        Returns:
            ndarray: The percentage of variance explained by each component.
        """
        res = pls_regression(
            self.X, self.Y, n_components=dim, n_perm=0, n_boot=0, seed=self.seed
        )
        pct_var = np.cumsum(res["varexp"] * 100, axis=0)
        # 绘图
        if plot:
            plt.figure()
            plt.plot(range(1, 16), pct_var)
            plt.xlabel("Number of components")
            plt.ylabel("Percentage of variance explained")
            if savefig is not None:
                plt.savefig(savefig)
        return pct_var

    def permutative_P_statistic(self, dim=8, plot=False, savefig=None):
        """
        Performs permutation testing to assess the significance of PLS result.

        Args:
            dim (int, optional): The number of components to include. Defaults to 8.
            plot (bool, optional): Whether to plot the p-values. Defaults to False.
            savefig (str, optional): The filename to save the plot. Defaults to None.

        Returns:
            ndarray: The p-values for each number of components.

        Raises:
            AssertionError: If parcellationLR is None and spin_test_method is not None.
        """
        # permutation testing to assess significance of PLS result as a function of
        # the number of components (dim) included:
        if self.use_spin_test:
            p = np.zeros(dim)
            for i in range(dim):
                res = pls_regression(
                    self.X,
                    self.Y,
                    n_components=i + 1,
                    n_perm=0,
                    n_boot=0,
                    seed=self.seed,
                )
                pct_var = np.cumsum(res["varexp"], axis=0)
                Rsquared = pct_var[i]
                Rsq = np.zeros(self.perm)
                # permutation test
                for j in tqdm(range(self.perm)):
                    Y_permutation = self.permutation_imaging[:, j].reshape(-1, 1)
                    res_perm = pls_regression(
                        self.X,
                        Y_permutation,
                        n_components=i + 1,
                        n_perm=0,
                        n_boot=0,
                        seed=self.seed,
                    )
                    pct_var = np.cumsum(res_perm["varexp"], axis=0)
                    Rsq[j] = pct_var[i]
                p[i] = np.sum(Rsq >= Rsquared) / self.perm
                print("p-value for {} components: {:.4f}".format(i + 1, p[i]))
        else:
            p = np.zeros(dim)
            for i in range(dim):
                res = pls_regression(
                    self.X,
                    self.Y,
                    n_components=i + 1,
                    n_perm=0,
                    n_boot=0,
                    seed=self.seed,
                )
                pct_var = np.cumsum(res["varexp"], axis=0)
                Rsquared = pct_var[i]
                Rsq = np.zeros(self.perm)
                # permutation test
                rng = np.random.default_rng(seed=self.seed)
                for j in tqdm(range(self.perm)):
                    order = rng.permutation(self.n_regions)
                    Y_permutation = self.Y[order, :]
                    res_perm = pls_regression(
                        self.X,
                        Y_permutation,
                        n_components=i + 1,
                        n_perm=0,
                        n_boot=0,
                        seed=self.seed,
                    )
                    pct_var = np.cumsum(res_perm["varexp"], axis=0)
                    Rsq[j] = pct_var[i]
                p[i] = np.sum(Rsq >= Rsquared) / self.perm
                print("p-value for {} components: {:.4f}".format(i + 1, p[i]))
        # 绘图
        if plot:
            plt.figure()
            plt.plot(range(1, dim + 1), p)
            plt.xlabel("Number of components")
            plt.ylabel("P-value")
            if savefig is not None:
                plt.savefig(savefig)
        return p

    def compute_bootstrap(self):
        """
        Computes bootstrap weights and standard errors.

        Returns:
            DataFrame: The bootstrap weights for the first and second components.
        """
        PLS1weight_perm = []
        PLS2weight_perm = []
        rng = np.random.default_rng(seed=self.seed)
        for _ in tqdm(range(self.perm)):
            bootindex = rng.choice(
                np.arange(0, self.n_regions), self.n_regions, replace=True
            )
            res = pls_regression(
                self.X[bootindex, :],
                self.Y[bootindex, :],
                n_components=self.dim,
                n_perm=0,
                n_boot=0,
                seed=self.seed,
            )
            # align sign with Y
            # PC1
            temp = res["x_weights"][:, 0]
            newW = temp[self.x1]
            if np.corrcoef(newW, self.PLS1w)[0, 1] < 0:
                newW = -newW
            PLS1weight_perm.append(newW)
            # PC2
            temp = res["x_weights"][:, 1]
            newW = temp[self.x2]
            if np.corrcoef(newW, self.PLS2w)[0, 1] < 0:
                newW = -newW
            PLS2weight_perm.append(newW)

        PLS1weight_perm = np.array(PLS1weight_perm)
        PLS2weight_perm = np.array(PLS2weight_perm)

        assert PLS1weight_perm.shape == (
            self.perm,
            len(self.gene_labels),
        ), "PLS1weight_perm shape is wrong"
        assert PLS2weight_perm.shape == (
            self.perm,
            len(self.gene_labels),
        ), "PLS2weight_perm shape is wrong"

        # compute standard error
        self.PLS1_weight_SE = np.std(PLS1weight_perm, axis=0, ddof=1)
        self.PLS2_weight_SE = np.std(PLS2weight_perm, axis=0, ddof=1)
        # compute z-score
        temp_w1 = self.PLS1w / self.PLS1_weight_SE
        temp_w2 = self.PLS2w / self.PLS2_weight_SE
        # order bootstrap weights (Z) and the names of regions
        # PC1
        Z1 = np.sort(temp_w1)[::-1]
        ind1 = np.argsort(temp_w1)[::-1]
        PLS1_gene_list = self.PLS1_gene_labels[ind1]
        # PC2
        Z2 = np.sort(temp_w2)[::-1]
        ind2 = np.argsort(temp_w2)[::-1]
        PLS2_gene_list = self.PLS2_gene_labels[ind2]

        # write to dataframe
        df_PLS1 = pd.DataFrame({"Gene": PLS1_gene_list, "Z-score": Z1})
        df_PLS2 = pd.DataFrame({"Gene": PLS2_gene_list, "Z-score": Z2})
        return df_PLS1, df_PLS2

    @staticmethod
    def align_sign_with_Y(XS, Y, stats_W):
        """
        Aligns the sign of the scores and weights with the imaging data.

        Args:
            XS (ndarray): The scores.
            Y (ndarray): The imaging data.
            stats_W (ndarray): The weights.

        Returns:
            tuple: The aligned scores and weights.
        """
        # 由于PLSweight的符号没有意义，所以需要根据相关性来调整正负
        R1 = correlation(XS, Y)
        if R1[0] < 0:
            XS[:, 0] *= -1
            stats_W[:, 0] *= -1
        if R1[1] < 0:
            XS[:, 1] *= -1
            stats_W[:, 1] *= -1

        return XS, stats_W
