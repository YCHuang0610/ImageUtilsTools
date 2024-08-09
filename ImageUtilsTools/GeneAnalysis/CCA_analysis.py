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
    if r < 0:  # 匹配方向
        X_c = -X_c
        r = -r
        X_weight = -X_weight
    return X_c, r, X_weight


class CCA_permutation:
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
        self.X_c, self.r, self.weights = CCA_function_align_weights(self.X, self.y)
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
        return self.X_c

    @property
    def x_weight(self):
        return self.weights

    def permutative_P_statistic(self):
        r_perm_list = np.zeros(self.perm)
        if self.use_spin_test:
            for i in tqdm(range(self.perm)):
                y_permutation = self.y_perm[:, i]
                _, r_perm, _ = CCA_function_align_weights(self.X, y_permutation)
                r_perm_list[i] = r_perm
        else:
            rng = np.random.default_rng(self.seed)
            for i in tqdm(range(self.perm)):
                y_permutation = rng.permutation(self.y)
                _, r_perm, _ = CCA_function_align_weights(self.X, y_permutation)
                r_perm_list[i] = r_perm

        p = (r_perm_list > self.r).sum() / self.perm

        print(f"p-value of CCA: {p}")
        return p

    def compute_bootstrap(self):
        rng = np.random.default_rng(self.seed)
        weight_perm = np.zeros((self.perm, self.n_features))
        for i in tqdm(range(self.perm)):
            bootindex = rng.choice(
                np.arange(0, self.n_regions), size=self.n_regions, replace=True
            )
            X_boot = self.X[bootindex, :]
            y_boot = self.y[bootindex]
            _, _, x_weight = CCA_function_align_weights(X_boot, y_boot)
            weight_perm[i, :] = x_weight.flatten()

        std = np.std(weight_perm, axis=0)

        temp_w = self.weights.flatten() / std

        Z = np.sort(temp_w)[::-1]
        ind = np.argsort(temp_w)[::-1]
        gene_list = self.x_label[ind]

        df = pd.DataFrame({"Gene": gene_list, "Z-score": Z})
        return df
