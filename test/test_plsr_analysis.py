import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def _load_plsr_module():
    """Load PLSR_analysis without importing optional neuroimaging dependencies."""
    package_names = [
        "ImageUtilsTools",
        "ImageUtilsTools.GeneAnalysis",
        "ImageUtilsTools.Stats",
        "ImageUtilsTools.utils",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    neuromaps = types.ModuleType("neuromaps")
    neuromaps.stats = types.SimpleNamespace()
    sys.modules["neuromaps"] = neuromaps

    pyls = types.ModuleType("pyls")
    pyls.pls_regression = lambda *args, **kwargs: None
    sys.modules["pyls"] = pyls

    spin_utils = types.ModuleType("ImageUtilsTools.Stats.spin_test_utils")
    spin_utils.generate_spin_permutation = lambda *args, **kwargs: None
    sys.modules[spin_utils.__name__] = spin_utils

    nan_utils = types.ModuleType("ImageUtilsTools.utils.nan_utils")
    nan_utils.NanHelper = object
    sys.modules[nan_utils.__name__] = nan_utils

    module_path = (
        Path(__file__).parents[1]
        / "ImageUtilsTools"
        / "GeneAnalysis"
        / "PLSR_analysis.py"
    )
    module_name = "ImageUtilsTools.GeneAnalysis.PLSR_analysis_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.tqdm = lambda iterable: iterable
    return module


class TestPLSRStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_plsr_module()

    def test_bootstrap_uses_component_specific_two_sided_p_values(self):
        model = self.module.TransImgPLS.__new__(self.module.TransImgPLS)
        model.perm = 3
        model.seed = 7
        model.n_regions = 4
        model.dim = 2
        model.X = np.zeros((4, 3))
        model.Y = np.zeros((4, 1))
        model.x1 = np.arange(3)
        model.x2 = np.arange(3)
        model.PLS1w = np.array([3.0, 2.0, 1.0])
        model.PLS2w = np.array([2.0, 0.5, -1.0])
        model.gene_labels = np.array(["A", "B", "C"])
        model.PLS1_gene_labels = model.gene_labels.copy()
        model.PLS2_gene_labels = model.gene_labels.copy()

        bootstrap_weights = iter(
            [
                np.array([[2.8, 1.7], [2.1, 0.4], [0.9, -0.8]]),
                np.array([[3.2, 2.4], [1.8, 0.7], [1.1, -1.4]]),
                np.array([[3.0, 1.9], [2.2, 0.3], [0.8, -0.9]]),
            ]
        )

        def fake_pls_regression(*args, **kwargs):
            return {"x_weights": next(bootstrap_weights)}

        self.module.pls_regression = fake_pls_regression
        pls1, pls2 = model.compute_bootstrap()

        for table in (pls1, pls2):
            expected_p = 2 * norm.sf(np.abs(table["Z-score"].to_numpy()))
            expected_fdr = multipletests(expected_p, method="fdr_bh")[1]
            np.testing.assert_allclose(table["P-value"], expected_p)
            np.testing.assert_allclose(table["P-adjusted"], expected_fdr)

        self.assertFalse(np.allclose(pls1["P-value"], pls2["P-value"]))

    def test_permutation_p_value_has_plus_one_correction(self):
        for use_spin_test in (False, True):
            with self.subTest(use_spin_test=use_spin_test):
                model = self.module.TransImgPLS.__new__(self.module.TransImgPLS)
                model.perm = 3
                model.seed = 7
                model.n_regions = 4
                model.X = np.zeros((4, 2))
                model.Y = np.arange(4.0).reshape(-1, 1)
                model.use_spin_test = use_spin_test
                if use_spin_test:
                    model.permutation_imaging = np.zeros((4, model.perm))

                calls = iter([1.0, 0.0, 0.0, 0.0])

                def fake_pls_regression(*args, **kwargs):
                    return {"varexp": np.array([next(calls)])}

                self.module.pls_regression = fake_pls_regression
                p_value = model.permutative_P_statistic(dim=1)
                np.testing.assert_allclose(p_value, [1 / (model.perm + 1)])


if __name__ == "__main__":
    unittest.main()
