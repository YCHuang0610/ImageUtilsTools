import pandas as pd
import numpy as np
from nilearn.image import threshold_img
from nilearn.input_data import NiftiMasker
from nilearn.glm.second_level import SecondLevelModel
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def run_two_sample_T_test(Group1_imgs, Group2_imgs):
    # 输入类型检查
    if not isinstance(Group1_imgs, list) or not isinstance(Group2_imgs, list):
        raise ValueError("Group1_imgs 和 Group2_imgs 必须是列表类型。")

    # 输入长度检查
    if len(Group1_imgs) == 0 or len(Group2_imgs) == 0:
        raise ValueError("Group1_imgs 和 Group2_imgs 不能是空列表。")

    try:
        n_Group1 = len(Group1_imgs)
        n_Group2 = len(Group2_imgs)
        condition_effect = np.hstack(
            ([1] * n_Group1, [0] * n_Group2)
        )  # Group1 vs Group2
        unpaired_design_matrix = pd.DataFrame(
            {
                "Group1 vs Group2": condition_effect,
                "intercept": 1,
            }
        )
        second_level_model = SecondLevelModel(n_jobs=2).fit(
            Group1_imgs + Group2_imgs, design_matrix=unpaired_design_matrix
        )
        t_map = second_level_model.compute_contrast(
            "Group1 vs Group2", output_type="stat"
        )
        return t_map
    except Exception as e:
        # 捕获并处理可能的异常
        raise RuntimeError(f"在执行T检验时发生错误: {e}")


def fwer_threshold(t_map, df, alpha=0.05, masker_strategy="background", two_tail=True):
    """
    Calculate the family-wise error rate (FWER) threshold for a given t-map.

    Parameters:
    - t_map: The t-map as a Nifti image or array.
    - df: Degrees of freedom for the t-distribution.
    - alpha: Significance level (default is 0.05).
    - two_tail: Whether to use a two-tailed test (default is True).

    Returns:
    - fwe_p_value: The family-wise error p-value.
    - t_threshold: The t-value threshold.

    """
    try:
        # 输入验证
        if not isinstance(alpha, (float, int)) or not (0 < alpha < 1):
            raise ValueError("alpha 必须是0到1之间的数字。")
        if not isinstance(df, int) or df <= 0:
            raise ValueError("df 必须是正整数。")
        if masker_strategy not in [
            "background",
            "whole-brain-template",
            "gm-template",
            "wm-template",
            "csf-template",
        ]:
            raise ValueError(
                "masker_strategy 必须是 'background' 或 'whole-brain-template'等。"
            )
        if not isinstance(two_tail, bool):
            raise ValueError("two_tail 必须是布尔值。")

        # 计算过程
        masker = NiftiMasker(mask_strategy=masker_strategy).fit(t_map)
        stats = np.ravel(masker.transform(t_map))
        n_voxels = np.size(stats)
        if two_tail:
            alpha = alpha / 2
        fwe_p_value = alpha / n_voxels
        t_threshold = t.ppf(1 - fwe_p_value, df)
        return fwe_p_value, t_threshold
    except Exception as e:
        # 捕获并处理可能的异常
        raise RuntimeError(f"计算FWER阈值时发生错误: {e}")


def fdr_threshold(t_map, df, alpha=0.05, masker_strategy="background", two_tail=True):
    """
    Calculate the false discovery rate (FDR) threshold for a given t-map.

    Parameters:
    - t_map: The t-map as a Nifti image or array.
    - df: Degrees of freedom for the t-distribution.
    - alpha: Significance level (default is 0.05).
    - masker_strategy: Strategy for mask application, 'background' or 'whole-brain'.
    - two_tail: Whether to use a two-tailed test (default is True).

    Returns:
    - fdr_p_value: The false discovery rate p-value.
    - t_threshold: The t-value threshold.
    """
    try:
        # 输入验证
        if not isinstance(alpha, (float, int)) or not (0 < alpha < 1):
            raise ValueError("alpha 必须是0到1之间的数字。")
        if not isinstance(df, int) or df <= 0:
            raise ValueError("df 必须是正整数。")
        if masker_strategy not in ["background", "whole-brain"]:
            raise ValueError("masker_strategy 必须是 'background' 或 'whole-brain'。")
        if not isinstance(two_tail, bool):
            raise ValueError("two_tail 必须是布尔值。")

        # 计算过程
        masker = NiftiMasker(mask_strategy=masker_strategy).fit(t_map)
        stats = np.ravel(masker.transform(t_map))
        if two_tail:
            alpha = alpha / 2
            p_values = 2 * t.sf(np.abs(stats), df)  # Adjust for two-tailed test
        else:
            p_values = t.sf(np.abs(stats), df)
        reject, fdr_p_values, _, _ = multipletests(
            p_values, alpha=alpha, method="fdr_bh"
        )
        significant_stats = stats[reject]  # Use original alpha for thresholding
        if significant_stats.size > 0:
            t_threshold = np.min(np.abs(significant_stats))
        else:
            t_threshold = None  # No significant results
        return fdr_p_values, t_threshold
    except Exception as e:
        # 捕获并处理可能的异常
        raise RuntimeError(f"计算FDR阈值时发生错误: {e}")


def threshold_t_map(
    t_map, df, alpha=0.05, two_tail=True, masker_strategy="background", method="fwer"
):
    """
    Thresholds a t-map based on the family-wise error rate (FWER) or false discovery rate (FDR).

    Parameters:
    - t_map (nifti image): The t-map to be thresholded.
    - df (int): Degrees of freedom.
    - alpha (float, optional): The significance level. Default is 0.05.
    - two_tail (bool, optional): Whether to perform a two-tailed test. Default is True.
    - method (str, optional): The method to use for thresholding. "fwer" or "fdr". Default is "fwer".

    Returns:
    - thresholded_t_map (nifti image): The thresholded t-map.

    """
    try:
        # 输入验证
        if not isinstance(df, int) or df <= 0:
            raise ValueError("df 必须是正整数。")
        if not isinstance(alpha, float) or not (0 < alpha < 1):
            raise ValueError("alpha 必须是0到1之间的浮点数。")
        if not isinstance(two_tail, bool):
            raise ValueError("two_tail 必须是布尔值。")
        if method not in ["fwer", "fdr"]:
            raise ValueError("method 必须是 'fwer' 或 'fdr'。")

        # 方法选择
        if method == "fwer":
            fwe_p_value, t_threshold = fwer_threshold(
                t_map, df, alpha, masker_strategy, two_tail
            )
        elif method == "fdr":
            fdr_p_value, t_threshold = fdr_threshold(
                t_map, df, alpha, masker_strategy, two_tail
            )

        # 应用阈值
        thresholded_t_map = threshold_img(
            t_map,
            threshold=t_threshold,
            cluster_threshold=0,
            two_sided=two_tail,
            mask_img=None,
        )
        return thresholded_t_map
    except Exception as e:
        # 捕获并处理可能的异常
        raise RuntimeError(f"在阈值化t-map时发生错误: {e}")
