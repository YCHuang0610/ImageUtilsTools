# this script was adapted from Anderson et al 2020 NatCom
# (https://github.com/HolmesLab/2020_NatComm_interneurons_cortical_function_schizophrenia/blob/master/scripts/)

import abagen
import pandas as pd
import numpy as np
from ._data_config import data
from ..utils._gii_io import load_gii

donors = [9861, 10021, 12876, 14380, 15496, 15697]


def preprocess_coords_fsLR(
    ibf_threshold=0.5,
    probe_selection="diff_stability",
    sample_norm="scaled_robust_sigmoid",
    gene_norm="scaled_robust_sigmoid",
    norm_structures="cortex",
    return_report=True,
    verbose=1,
    **kwargs,
):
    expression, coords = abagen.get_samples_in_mask(
        mask=None,
        ibf_threshold=ibf_threshold,
        probe_selection=probe_selection,
        sample_norm=sample_norm,
        gene_norm=gene_norm,
        norm_structures=norm_structures,
        return_report=return_report,
        verbose=verbose,
        **kwargs,
    )
    gene_symbols = expression.columns.tolist()
    gene_symbols = pd.DataFrame(gene_symbols)

    sample_info_vertex_mapped = pd.read_csv(data["sample_info_vertex_mapped"])

    # match the coordinates names
    coords = coords.rename(columns={"x": "mni_x", "y": "mni_y", "z": "mni_z"})
    # set the well_id as index
    sample_info_vertex_mapped = sample_info_vertex_mapped.set_index("well_id")

    # select by intersected indices
    sample_info_vertex_mapped = sample_info_vertex_mapped[
        sample_info_vertex_mapped.index.isin(coords.index)
    ]

    # replace the original mni coordinates with reannotated coordinates
    sample_info_vertex_mapped.update(coords)

    results = {
        "expression": expression,
        "gene_symbols": gene_symbols,
        "sample_info_vertex_mapped": sample_info_vertex_mapped,
    }
    return results


class AHBA_Prerocess_fsLR:
    def __init__(
        self,
        ibf_threshold=0.5,
        probe_selection="diff_stability",
        sample_norm="scaled_robust_sigmoid",
        gene_norm="scaled_robust_sigmoid",
        norm_structures="cortex",
        verbose=1,
        distance_threshold=4,
        **kwargs,
    ):
        self.ibf_threshold = ibf_threshold
        self.probe_selection = probe_selection
        self.sample_norm = sample_norm
        self.gene_norm = gene_norm
        self.norm_structures = norm_structures
        self.verbose = verbose
        self.kwargs = kwargs
        self.distance_threshold = distance_threshold

        print("Preprocessing AHBA data...")
        results = preprocess_coords_fsLR(
            ibf_threshold=self.ibf_threshold,
            probe_selection=self.probe_selection,
            sample_norm=self.sample_norm,
            gene_norm=self.gene_norm,
            norm_structures=self.norm_structures,
            return_report=False,
            verbose=self.verbose,
            **self.kwargs,
        )

        print("AHBA data preprocessed.")
        self.expression = results["expression"]
        self.gene_symbols = results["gene_symbols"]
        self.sample_info_vertex_mapped = results["sample_info_vertex_mapped"]

        right_idx = self.sample_info_vertex_mapped.index[
            self.sample_info_vertex_mapped["structure_name"].str.contains("right")
        ]
        left_idx = self.sample_info_vertex_mapped.index[
            self.sample_info_vertex_mapped["structure_name"].str.contains("left")
        ]
        self.sample_info_vertex_mapped = self.sample_info_vertex_mapped.copy()
        self.sample_info_vertex_mapped.loc[right_idx, "hemi"] = "right"
        self.sample_info_vertex_mapped.loc[left_idx, "hemi"] = "left"

        print("Expression matrix: ", self.expression.shape)
        print("Sample information: ", self.sample_info_vertex_mapped.shape)

        if self.distance_threshold is not None:
            self.remove_samples_far_from_surface()

    def remove_samples_far_from_surface(self):
        print(
            f"Removing samples more than {self.distance_threshold}mm from the surface..."
        )
        original_sample_num = self.sample_info_vertex_mapped.shape[0]
        # remove samples more than distance_threshold mm from the surface
        self.sample_info_vertex_mapped = self.sample_info_vertex_mapped[
            self.sample_info_vertex_mapped["mm_to_surf"].abs() < self.distance_threshold
        ]
        self.expression = self.expression.loc[self.sample_info_vertex_mapped.index]
        print(
            f"Removed {original_sample_num - self.sample_info_vertex_mapped.shape[0]} samples."
        )
        print("After removing shapes: ")
        print("Expression matrix: ", self.expression.shape)
        print("Sample information: ", self.sample_info_vertex_mapped.shape)

    @property
    def get_expression(self):
        return self.expression

    @property
    def get_sample_info(self):
        return self.sample_info_vertex_mapped

    def assign_atlas_to_info(self, lh_parc=None, rh_parc=None):
        if lh_parc is not None:
            lh_parc = load_gii(lh_parc).agg_data()
            lh_unique_label = np.unique(lh_parc)
            lh_index = self.sample_info_vertex_mapped[
                self.sample_info_vertex_mapped["hemi"] == "left"
            ].index
            self.sample_info_vertex_mapped.loc[lh_index, "atlas_label"] = lh_parc[
                self.sample_info_vertex_mapped.loc[lh_index, "vertex"].values
            ]
            lh_label_assiged = self.sample_info_vertex_mapped.loc[
                lh_index, "atlas_label"
            ].unique()
            self.lh_label_unassigned = np.setdiff1d(lh_unique_label, lh_label_assiged)
            print(
                f"Left hemisphere: {len(self.lh_label_unassigned)} labels unassigned."
            )
        else:
            self.lh_label_unassigned = None

        if rh_parc is not None:
            rh_parc = load_gii(rh_parc).agg_data()
            rh_unique_label = np.unique(rh_parc)
            rh_index = self.sample_info_vertex_mapped[
                self.sample_info_vertex_mapped["hemi"] == "right"
            ].index
            self.sample_info_vertex_mapped.loc[rh_index, "atlas_label"] = rh_parc[
                self.sample_info_vertex_mapped.loc[rh_index, "vertex"].values
            ]
            rh_label_assiged = self.sample_info_vertex_mapped.loc[
                rh_index, "atlas_label"
            ].unique()
            self.rh_label_unassigned = np.setdiff1d(rh_unique_label, rh_label_assiged)
            print(
                f"Right hemisphere: {len(self.rh_label_unassigned)} labels unassigned."
            )
        else:
            self.rh_label_unassigned = None

        if lh_parc is None and rh_parc is None:
            raise ValueError("Please provide at least one parcellation file.")

        self.sample_info_vertex_mapped.dropna(subset=["atlas_label"], inplace=True)
        self.sample_info_vertex_mapped = self.sample_info_vertex_mapped.loc[
            self.sample_info_vertex_mapped["atlas_label"] != 0
        ]
        self.expression = self.expression.loc[self.sample_info_vertex_mapped.index]

        return self.sample_info_vertex_mapped

    def aggregate_expression(self, return_donors=False, agg_metric="mean"):
        # aggregate expression by atlas_label
        # samples were first averaged at the individual donor level within parcels and then averaged across donors.
        donor_expression = {}
        for donor in donors:
            # get the expression of the donor
            donor_expression[donor] = self.expression[
                self.sample_info_vertex_mapped["brain"] == donor
            ]
            # aggregate the expression by atlas_label
            if agg_metric == "mean":
                donor_expression[donor] = (
                    donor_expression[donor]
                    .groupby(self.sample_info_vertex_mapped["atlas_label"])
                    .mean()
                )
            elif agg_metric == "median":
                donor_expression[donor] = (
                    donor_expression[donor]
                    .groupby(self.sample_info_vertex_mapped["atlas_label"])
                    .median()
                )
            # add unassigned labels with NaN
            if self.lh_label_unassigned is not None:
                for label in self.lh_label_unassigned:
                    donor_expression[donor].loc[label] = np.nan
            if self.rh_label_unassigned is not None:
                for label in self.rh_label_unassigned:
                    donor_expression[donor].loc[label] = np.nan
            donor_expression[donor].sort_index(inplace=True)

        if return_donors:
            return donor_expression
        else:
            # aggregate the expression across donors
            agg_expression = pd.concat(donor_expression)
            if agg_metric == "mean":
                agg_expression = agg_expression.groupby("atlas_label").mean()
            elif agg_metric == "median":
                agg_expression = agg_expression.groupby("atlas_label").median()

            return agg_expression
