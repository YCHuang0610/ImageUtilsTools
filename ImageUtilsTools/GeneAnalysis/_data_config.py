import os

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

data = {
    "Glasser2016_lh_parc": os.path.join(DATA, "Glasser2016.L.32k_fs_LR.label.gii"),
    "Glasser2016_rh_parc": os.path.join(
        DATA, "GeneAnalysis/data/Glasser2016.R.32k_fs_LR.label.gii"
    ),
    "Glasser2016_left_dist_mat_file": os.path.join(
        DATA, "GeneAnalysis/data/LeftParcelGeodesicDistmat_Glasser2016_180regions.txt"
    ),
    "Glasser2016_right_dist_mat_file": os.path.join(
        DATA, "GeneAnalysis/data/RightParcelGeodesicDistmat_Glasser2016_180regions.txt"
    ),
    "Glasser2016_AHBA_abagen_LR_noheader": os.path.join(
        DATA, "GeneAnalysis/data/ahba_expression_Glasser2016LR_noHeader.csv"
    ),
    "Glasser2016_AHBA_gene_list": os.path.join(
        DATA, "GeneAnalysis/data/genename_Glasser2016LR.csv"
    ),
    "Glasser2016_AHBA_GSVA_LR_withheader": os.path.join(
        DATA, "GeneAnalysis/data/ahba_GSVA_GO_Glassser2016LR_withHeader.csv"
    ),
}
