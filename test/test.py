# %%
import os
os.chdir('..')
import numpy as np

from neuromaps.datasets import fetch_fslr
from ImageUtilsTools.Plotting.plot_utils import *
from ImageUtilsTools.Stats.surf_local_correlation import surflocalcorr
from ImageUtilsTools.GeneAnalysis._data_config import data
from ImageUtilsTools.utils._gii_io import load_gii

# %%
hum_surfaces = fetch_fslr(density='32k')
hum_lh, hum_rh = hum_surfaces['inflated']
hum_lh_parc, hum_rh_parc = data['Glasser2016_lh_parc'], data['Glasser2016_rh_parc']

test = load_gii(hum_lh)
test = load_gii(test)
# %%
# 1. Plotting test
plot_parc_data_lr = np.random.rand(360)
plot_vertex_data = np.random.rand(32492)

figure = Plot_MySurf_RegionWise(plot_parc_data_lr, hum_lh_parc, hum_rh_parc, hum_lh, hum_rh)

figure = Plot_MySurf_VertexWise(plot_vertex_data, plot_vertex_data, hum_lh, hum_rh)

figure = Plot_MySurf_RegionWise_OneHemi(plot_parc_data_lr[:180], hum_lh_parc, hum_lh, as_outline=True)

figure = Plot_Each_Region_Num([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], hum_lh_parc, hum_lh)

# %%
# 2. Stats local correlation test
gii_1_L = 'test/data/Monkey_GC1_nPDF_to_Human.L.32k_fs_LR.shape.gii'
gii_2_L = 'test/data/Monkey_GC1_PDF_to_Human.L.32k_fs_LR.shape.gii'
gii_1_R = 'test/data/Monkey_GC1_nPDF_to_Human.R.32k_fs_LR.shape.gii'
gii_2_R = 'test/data/Monkey_GC1_PDF_to_Human.R.32k_fs_LR.shape.gii'
sph_L, sph_R = hum_surfaces['sphere']

corr_L = surflocalcorr(gii_1_L, gii_2_L, sph_L, a=30, method="spearmanr")
figure = Plot_MySurf_VertexWise(corr_L, corr_L, hum_lh, hum_rh, cmap='bwr')

# %%
