# %% 
import numpy as np
from neuromaps.datasets import fetch_fslr
from ImageUtilsTools.Plotting import Plot_MySurf_VertexWise

# %% 1. Visualization
# 1.1 Vertex data

surfaces = fetch_fslr(density="32k")
lh, rh = surfaces["inflated"]

lh_myelin, rh_myelin = [
    "metric/S1200.MyelinMap.L.shape.gii",
    "metric/S1200.MyelinMap.R.shape.gii",
]

figure = Plot_MySurf_VertexWise(
    lh_myelin,
    rh_myelin,
    lh,
    rh,
    cmap="inferno",
    color_range=(1, 1.5),
    title="Myelin Map",
)

# %%
lh_thick, rh_thick = [
    "metric/S1200.thickness.L.shape.gii",
    "metric/S1200.thickness.R.shape.gii",
]

figure = Plot_MySurf_VertexWise(
    lh_thick,
    rh_thick,
    lh,
    rh,
    cmap="inferno",
    color_range=(1.5, 3.3),
    layout="row",
    size=(1000, 200),
    title="Cortical Thickness",
)
# %%
import nibabel as nib

lh_flat, rh_flat = [
    'anat/S1200.L.flat.32k_fs_LR.surf.gii',
    'anat/S1200.R.flat.32k_fs_LR.surf.gii'
]

lh_myelin, rh_myelin = [
    "metric/S1200.MyelinMap.L.shape.gii",
    "metric/S1200.MyelinMap.R.shape.gii",
]

figure = Plot_MySurf_VertexWise(
    lh_myelin,
    rh_myelin,
    lh_flat,
    rh_flat,
    views=["dorsal"],
    cmap="inferno",
    color_range=(1, 1.5),
    title="Myelin Map",
    size=(600, 200),
)
# %%
# 1.2 Region wise data
from ImageUtilsTools.Plotting import Plot_Each_Region_Num

lh_parc, rh_parc = [
    "parc/L.Schaefer2018_400Parcels_7Networks.32k_fs_LR.label.gii",
    "parc/R.Schaefer2018_400Parcels_7Networks.32k_fs_LR.label.gii",
]

figure = Plot_Each_Region_Num(
    [1, 5, 10, 50, 100, 150],
    lh_parc,
    lh,
    as_outline=True,
    title="Some selected regions of Schaefer 400",
)

# %%
import nibabel as nib
from ImageUtilsTools.utils.parcellater import parcellate_surface_data
from ImageUtilsTools.Plotting import Plot_MySurf_RegionWise

# myelin
lh_myelin_data = nib.load(lh_myelin).darrays[0].data
rh_myelin_data = nib.load(rh_myelin).darrays[0].data

lh_myelin_parc = parcellate_surface_data(lh_myelin_data, lh_parc).squeeze()
rh_myelin_parc = parcellate_surface_data(rh_myelin_data, rh_parc).squeeze()

figure = Plot_MySurf_RegionWise(
    np.concatenate([lh_myelin_parc, rh_myelin_parc]),
    lh_parc,
    rh_parc,
    lh,
    rh,
    cmap="inferno",
    color_range=(1, 1.5),
    title="Myelin Map on Schaefer 400",
)

# thickness
lh_thick_data = nib.load(lh_thick).darrays[0].data
rh_thick_data = nib.load(rh_thick).darrays[0].data

lh_thick_parc = parcellate_surface_data(lh_thick_data, lh_parc).squeeze()
rh_thick_parc = parcellate_surface_data(rh_thick_data, rh_parc).squeeze()

figure = Plot_MySurf_RegionWise(
    np.concatenate([lh_thick_parc, rh_thick_parc]),
    lh_parc,
    rh_parc,
    lh,
    rh,
    cmap="inferno",
    color_range=(1.5, 3.3),
    title="Cortical Thickness on Schaefer 400",
)
# %%
# 1.3 Volume data
from ImageUtilsTools.Plotting import Plot_MySurf_mni152Volume
from nilearn.datasets import fetch_neurovault_ids

data = fetch_neurovault_ids(image_ids=[47307], verbose=0)
img = data['images'][0]

figure = Plot_MySurf_mni152Volume(
    img,
    cutoff=None,
    two_side=False,
    cmap="bwr",
    color_range=(-4, 4),
    cbar=True,
    title="MSC05 Left > Right Hand",
    layout="row",
    size=(1000, 200),
    brightness=0.7,
)
figure = Plot_MySurf_mni152Volume(
    img,
    cutoff=3,
    two_side=False,
    cmap="hot",
    color_range=(3, 5),
    cbar=True,
    title="MSC05 Left > Right Hand Significant",
    layout="row",
    size=(1000, 200),
    brightness=0.7,
)
# %%
# 2. Correlation analysis and Spin Test
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

data = pd.DataFrame({"Myelin": np.concatenate([lh_myelin_parc, rh_myelin_parc]), 
                    "Thickness": np.concatenate([lh_thick_parc, rh_thick_parc])})
r, p = pearsonr(data["Myelin"], data["Thickness"])

sns.regplot(x="Myelin", y="Thickness", data=data)
plt.text(1.5, 3.5, f"r = {r:.5f}, p = {p:.5f}", fontsize=12)
# %%
# Spin test
from neuromaps import stats
from ImageUtilsTools.Stats.spin_test_utils import generate_spin_permutation

null_model = generate_spin_permutation(
    data['Thickness'],
    (lh_parc, rh_parc),
    perm=1000,
    method='vasa',
)

r, p = stats.compare_images(
    src=data['Thickness'],
    trg=data['Myelin'],
    nulls=null_model,
    metric='pearsonr'
)

print(f"r = {r:.5f}, p_spin = {p:.5f}")

sns.regplot(x="Myelin", y="Thickness", data=data)
plt.text(1.5, 3.5, f"r = {r:.5f}, p_spin = {p:.5f}", fontsize=12)
# %%
# 3. Local correlation analysis
from ImageUtilsTools.Stats.surf_local_correlation import surflocalcorr

lh_sphere, rh_sphere = [
    "anat/S1200.L.sphere.32k_fs_LR.surf.gii",
    "anat/S1200.R.sphere.32k_fs_LR.surf.gii",
]

lh_thickxmyelin = surflocalcorr(lh_thick, lh_myelin, a=30, sph=lh_sphere, method='pearsonr')
rh_thickxmyelin = surflocalcorr(rh_thick, rh_myelin, a=30, sph=rh_sphere, method='pearsonr')

figure = Plot_MySurf_VertexWise(
    lh_thickxmyelin,
    rh_thickxmyelin,
    lh,
    rh,
    layout='row',
    cmap='coolwarm',
    cbar=True,
    title='Local Correlation between Thickness and Myelin',
    size=(1000, 200),
    color_range=(-0.5, 0.5)
)
# %%
# 4. Gifti manipulation
# 4.1 Single column file

from ImageUtilsTools.utils._gii_io import Save_Gii

Save_Gii(lh_thickxmyelin, file='lh_thickxmyelin.shape.gii')

# %%
# 4.2 Multiple column file
print(lh_myelin_data.shape, lh_thick_data.shape, lh_thickxmyelin.shape)

lh_data_to_save = np.column_stack([lh_myelin_data, lh_thick_data, lh_thickxmyelin])
print(lh_data_to_save.shape)

Save_Gii(lh_data_to_save, file='lh_myelin_thickness_corr.shape.gii')