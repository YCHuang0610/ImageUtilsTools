# ImageUtilsTools

ImageUtilsTools is a Python package for brain surface visualization and transcriptomic association analysis. It includes a variety of tools for surface manipulation and statistical methods. 

## Installation

You can install this package using the following command:

```sh
pip install git+https://github.com/YCHuang0610/ImageUtilsTools.git
```

or

```sh
git clone https://github.com/YCHuang0610/ImageUtilsTools.git
cd ImageUtilsTools
pip install .
```

## Structure

- ImageUtilsTools/
    - GeneAnalysis/ - Contains tools for imaging transcriptomic analysis
    - Plotting/ - Contains tools for surface plotting
    - Stats/ - Contains tools for statistical analysis
    - utils/ - Contains utility tools for surface data processing or basic math operations

## Usage example

Plot a surface data with the HCP-MMP1.0 parcellation (Glasser2016).

```python
from ImageUtilsTools.Plotting import *
from ImageUtilsTools.Stats.surf_local_correlation import surflocalcorr
from ImageUtilsTools.GeneAnalysis._data_config import data
from neuromaps.datasets import fetch_fslr

hum_surfaces = fetch_fslr(density='32k')
hum_lh, hum_rh = hum_surfaces['inflated']
hum_lh_parc, hum_rh_parc = data['Glasser2016_lh_parc'], data['Glasser2016_rh_parc']

plot_parc_data_lr = np.random.rand(360)
figure = Plot_MySurf_RegionWise(plot_parc_data_lr, hum_lh_parc, hum_rh_parc, hum_lh, hum_rh)
```
