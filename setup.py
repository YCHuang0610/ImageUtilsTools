from setuptools import setup, find_packages

setup(
    name="ImageUtilsTools",
    version="0.0.1",
    description='...',
    author="Yichun Huang",
    author_email="ychunhuang@foxmail.com",
    url='...',
    packages=find_packages(),
    install_requires=['matplotlib', 'pandas', 'tqdm', 'surfplot', 'numpy', 'scipy', 'scikit-learn', 'neuromaps', 'nilearn', 'nibabel', 'numba', 'brainsmash', 'statsmodels', 'seaborn'],
    python_requires=">=3.6,<3.11",
    keywords=['statical analysis', 'neuroimaging'],
    license='GPLv3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)