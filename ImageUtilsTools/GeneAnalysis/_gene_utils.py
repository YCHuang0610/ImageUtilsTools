"""
Author: [Yichun Huang]
Date: [12/26/2024]

This module contains functions and classes for performing gene annotation in python.

"""

import pandas as pd
import gzip

def parse_attributes(attribute_string):
    """
    Parse the attribute string in a GTF file.
    """
    attributes = {}
    for attribute in attribute_string.split(';'):
        if attribute.strip():
            key, value = attribute.strip().split(' ')
            attributes[key] = value.strip('"')
    return attributes

def parse_GTF(gtf_file_path, return_attributes_only=True):
    """
    Parse GTF file and return a DataFrame with gene attributes.

    Parameters
    ----------
    gtf_file_path : str
        Path to the GTF file.
    return_attributes_only : bool, optional
        If True, only the gene attributes are returned. If False, the entire GTF file is returned along with the gene attributes. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene attributes.
    """
    # Read GTF file
    if gtf_file_path.endswith('.gz'):
        with gzip.open(gtf_file_path, 'rt') as f:
            gtf = pd.read_csv(f, sep='\t', comment='#', header=None, low_memory=False)
    else:
        gtf = pd.read_csv(gtf_file_path, sep='\t', comment='#', header=None, low_memory=False)

    # Extract gene attributes
    attr = gtf.iloc[:, -1].apply(parse_attributes)
    attr_df = pd.json_normalize(attr)

    if return_attributes_only:
        return attr_df
    
    else:
        gtf.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        return gtf, attr_df
