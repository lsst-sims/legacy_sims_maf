import numpy as np

def percentileClip(data, percentile=95.):
    """clip off high and/or low outliers from a distribution.
    Returns the max and min values that are inside"""
    temp_data = np.abs(data-np.median(data))
    indx = np.argsort(temp_data)
    indx = indx[:len(indx)*percentile/100.]
    min_value = data[indx].min()
    max_value = data[indx].max()
    return  min_value, max_value

