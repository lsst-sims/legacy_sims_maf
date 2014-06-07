import numpy as np

def percentileClip(data, percentile=95.):
    """
    Clip off high and/or low outliers from a distribution in a numpy array.
    Returns the max and min values that are inside.
    """
    if np.size(data) > 0:
        # Use absolute value to get both high and low outliers.
        temp_data = np.abs(data-np.median(data))
        indx = np.argsort(temp_data)
        # Find the indices of those values which are closer than percentile to the median.
        indx = indx[:len(indx)*percentile/100.]
        # Find min/max values of those (original) data values.
        min_value = data[indx].min()
        max_value = data[indx].max()
    else:
        min_value = 0
        max_value = 0
    return  min_value, max_value

