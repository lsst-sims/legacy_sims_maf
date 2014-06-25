import numpy as np

def optimalBins(datain, min=None, max=None):
    """Use Freedman-Diaconis rule to set binsize.
    Allow user to pass min/max data values to consider."""
    # if it's a masked array, only use unmasked values
    if hasattr(datain, 'compressed'):
        data = datain.compressed()
    else:
        data = datain
    if min is None:
        min = data.min()
    if max is None:
        max = data.max()
    condition = ((data >= min)  & (data <= max))
    binwidth = 2.*(np.percentile(data[condition],75) - np.percentile(data[condition],25))/np.size(data[condition])**(1./3.)
    nbins = (max - min) / binwidth
    if np.isinf(nbins) or np.isnan(nbins):
        return 1
    else:
        return int(nbins)



def percentileClipping(data, percentile=95.):
    """
    Clip off high and low outliers from a distribution in a numpy array.
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
