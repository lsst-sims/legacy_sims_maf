import numpy as np

def optimalBins(datain):
    """Use Freedman-Diaconis rule to set binsize"""
    # if it's a masked array, only use unmasked values
    if hasattr(datain, 'compressed'):
        data = datain.compressed()
    else:
        data = datain
    if np.size(data) == 0:
        return 1
    else:
        binwidth = 2.*(np.percentile(data,75) - np.percentile(data,25))/np.size(data)**(1./3.)
        nbins = int((data.max()-data.min())/binwidth)
        return nbins
