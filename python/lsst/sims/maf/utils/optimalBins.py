def optimalBins(data):
    """Use Freedman-Diaconis rule to set binsize"""
    binwidth = 2.*(np.percentile(data,75) - np.percentile(data,25))/np.size(data)**(1./3.)
    nbins = (data.max()-data.min())/binwidth
    return nbins
