import os, sys, warnings
import numpy as np

# Example of adding modules:
#moduleDict = makeDict(['~/myMetrics.py', 'desc/SNmetrics']
#root.modules = moduleDict

# Example __init__.py file
#from .SNmetrics import *
#from .myNewSNmetrics import *

# Example metric configuration:
#m1 = configureMetric('PercentileMetric', params=['Airmass'], kwargs={'percentile':75})
#slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1),
#                         constraints=[''])

def moduleLoader(moduleList):
    for m in moduleList:
        mpath, mname = os.path.split(m)
        mname = mname.replace('.py', '')
        if len(mpath) > 0:
            if mpath == '~':
                mpath = os.getenv('HOME')
            if mpath not in sys.path:
                sys.path.insert(0, mpath)
            os.listdir(mpath)
        if mname == '~':
            warnings.warn('Warning! Cannot import modules directly from home directory.')
            continue
        __import__(mname)


def optimalBins(datain, binmin=None, binmax=None):
    """Use Freedman-Diaconis rule to set binsize.
    Allow user to pass min/max data values to consider."""
    # if it's a masked array, only use unmasked values
    if hasattr(datain, 'compressed'):
        data = datain.compressed()
    else:
        data = datain
    if binmin is None:
        binmin = data.min()
    if binmax is None:
        binmax = data.max()
    condition = ((data >= binmin)  & (data <= binmax))
    binwidth = 2.*(np.percentile(data[condition],75) - np.percentile(data[condition],25))/np.size(data[condition])**(1./3.)
    nbins = (binmax - binmin) / binwidth
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
