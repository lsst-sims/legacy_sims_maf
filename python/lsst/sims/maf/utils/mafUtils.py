import os, sys, importlib
import numpy as np
import warnings


def moduleLoader(moduleList):
    """
    Load additional modules (beyond standard MAF modules) provided by the user at runtime.
    If the modules contain metrics, slicers or stackers inheriting from MAF base classes, these
    will then be available from the driver configuration file identified by 'modulename.classname'.
    """
    for m in moduleList:
        importlib.import_module(m)


def optimalBins(datain, binmin=None, binmax=None, nbinMax=1e3):
    """
    Use Freedman-Diaconis rule to set binsize.
    Allow user to pass min/max data values to consider.
    nbinMax sets the maximum value (to keep it from trying to make a trillion bins)
    """
    # if it's a masked array, only use unmasked values
    if hasattr(datain, 'compressed'):
        data = datain.compressed()
    else:
        data = datain
    if binmin is None:
        if data.size == 0:
            binmin = 0
        else:
            binmin = data.min()
    if binmax is None:
        if data.size == 0:
            binmax = 1
        else:
            binmax = data.max()
    cond = np.where((data >= binmin)  & (data <= binmax))[0]
    if np.size(data[cond]) > 0:
        binwidth = (2.*(np.percentile(data[cond], 75) - np.percentile(data[cond], 25))
                    /np.size(data[cond])**(1./3.))
        nbins = (binmax - binmin) / binwidth
        if nbins > nbinMax:
            warnings.warn('Warning! Optimal bin calculation tried to make %f bins, returning %i'%(nbins, nbinMax))
            nbins = nbinMax
    else:
        warnings.warn('Warning! No data available for calculating optimal bin size within range of %f, %f'
                      %(binmin, binmax), ' returning %i' %(nbinMax))
        nbins = nbinMax
    if np.isnan(nbins):
        warnings.warn('Warning! Optimal bin calculation calculated NaN: returning %i' %(nbinMax))
        nbins = nbinMax
    return int(nbins)


def percentileClipping(data, percentile=95.):
    """
    Clip off high and low outliers from a distribution in a numpy array.
    Returns the max and min values of the clipped data.
    Useful for determining plotting ranges.  
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
