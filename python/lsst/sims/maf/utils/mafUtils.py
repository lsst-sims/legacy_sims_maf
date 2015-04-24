import importlib
import os
import numpy as np
import warnings

__all__ = ['moduleLoader', 'connectResultsDb', 'optimalBins', 'percentileClipping', 'gnomonic_project_toxy']

def moduleLoader(moduleList):
    """
    Load additional modules (beyond standard MAF modules) provided by the user at runtime.
    If the modules contain metrics, slicers or stackers inheriting from MAF base classes, these
    will then be available from the driver configuration file identified by 'modulename.classname'.
    """
    for m in moduleList:
        importlib.import_module(m)

def connectResultsDb(dbDir, dbFilename='resultsDb_sqlite.db'):
    """
    Connect to a MAF-generated results database (usually called 'resultsDb_sqlite.db').
    """
    import lsst.sims.maf.db as db
    dbAddress = 'sqlite:///' + os.path.join(dbDir, dbFilename)
    database = db.Database(dbAddress, longstrings=True,
                            dbTables={'metrics':['metrics','metricID'],
                                      'displays':['displays', 'displayId'],
                                      'plots':['plots','plotId'],
                                      'stats':['summarystats','statId']})
    return database

def optimalBins(datain, binmin=None, binmax=None, nbinMax=200, nbinMin=1):
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
    # Check that any good data values remain.
    if data.size == 0:
        nbins = nbinMax
        warnings.warn('No unmasked data available for calculating optimal bin size: returning %i bins' %(nbins))
    # Else proceed.
    else:
        if binmin is None:
            binmin = data.min()
        if binmax is None:
            binmax = data.max()
        cond = np.where((data >= binmin)  & (data <= binmax))[0]
        # Check if any data points remain within binmin/binmax.
        if np.size(data[cond]) == 0:
            nbins = nbinMax
            warnings.warn('No data available for calculating optimal bin size within range of %f, %f'
                          %(binmin, binmax) + ': returning %i bins' %(nbins))
        else:
            iqr = np.percentile(data[cond], 75) - np.percentile(data[cond], 25)
            binwidth = 2 * iqr * (np.size(data[cond])**(-1./3.))
            nbins = (binmax - binmin) / binwidth
            if nbins > nbinMax:
                warnings.warn('Optimal bin calculation tried to make %.0f bins, returning %i'%(nbins, nbinMax))
                nbins = nbinMax
            if nbins < nbinMin:
                warnings.warn('Optimal bin calculation tried to make %.0f bins, returning %i'%(nbins, nbinMin))
                nbins = nbinMin
    if np.isnan(nbins):
        warnings.warn('Optimal bin calculation calculated NaN: returning %i' %(nbinMax))
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

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians."""
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y
