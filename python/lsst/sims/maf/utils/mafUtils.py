import importlib
import os
import numpy as np
from scipy.spatial import cKDTree as kdtree
from lsst.sims.survey.fields import FieldsDatabase
import healpy as hp
import warnings

__all__ = ['optimalBins', 'percentileClipping',
           'gnomonic_project_toxy', 'radec2pix',
           'getOpSimField', 'treexyz', 'rad_length']


def optimalBins(datain, binmin=None, binmax=None, nbinMax=200, nbinMin=1):
    """
    Set an 'optimal' number of bins using the Freedman-Diaconis rule.

    Parameters
    ----------
    datain : numpy.ndarray or numpy.ma.MaskedArray
        The data for which we want to set the binsize.
    binmin : float
        The minimum bin value to consider (if None, uses minimum data value).
    binmax : float
        The maximum bin value to consider (if None, uses maximum data value).
    nbinMax : int
        The maximum number of bins to create. Sometimes the 'optimal binsize' implies
        an unreasonably large number of bins, if the data distribution is unusual.
    nbinMin : int
        The minimum number of bins to create. Default is 1.

    Returns
    -------
    int
        The number of bins.
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
            binmin = np.nanmin(data)
        if binmax is None:
            binmax = np.nanmax(data)
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
    Calculate the minimum and maximum values of a distribution of points, after
    discarding data more than 'percentile' from the median.
    This is useful for determining useful data ranges for plots.
    Note that 'percentile' percent of the data is retained.

    Parameters
    ----------
    data : numpy.ndarray
        The data to clip.
    percentile : float
        Retain values within percentile of the median.

    Returns
    -------
    float, float
        The minimum and maximum values of the clipped data.
    """
    lower_percentile = (100 - percentile) / 2.0
    upper_percentile = 100 - lower_percentile
    min_value = np.percentile(data, lower_percentile)
    max_value = np.percentile(data, upper_percentile)
    return  min_value, max_value

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """
    Calculate the x/y values of RA1/Dec1 in a gnomonic projection with center at RAcen/Deccen.

    Parameters
    ----------
    RA1 : numpy.ndarray
        RA values of the data to be projected, in radians.
    Dec1 : numpy.ndarray
        Dec values of the data to be projected, in radians.
    RAcen: float
        RA value of the center of the projection, in radians.
    Deccen : float
        Dec value of the center of the projection, in radians.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The x/y values of the projected RA1/Dec1 positions.
    """
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def radec2pix(nside, ra, dec):
    """
    Calculate the nearest healpixel ID of an RA/Dec array, assuming nside.

    Parameters
    ----------
    nside : int
        The nside value of the healpix grid.
    ra : numpy.ndarray
        The RA values to be converted to healpix ids, in radians.
    dec : numpy.ndarray
        The Dec values to be converted to healpix ids, in radians.

    Returns
    -------
    numpy.ndarray
        The healpix ids.
    """
    lat = np.pi/2. - dec
    hpid = hp.ang2pix(nside, lat, ra )
    return hpid


def getOpSimField(sqlconstraint="select * from Field"):
    """
    Get list of OpSim fields.

    Parameters:
    -----------
    sqlconstraint : string
        Sql constraints for the field selection. Default is, get all fields.

    Returns:
    --------
    numpy.ndarray
        A numpy structured array with columns for fields matching the sqlconstraint.
    """

    db = FieldsDatabase()
    res = db.get_field_set(sqlconstraint)
    names = ['field_id', 'fov_rad', 'RA', 'dec', 'gl', 'gb', 'el', 'eb']
    types = [int, float, float, float, float, float, float, float]
    fields = np.zeros(len(res), dtype=list(zip(names, types)))

    for i, row in enumerate(res):
        fields['field_id'][i] = row[0]
        fields['fov_rad'][i] = np.radians(row[1])
        fields['RA'][i] = np.radians(row[2])
        fields['dec'][i] = np.radians(row[3])
        fields['gl'][i] = row[4]
        fields['gb'][i] = row[5]
        fields['el'][i] = row[6]
        fields['eb'][i] = row[7]

    return fields


def opsimfields_kd_tree(leafsize=100):
    """
    Generate a KD-tree of OpSim fields locations

    Parameters
    ----------
    leafsize : int (100)
        Leafsize of the kdtree

    Returns
    -------
    tree : scipy kdtree
    """

    fields = getOpSimField()
    x, y, z = treexyz(fields['RA'], fields['dec'])
    tree = kdtree(list(zip(x, y, z)), leafsize=leafsize, balanced_tree=False, compact_nodes=False)
    return tree


def treexyz(ra, dec):
    """
    Utility to convert RA,dec postions in x,y,z space, useful for constructing KD-trees.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def rad_length(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius), 0)
    result = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return result