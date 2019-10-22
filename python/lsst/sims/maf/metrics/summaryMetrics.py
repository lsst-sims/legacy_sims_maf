import numpy as np
import healpy as hp
from .baseMetric import BaseMetric

# A collection of metrics which are primarily intended to be used as summary statistics.

__all__ = ['fOArea', 'fONv', 'TableFractionMetric', 'IdentityMetric',
           'NormalizeMetric', 'ZeropointMetric', 'TotalPowerMetric']


class fONv(BaseMetric):
    """
    Metrics based on a specified area, but returning NVISITS related to area:
    given Asky, what is the minimum and median number of visits obtained over that much area?
    (choose the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Asky : float, opt
        Area of the sky to base the evaluation of number of visits over.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    Nvisit : int, opt
        Number of visits to use as the benchmark value, if choosing to return a normalized Nvisit value.
    norm : boolean, opt
        Normalize the returned "nvisit" (min / median) values by Nvisit, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fONv.
    """
    def __init__(self, col='metricdata', Asky=18000., nside=128, Nvisit=825,
                 norm=False, metricName='fONv',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        # Determine how many healpixels are included in Asky sq deg.
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.npix_Asky = np.int(np.ceil(self.Asky / self.scale))
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        result = np.empty(2, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'][0] = "MedianNvis"
        result['name'][1] = "MinNvis"
        # If there is not even as much data as needed to cover Asky:
        if len(dataSlice) < self.npix_Asky:
            # Return the same type of metric value, to make it easier downstream.
            result['value'][0] = self.badval
            result['value'][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis_Asky = nvis_sorted[-self.npix_Asky:]
        result['value'][0] = np.median(nvis_Asky)
        result['value'][1] = np.min(nvis_Asky)
        if self.norm:
            result['value'] /= float(self.Nvisit)
        return result


class fOArea(BaseMetric):
    """
    Metrics based on a specified number of visits, but returning AREA related to Nvisits:
    given Nvisit, what amount of sky is covered with at least that many visits?

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Nvisit : int, opt
        Number of visits to use as the minimum required -- metric calculated area that has this many visits.
        Default 825.
    Asky : float, opt
        Area to use as the benchmark value, if choosing to returned a normalized Area value.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    norm : boolean, opt
        Normalize the returned "area" (area with minimum Nvisit visits) value by Asky, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fOArea.
    """
    def __init__(self, col='metricdata', Nvisit=825, Asky = 18000.0, nside=128,
                  norm=False, metricName='fOArea',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Identify the healpixels with more than Nvisits.
        nvis_min = nvis_sorted[np.where(nvis_sorted >= self.Nvisit)]
        if len(nvis_min) == 0:
            result = self.badval
        else:
            result = nvis_min.size * self.scale
            if self.norm:
                result /= float(self.Asky)
        return result


class TableFractionMetric(BaseMetric):
    """
    Count the completeness (for many fields) and summarize how many fields have given completeness levels
    (within a series of bins). Works with completenessMetric only.

    This metric is meant to be used as a summary statistic on something like the completeness metric.
    The output is DIFFERENT FROM SSTAR and is:
    element   matching values
    0         0 == P
    1         0 < P < .1
    2         .1 <= P < .2
    3         .2 <= P < .3
    ...
    10        .9 <= P < 1
    11        1 == P
    12        1 < P
    Note the 1st and last elements do NOT obey the numpy histogram conventions.
    """
    def __init__(self, col='metricdata',  nbins=10, maskVal=0.):
        """
        colname = the column name in the metric data (i.e. 'metricdata' usually).
        nbins = number of bins between 0 and 1. Should divide evenly into 100.
        """
        super(TableFractionMetric, self).__init__(col=col, maskVal=maskVal, metricDtype='float')
        self.nbins = nbins

    def run(self, dataSlice, slicePoint=None):
        # Calculate histogram of completeness values that fall between 0-1.
        goodVals = np.where((dataSlice[self.colname] > 0) & (dataSlice[self.colname] < 1)  )
        bins = np.arange(self.nbins+1.)/self.nbins
        hist, b = np.histogram(dataSlice[self.colname][goodVals], bins=bins)
        # Fill in values for exact 0, exact 1 and >1.
        zero = np.size(np.where(dataSlice[self.colname] == 0)[0])
        one = np.size(np.where(dataSlice[self.colname] == 1)[0])
        overone = np.size(np.where(dataSlice[self.colname] > 1)[0])
        hist = np.concatenate((np.array([zero]), hist, np.array([one]), np.array([overone])))
        # Create labels for each value
        binNames = ['0 == P']
        binNames.append('0 < P < 0.1')
        for i in np.arange(1, self.nbins):
            binNames.append('%.2g <= P < %.2g'%(b[i], b[i+1]) )
        binNames.append('1 == P')
        binNames.append('1 < P')
        # Package the names and values up
        result = np.empty(hist.size, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'] = binNames
        result['value'] = hist
        return result


class IdentityMetric(BaseMetric):
    """
    Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
    """
    def run(self, dataSlice, slicePoint=None):
        if len(dataSlice[self.colname]) == 1:
            result = dataSlice[self.colname][0]
        else:
            result = dataSlice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions.
    """
    def __init__(self, col='metricdata', normVal=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.normVal = float(normVal)
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname]/self.normVal
        if len(result) == 1:
            return result[0]
        else:
            return result

class ZeropointMetric(BaseMetric):
    """
    Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
    """
    def __init__(self, col='metricdata', zp=0, **kwargs):
        super(ZeropointMetric, self).__init__(col=col, **kwargs)
        self.zp = zp
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname] + self.zp
        if len(result) == 1:
            return result[0]
        else:
            return result

class TotalPowerMetric(BaseMetric):
    """
    Calculate the total power in the angular power spectrum between lmin/lmax.
    """
    def __init__(self, col='metricdata', lmin=100., lmax=300., removeDipole=True, maskVal=hp.UNSEEN, **kwargs):
        self.lmin = lmin
        self.lmax = lmax
        self.removeDipole = removeDipole
        super(TotalPowerMetric, self).__init__(col=col, maskVal=maskVal, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Calculate the power spectrum.
        if self.removeDipole:
            cl = hp.anafast(hp.remove_dipole(dataSlice[self.colname], verbose=False))
        else:
            cl = hp.anafast(dataSlice[self.colname])
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition]*(2*ell[condition]+1))
        return totalpower
