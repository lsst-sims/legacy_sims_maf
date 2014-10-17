import numpy as np
import healpy as hp
from .baseMetric import BaseMetric

# A collection of metrics which are primarily intended to be used as summary statistics.

class fOArea(BaseMetric):
    """
    Metric to calculate the FO Area; works with FO slicer only.
    """
    def __init__(self, col='metricdata', Asky=18000., Nvisit=825,
                 metricName='fOArea', nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(fOArea, self).__init__(col=col, metricName=metricName, **kwargs)
        self.Asky = Asky
        self.Nvisit = Nvisit
        self.nside = nside
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort()
        name = dataSlice.dtype.names[0]
        scale = hp.nside2pixarea(self.nside, degrees=True)
        cumulativeArea = np.arange(1,dataSlice.size+1)[::-1]*scale
        good = np.where(cumulativeArea >= self.Asky)[0]
        if good.size > 0:
            nv = np.max(dataSlice[name][good])
            if self.norm:
                nv = nv/float(self.Nvisit)
            return nv
        else:
            return self.badval


class fONv(BaseMetric):
    """
    Metric to calculate the FO_Nv; works with FO slicer only.
    """
    def __init__(self, col='metricdata', Asky=18000., metricName='fONv', Nvisit=825,
                 nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(fONv, self).__init__(col=col, metricName=metricName, **kwargs)
        self.Asky = Asky
        self.Nvisit = Nvisit
        self.nside = nside
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort()
        name = dataSlice.dtype.names[0]
        scale = hp.nside2pixarea(self.nside, degrees=True)
        cumulativeArea = np.arange(1,dataSlice.size+1)[::-1]*scale
        good = np.where(dataSlice[name] >= self.Nvisit)[0]
        if good.size > 0:
            area = np.max(cumulativeArea[good])
            if self.norm:
                area = area/float(self.Asky)
            return area
        else:
            return self.badval


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
    def __init__(self, col='metricdata',  nbins=10):
        """
        colname = the column name in the metric data (i.e. 'metricdata' usually).
        nbins = number of bins between 0 and 1. Should divide evenly into 100.
        """
        super(TableFractionMetric, self).__init__(col=col, metricDtype='float')
        self.nbins = nbins
        # set this so runSliceMetric knows masked values should be set to zero and passed
        self.maskVal = 0.

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
        for i in np.arange(0,self.nbins):
            binNames.append('%.2g < P < %.2g'%(b[i], b[i+1]) )
        binNames.append('1 == P')
        binNames.append('1 < P')
        # Package the names and values up
        result = np.empty(hist.size, dtype=[('name', '|S20'), ('value', float)])
        result['name'] = binNames
        result['value'] = hist
        return result


class IdentityMetric(BaseMetric):
    """Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics."""
    def run(self, dataSlice, slicePoint=None):
        if len(dataSlice[self.colname]) == 1:
            result = dataSlice[self.colname][0]
        else:
            result = dataSlice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions."""
    def __init__(self, col='metricdata', normVal=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.normVal = normVal
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname]/self.normVal
        if len(result) == 1:
            return result[0]
        else:
            return result
