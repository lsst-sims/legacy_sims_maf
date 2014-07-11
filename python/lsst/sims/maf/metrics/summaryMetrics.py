import numpy as np
from .simpleMetrics import SimpleScalarMetric
from .baseMetric import BaseMetric
import healpy as hp

# A collection of metrics which are primarily intended to be used as summary statistics.
    
class fOArea(BaseMetric):
    """Metric to calculate the FO Area; works with FO slicer only."""
    def __init__(self, cols, Asky=18000., Nvisit=825, 
                 metricName='fOArea', nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(fOArea, self).__init__(cols,metricName=metricName,**kwargs)
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
    """Metric to calculate the FO_Nv; works with FO slicer only."""
    def __init__(self, cols, Asky=18000., metricName='fONv', Nvisit=825, 
                 nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(fONv, self).__init__(cols,metricName=metricName,**kwargs)
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
    (within a series of bins).

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
    def __init__(self, colname, nbins=10):
        """
        colname = the column name in the metric data (i.e. 'metricdata' usually).
        nbins = number of bins between 0 and 1. Should divide evenly into 100.  
        """
        super(TableFractionMetric, self).__init__(colname)
        self.colname = colname
        binsize = 1.0/float(nbins)
        self.tableBins = np.arange(0, 1 + binsize/2., binsize)
        self.tableBins = np.concatenate((np.zeros(1, float), self.tableBins))
        self.tableBins = np.concatenate((self.tableBins, np.ones(1, float)))
        self.tableBins[-1:] = 1.01
        
    def run(self, dataSlice, slicePoint=None):
        # Calculate histogram of completeness values that fall between 0-1.
        print self.tableBins[1:-2], len(self.tableBins[1:-2])
        hist, b = np.histogram(dataSlice[self.colname], bins=self.tableBins[1:-2])
        print hist, len(hist)
        # Fill in values for exact 0, exact 1 and >1.
        zero = np.size(np.where(dataSlice[self.colname] == 0)[0])
        # Remove the fields which were exactly 0 from the histogrammed values.
        hist[0] -= zero      
        one = np.size(np.where(dataSlice[self.colname] == 1)[0])
        overone = np.size(np.where(dataSlice[self.colname] > 1)[0])
        hist = np.concatenate((np.array([zero]), hist, np.array([one]), np.array([overone])))
        return self.tableBins, hist

class SSTARTableFractionMetric(BaseMetric):
    # Using SimpleScalarMetric, but returning a histogram.
    """This metric is meant to be used as a summary statistic on something like the completeness metric.
    
    This table matches the SSTAR table of the format:
    element   matching values
    0         0 < P < 10
    1         10 <= P < 20
    2         20 <= P < 30
    ...
    9         90 <= P < 100
    10        100 <= P
    Note the 1st and last elements do NOT obey the numpy histogram conventions."""
    def run(self, dataSlice, slicePoint=None):    
        # Use int step sizes to try and avoid floating point round-off errors.
        bins = np.arange(0,12,1)/10. 
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        hist[-1] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] >= 1. )[0])
        # clip off fields that were not observed, matching SSTAR table
        hist[0] = np.size(np.where( (dataSlice[dataSlice.dtype.names[0]] > 0.) &
                                    (dataSlice[dataSlice.dtype.names[0]] < 0.1))[0] ) 
        return hist


class IdentityMetric(SimpleScalarMetric):
    """Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics."""
    def run(self, dataSlice, slicePoint=None):
        return dataSlice[self.colname]


class NormalizeMetric(SimpleScalarMetric):
    """Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions."""
    def __init__(self, colname, normVal=1):
        super(NormalizeMetric, self).__init__(colname)
        self.normVal = normVal
    def run(self, dataSlice, slicePoint=None):
        return dataSlice[self.colname]/self.normVal
