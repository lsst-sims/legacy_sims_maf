import numpy as np
from .simpleMetrics import SimpleScalarMetric
from .baseMetric import BaseMetric
import healpy as hp

# A collection of metrics which are primarily intended to be used as summary statistics.
    
class f0Area(BaseMetric):
    def __init__(self, cols, Asky=18000., Nvisit=825, 
                 metricName='f0Area', nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(f0Area, self).__init__(cols,metricName=metricName,**kwargs)
        self.Asky = Asky
        self.Nvisit = Nvisit
        self.nside = nside
        self.norm = norm

    def run(self, dataSlice):
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
        

class f0Nv(BaseMetric):
    def __init__(self, cols, Asky=18000., metricName='f0Nv', Nvisit=825, 
                 nside=128, norm=True, **kwargs):
        """Asky = square degrees """
        super(f0Nv, self).__init__(cols,metricName=metricName,**kwargs)
        self.Asky = Asky
        self.Nvisit = Nvisit
        self.nside = nside
        self.norm = norm

    def run(self, dataSlice):
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
    

class TableFractionMetric(SimpleScalarMetric):
    def __init__(self, colname, nbins=10):
        """nbins = number of bins between 0 and 100.  100 must be evenly divisable by nbins. """
        super(SimpleScalarMetric, self).__init__(colname)
        self.nbins=nbins
        """This metric is meant to be used as a summary statistic on something like the completeness metric.
        The output is DIFFERENT FROM SSTAR and is:
        element   matching values
        0         0 == P
        1         0 < P < 10
        2         10 <= P < 20
        3         20 <= P < 30
        ...
        10        90 <= P < 100
        11        100 == P
        12        100 < P
        Note the 1st and last elements do NOT obey the numpy histogram conventions."""

    def run(self, dataSlice):    
        # Use int step sizes to try and avoid floating point round-off errors.
        bins = np.arange(0,100/self.nbins+3,1)/float(self.nbins) 
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        hist[-1] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] > 1. )[0])
        hist[-2] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] == 1. )[0])
        # clipping fields that were not observed        
        hist[0] = np.size(np.where((dataSlice[dataSlice.dtype.names[0]] > 0.) & (dataSlice[dataSlice.dtype.names[0]] < 0.1))[0]) 
        exact_zero = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] == 0. )[0])
        hist = np.concatenate((np.array([exact_zero]),hist))
        return hist

class SSTARTableFractionMetric(SimpleScalarMetric):
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
    def run(self, dataSlice):    
        # Use int step sizes to try and avoid floating point round-off errors.
        bins = np.arange(0,12,1)/10. 
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        hist[-1] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] >= 1. )[0])
        # clip off fields that were not observed, matching SSTAR table
        hist[0] = np.size(np.where( (dataSlice[dataSlice.dtype.names[0]] > 0.) & (dataSlice[dataSlice.dtype.names[0]] < 0.1))[0] ) 
        return hist



class IdentityMetric(SimpleScalarMetric):
    """Return the metric value itself .. this is primarily useful as a summary statistic for UniBinner metrics."""
    def run(self, dataSlice):
        return dataSlice[self.colname]


class NormalizeMetric(SimpleScalarMetric):
    """Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions."""
    def __init__(self, colname, normVal=1):
        super(NormalizeMetric, self).__init__(colname)
        self.normVal = normVal
    def run(self, dataSlice):
        return dataSlice[self.colname]/self.normVal
