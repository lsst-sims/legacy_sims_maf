import numpy as np
from .simpleMetrics import SimpleScalarMetric
from .baseMetric import BaseMetric

class SummaryMetrics(BaseMetric):
    """A class for metrics which are intended to be primarily used as summary statistics on other metrics.  SimpleScalarMetrics can be used as well, but since they can return more than a scalar, they should not be placed with the SimpleMetrics."""
    def __init__(self, cols, *args,**kwargs):
        super(SummaryMetrics, self).__init__(cols,*args,**kwargs)

    def run(self, dataSlice):
        raise NotImplementedError()

    
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
        bins = np.arange(0,100/self.nbins+3,1)/float(self.nbins) # Use int step sizes to try and avoid floating point round-off errors.
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        hist[-1] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] > 1. )[0])
        hist[-2] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] == 1. )[0])
        hist[0] = np.size(np.where( (dataSlice[dataSlice.dtype.names[0]] > 0.) & (dataSlice[dataSlice.dtype.names[0]] < 0.1))[0] ) #clipping fields that were not observed
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
        bins = np.arange(0,12,1)/10. # Use int step sizes to try and avoid floating point round-off errors.
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        hist[-1] = np.size(np.where(dataSlice[dataSlice.dtype.names[0]] >= 1. )[0])
        hist[0] = np.size(np.where( (dataSlice[dataSlice.dtype.names[0]] > 0.) & (dataSlice[dataSlice.dtype.names[0]] < 0.1))[0] ) #clip off fields that were not observed, matching SSTAR table
        return hist


#class ExactCompleteMetric(SimpleScalarMetric):
#    """Calculate the fraction of fields that have exactly 100% of the requested visits. """
#    def run(self, dataSlice):    
#        good = np.where(dataSlice[dataSlice.dtype.names[0]] == 1.)
#        if float(np.size(dataSlice)) == 0:
#            result = self.badval
#        else:
#            result = np.size(good)/float(np.size(dataSlice))
#        return np.array(result)
    
    
