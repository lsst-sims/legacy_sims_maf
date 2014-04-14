import numpy as np
from .simpleMetrics import SimpleScalarMetric

class TableFractionMetric(SimpleScalarMetric):
    # Using SimpleScalarMetric, but returning a histogram.
    """This metric is meant to be used as a summary statistic on something like the completeness metric.  """
    def run(self, dataSlice):    
        bins = np.arange(0,1.1,.1)
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        return hist
    

class ExactCompleteMetric(SimpleScalarMetric):
    """Calculate the fraction of fields that have exactly 100% of the requested visits. """
    def run(self, dataSlice):    
        good = np.where(dataSlice[dataSlice.dtype.names[0]] == 1.)
        if float(np.size(dataSlice)) == 0:
            result = self.badval
        else:
            result = np.size(good)/float(np.size(dataSlice))
        return np.array(result)
    
    
