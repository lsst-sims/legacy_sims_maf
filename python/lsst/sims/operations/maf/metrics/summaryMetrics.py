import numpy as np
from .simpleMetrics import SimpleScalarMetric


class SummaryMetrics(BaseMetric):
    """A class for metrics which are intended to be primarily used as summary statistics on other metrics.  SimpleScalarMetrics can be used as well, but since they can return more than a scalar, they should not be placed with the SimpleMetrics."""
    def __init__(self, cols, *args,**kwargs):
        super(SummaryMetrics, self).__init__(cols,*args,**kwargs)

    def run(self, dataSlice):
        raise NotImplementedError()


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
    
    
