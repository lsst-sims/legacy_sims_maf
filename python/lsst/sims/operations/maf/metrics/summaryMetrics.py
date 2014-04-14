import numpy as np
from .simpleMetrics import SimpleScalarMetric

class TableFractionMetric(SimpleScalarMetric):
    # Using SimpleScalarMetric, but returning a histogram.
    """This metric is meant to be used as a summary statistic on something like the completeness metric.  """
    def run(self, dataSlice):    #
        bins = np.arange(0,1.1,.1)
        hist, binEdges = np.histogram(dataSlice[dataSlice.dtype.names[0]], bins=bins)
        return hist
    
