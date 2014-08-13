import numpy as np
from .baseMetric import BaseMetric

class Tgaps(BaseMetric):
    """Histogram up all the time gaps """

    def __init__(self, timesCol='expMJD', binMin=0,
                 binMax=5000., binsize=120., **kwargs):
        """  """
        self.timesCol = timesCol
        super(Tgaps, self).__init__(col=[self.timesCol],
                                    metricDtype=object, **kwargs)
        self.bins=np.arange(binMin, binMax+binsize,binsize)

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        dts = (dataSlice[self.timesCol] -
               np.roll(dataSlice[self.timesCol], 1))[1:]
        # Convert to seconds
        dts = dts*24.*3600. 
        # do we want an option that unrolls and does all possible lags?
        result, bins = np.histogram(dts, self.bins)
        return result
        
