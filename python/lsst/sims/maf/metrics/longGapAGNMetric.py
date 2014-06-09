import numpy as np
from .baseMetric import BaseMetric

class LongGapAGNMetric(BaseMetric):
    """max delta-t and average of the top-10 longest gaps. r and i filters separately.
    """

    def __init__(self, metricName='longGapAGNMetric',
                 mjdcol='expMJD', units='days', xgaps=10, **kwargs):
        """ Instantiate metric.
        mjdcol = column name for exposure time dates
        """
        cols = [mjdcol]
        super(LongGapAGNMetric, self).__init__(cols, metricName, units=units, **kwargs)
        # set return type
        self.metricDtype = 'float'
	self.xgaps = xgaps
        self.units = units

    def run(self, dataslice):
	metricval = np.diff(dataslice)
        return metricval
    
    def reduceMaxGap(self, metricval):
	return np.max(metricval)

    def reduceAverageLongestXGaps(self, metricval):
	return np.average(np.sort(metricval)[np.size(metricval)-self.xgaps:])
