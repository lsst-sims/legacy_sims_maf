import numpy as np
from .baseMetric import BaseMetric

__all__ = ['CoverageMetric']


class CoverageMetric(BaseMetric):
    """Metric that checks how many unique years a spot has been observed. Handy for checking that things get observed every year.
    """

    def __init__(self, nightCol='night', bins=None, metricName='CoverageMetric', units=None, **kwargs):
        self.nightCol = nightCol
        if bins is None:
            self.bins = np.arange(0, np.ceil(365.25*10.), 365.25) - 0.5
        else:
            self.bins = bins

        if units is None:
            units = 'N years'

        super(CoverageMetric, self).__init__([nightCol], metricName=metricName, units=units)

    def run(self, dataSlice, slicePoint):
        hist, be = np.histogram(dataSlice[self.nightCol], bins=self.bins)
        result = np.where(hist > 0)[0].size
        return result
