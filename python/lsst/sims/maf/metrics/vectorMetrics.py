import numpy as np
from .baseMetric import BaseMetric
from scipy import stats

__all__ = ['BaseVectorMetric','CountVMetric', 'CoaddM5VMetric']


# Create a base metric and set the default dtype to object
class BaseVectorMetric(BaseMetric):
    def __init__(self, metricDtype=object, mode='accumulate', **kwargs):
        """
        mode: Can be 'accumulate' or 'histogram'
        """
        self.mode = mode
        super(BaseVectorMetric,self).__init__(metricDtype=metricDtype,**kwargs)
    def run(self, dataSlice, slicePoint=None):
        raise NotImplementedError('Please implement your metric calculation.')

    def _pullAccumulateVals(self, result, timeValues, bins):

        indices = np.searchsorted(timeValues, bins, side='left')
        result[np.where(indices == 0)] = self.badval
        return result[indices]


class CountVMetric(BaseVectorMetric):
    def __init__(self, col='night', units='Count',
                 statistic='count', **kwargs):
        self.statistic = statistic
        super(CountVMetric, self).__init__(col=col, units=units, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        toCount = np.ones(dataSlice.size,dtype=int)
        if self.mode == 'accumulate':
            result = np.add.accumulate(toCount)
            result = self._pullAccumulateVals(result, dataSlice[slicePoint['binCol']],
                                              slicePoint['bins'])
        elif self.mode == 'histogram':
            result, binEdges,binNumber = stats.binned_statistic(toCount, bins,
                                                                statistic=self.statistic)
        else:
            raise ValueError('mode kwarg not set to "accumulate" or "histogram"')

        return result

class CoaddM5VMetric(BaseVectorMetric):
    def __init__(self, col='night', m5Col = 'fiveSigmaDepth', metricName='CoaddM5',
                 units='mags',**kwargs):
        self.statistic = statistic
        self.m5Col = m5Col
        super(CoaddM5VMetric, self).__init__(col=[col,m5Col],  metricName=metricName,
                                             units=units, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        flux = 10.**(.8*dataSlice[self.m5Col])

        if self.mode == 'accumulate':
            result = np.add.accumulate(flux)
            result = _pullAccumulateVals(result, dataSlice[slicePoint['binCol']],
                                         slicePoint['bins'])
        elif self.mode == 'histogram':
            result, binEdges,binNumber = stats.binned_statistic(flux, bins,
                                                                statistic='sum')
        else:
            raise ValueError('mode kwarg not set to "accumulate" or "histogram"')

        result = 1.25*np.log10(result)

        return result
