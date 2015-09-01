import numpy as np
from .baseMetric import BaseMetric
from scipy import stats

__all__ = []


# Create a base metric and set the default dtype to object
class BaseVectorMetric(BaseMetric):
    def __init__(self, metricDtype=object, mode='accumulate', **kwargs):
        """
        mode: Can be 'accumulate' or 'histogram'
        """
        self.mode = mode
        super(BaseVectorMetric,self).__init__(metricDtype,**kwargs)
    def run(self, dataSlice, slicePoint=None):
        raise NotImplementedError('Please implement your metric calculation.')


def _pullAccumulateVals(result, timeValues, bins):

    indices = np.searchsorted(timeVals, bins, side='left')
    return result[indices]


class CountVMetric(BaseVectorMetric):
    def __init__(self, col='night', units='Count',mode='accumulate',**kwargs):
        super(CountVMetric, self).__init__(col=col, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        toCount = np.ones(dataSlice.size,dtype=int)
        if self.mode == 'accumulate':
            result = np.add.accumulate(toCount)
            result = _pullAccumulateVals(result, dataSlice[slicePoint['binCol']],
                                         slicePoint['bins'])
        elif self.mode == 'histogram':
            result, binEdges = np.histogram(toCount, bins)
        else:
            raise ValueError('mode kwarg not set to "accumulate" or "histogram"')

        return result
