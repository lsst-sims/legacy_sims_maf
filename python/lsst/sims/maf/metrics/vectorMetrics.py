import numpy as np
from .baseMetric import BaseMetric
from scipy import stats

__all__ = ['BaseVectorMetric','CountVMetric', 'CoaddM5VMetric']


# Create a base metric and set the default dtype to object
class BaseVectorMetric(BaseMetric):
    def __init__(self, metricDtype=float, mode='accumulate', **kwargs):
        """
        mode: Can be 'accumulate' or 'histogram'
        """
        self.mode = mode
        super(BaseVectorMetric,self).__init__(metricDtype=metricDtype,**kwargs)
    def run(self, dataSlice, slicePoint=None):
        raise NotImplementedError('Please implement your metric calculation.')

    def _pullAccumulateVals(self, result, timeValues, bins):

        indices = np.searchsorted(timeValues, bins, side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval

        return result


class CountVMetric(BaseVectorMetric):
    def __init__(self, col='night', units='Count',
                 statistic='count', mode='accumulate',**kwargs):
        self.statistic = statistic
        super(CountVMetric, self).__init__(col=col, units=units, mode=mode, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        toCount = np.ones(dataSlice.size,dtype=int)
        if self.mode == 'accumulate':
            result = np.add.accumulate(toCount)
            result = self._pullAccumulateVals(result, dataSlice[slicePoint['binCol']],
                                              slicePoint['bins'])
        elif self.mode == 'histogram':
            result, binEdges,binNumber = stats.binned_statistic(dataSlice[slicePoint['binCol']],
                                                                toCount,
                                                                bins=slicePoint['bins'],
                                                                statistic=self.statistic)
            # Need to append a dummy to make same length as bins
            result = np.append(result,0)
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
            result, binEdges,binNumber = stats.binned_statistic(dataSlice[slicePoint['binCol']],
                                                                flux, bins=slicePoint['bins'],
                                                                statistic='sum')
            # Need to append a dummy to make same length as bins
            result = np.append(result,0)
        else:
            raise ValueError('mode kwarg not set to "accumulate" or "histogram"')

        result = 1.25*np.log10(result)

        return result
