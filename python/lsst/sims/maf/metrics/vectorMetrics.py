import numpy as np
from .baseMetric import BaseMetric
from scipy import stats

__all__ = ['HistogramMetric','AccumulateMetric', 'AccumulateCountMetric',
           'HistogramM5Metric', 'AccumulateM5Metric']


class HistogramMetric(BaseMetric):
    """
    A wrapper to stats.binned_statistic
    """
    def __init__(self, col='night', units='Count', statistic='count',
                 metricDtype=float, **kwargs):
        self.statistic = statistic
        super(BaseVectorMetric,self).__init__(col=col, units=units,
                                              metricDtype=metricDtype,**kwargs)

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        result, binEdges,binNumber = stats.binned_statistic(dataSlice[slicePoint['binCol']],
                                                            self.col,
                                                            bins=slicePoint['bins'],
                                                            statistic=self.statistic)
        # Make the result the same length as bins
        result = np.append(result,0)
        return result

class AccumulateMetric(BaseMetric):
    """
    Calculate the accumulated stat
    """
    def __init__(self, col='night', function=np.add,
                 metricDtype=float, **kwargs):
        self.function = function
        super(BaseVectorMetric,self).__init__(col=col, units=units,
                                              metricDtype=metricDtype,**kwargs)

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])

        result = self.function.accumulate(dataSlice[self.col])
        indices = np.searchsorted(dataSlice[slicePoint['binCol']], bins, side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval

        return result

class AccumulateCountMetric(AccumulateMetric):
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        toCount = np.ones(dataSlice.size, dtype=int)
        result = self.function.accumulate(toCount)
        indices = np.searchsorted(dataSlice[slicePoint['binCol']], bins, side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval

        return result

class HistogramM5Metric(HistogramMetric):
    pass

class AccumulateM5Metric(AccumulateMetric):
    pass


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
