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
        self.col=col
        super(HistogramMetric,self).__init__(col=col, units=units,
                                              metricDtype=metricDtype,**kwargs)

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        result, binEdges,binNumber = stats.binned_statistic(dataSlice[slicePoint['binCol']],
                                                            dataSlice[self.col],
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
        self.col=col
        super(AccumulateMetric,self).__init__(col=col,
                                              metricDtype=metricDtype,**kwargs)

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])

        result = self.function.accumulate(dataSlice[self.col])
        indices = np.searchsorted(dataSlice[slicePoint['binCol']], slicePoint['bins'], side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval

        return result

class AccumulateCountMetric(AccumulateMetric):
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        toCount = np.ones(dataSlice.size, dtype=int)
        result = self.function.accumulate(toCount)
        indices = np.searchsorted(dataSlice[slicePoint['binCol']], slicePoint['bins'], side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result[np.where(indices == 0)] = self.badval

        return result

class HistogramM5Metric(HistogramMetric):
    """
    Calculate the coadded depth for each bin (e.g., per night).
    """
    def __init_(self, col='night', m5Col='fiveSigmaDepth', units='mag',
                metricName='HistogramM5Metric',**kwargs):
        self.m5Col=m5Col
        self.col = col
        super(HistogramM5Metric,self).__init__(col=[col,m5Col],
                                               metricName=metricName,
                                               units=units**kwargs)
    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        flux = 10.**(.8*dataSlice[self.m5Col])
        result, binEdges,binNumber = stats.binned_statistic(dataSlice[self.col],
                                                            flux,
                                                            bins=slicePoint['bins'],
                                                            statistic='sum')
        result = 1.25*np.log10(result)
        # Make the result the same length as bins
        result = np.append(result,self.badval)
        return result



class AccumulateM5Metric(AccumulateMetric):
    def __init__(self, col='night', m5Col='fiveSigmaDepth',
                metricName='AccumulateM5Metric',**kwargs):
        self.m5Col = m5Col
        super(AccumulateM5Metric,self).__init__(col=[col,m5Col], metricName=metricName,**kwargs)


    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=slicePoint['binCol'])
        flux = 10.**(.8*dataSlice[self.m5Col])

        result = np.add.accumulate(flux)
        indices = np.searchsorted(dataSlice[slicePoint['binCol']], slicePoint['bins'], side='left')
        indices[np.where(indices >= np.size(result))] = np.size(result)-1
        result = result[indices]
        result = 1.25*np.log10(result)
        result[np.where(indices == 0)] = self.badval

        return result
