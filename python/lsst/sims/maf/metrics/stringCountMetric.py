import numpy as np
from .baseMetric import BaseMetric
from collections import Counter

__all__ = ['StringCountMetric']


class keylookerupper(object):
    """My god this is convoluted
    """
    def __init__(self, key='blank', name=None):
        self.key = key
        self.__name__ = name

    def __call__(self, indict):
        return np.max(indict[self.key])


class StringCountMetric(BaseMetric):
    """Count up the number of times each string appears in a column
    """

    def __init__(self, metricName='stringCountMetric',
                 col='filter', percent=False, **kwargs):
        """ Instantiate metric.
        mjdcol = column name for exposure time dates
        """
        if percent:
            units = 'percent'
        else:
            units = 'count'
        self.percent = percent
        cols = [col]
        super(StringCountMetric, self).__init__(cols, metricName, units=units,
                                                metricDtype=object, **kwargs)
        self.col = col

    def run(self, dataslice, slicePoint=None):
        counter = Counter(dataslice[self.col])
        # convert to a numpy array
        lables = list(counter.keys())
        # Numpy can't handle empty string as a dtype
        lables = [x if x != '' else 'blank' for x in lables]
        metricValue = np.zeros(1, dtype=list(zip(lables, [float]*len(counter.keys()))))
        for key in counter:
            if key == '':
                metricValue['blank'] = counter[key]
            else:
                metricValue[key] = counter[key]
        if self.percent:
            norm = sum(metricValue[0])/100.
            # Not sure I really like having to loop here, but the dtype is inflexible
            for key in metricValue.dtype.names:
                metricValue[key] = metricValue[key]/norm

        # Now to dynamically set up the reduce functions
        for i, key in enumerate(metricValue.dtype.names):
            name = key
            self.reduceFuncs[name] = keylookerupper(key=key, name=name)
            self.reduceOrder[name] = i

        return metricValue
