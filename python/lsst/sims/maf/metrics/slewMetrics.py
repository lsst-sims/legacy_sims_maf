import numpy as np
from .baseMetric import BaseMetric

# Metrics for dealing with things from the SlewActivities table

class ActivePercentMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity', **kwargs):
        col = [col]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(ActivePercentMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = good.size/float(dataSlice.size)*100.
        return result


class ActiveAveMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity', **kwargs):
        self.col = col
        self.activity = activity
        col = [col]
        col.append(activeCol)
        self.activeCol = activeCol
        super(ActiveAveMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = np.mean(dataSlice[self.col][good])
        return result


