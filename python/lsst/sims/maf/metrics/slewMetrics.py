import numpy as np
from .baseMetric import BaseMetric

# Metrics for dealing with things from the SlewActivities table

class ActivePercentMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity',
                 norm=1., **kwargs):
        """Return the Count multiplied by some norm """
        col = [col]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        self.norm=norm
        super(ActivePercentMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = good.size*self.norm
        return result


class ActiveMeanMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity',
                 norm=1., **kwargs):
        """Return the Mean multiplied by some norm """
        self.col = col
        col = [col]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        self.norm=norm
        super(ActiveMeanMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = np.mean(dataSlice[self.col][good])*self.norm
        return result

class ActiveMaxMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity',
                 **kwargs):
        """Return the Max """
        self.col = col
        col = [col]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(ActiveMaxMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = np.max(dataSlice[self.col][good])
        return result
