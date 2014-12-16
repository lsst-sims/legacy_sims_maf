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


class ContributionMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity',
                 inCritCol='inCriticalPath', **kwargs):
        """Return the contribution of an activity to the average slew time """
        self.col = col
        self.inCritCol = inCritCol
        col = [col, inCritCol]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(ContributionMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        goodInCrit = np.where( (dataSlice[self.activeCol] == self.activity) &
                               (dataSlice[self.inCritCol] == 'True'))[0]
        inCrit = np.where( (dataSlice[self.inCritCol] == 'True'))[0]

        result = np.sum(dataSlice[self.col][goodInCrit])/np.sum(dataSlice[self.col][inCrit])* \
                 np.mean(dataSlice[self.col][goodInCrit])
        return result

class AveSlewFracMetric(BaseMetric):
    def __init__(self, col=None, activity=None, activeCol='activity',
                 idCol='SlewHistory_slewID', **kwargs):
        """Return the average time multiplied by fraction of slews """
        self.col = col
        self.idCol = idCol
        col = [col, idCol]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(AveSlewFracMetric, self).__init__(col=col, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        result = np.mean(dataSlice[self.col][good])
        nslews = np.size(np.unique(dataSlice[self.idCol]))
        result = result * np.size(good)/np.float(nslews)
        return result
