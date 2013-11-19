import numpy as np
from .baseMetric import BaseMetric

# Base class for simple metrics. 
class SimpleScalarMetric(BaseMetric):
    """This is the base class for the simplist metrics: ones that calculate one
       number on one column of data and return a scalar. 
    """
    def __init__(self, colname, metricName):
        """Intantiate metric.
        """
        super(SimpleScalarMetric, self).__init__(colname, metricName)
        if len(self.colNameList) > 1:
            raise Exception('m5col should be single column name: %s' %(m5col))
        self.colname = self.colNameList[0]
    def run(self, dataSlice):
        raise NotImplementedError()


# Subclasses of simple metrics, that perform calculations at each gridpoint.

class Coaddm5Metric(SimpleScalarMetric):
    """Calculate the coadded m5 value at this gridpoint."""
    def __init__(self, m5col = '5sigma_modified', metricName = 'coaddm5'):
        """Instantiate metric.
        m5col should be the column name of the individual visit m5 data. """
        super(Coaddm5Metric, self).__init__(m5col, metricName)            
    def run(self, dataSlice):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname])))


class MaxMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.max(dataSlice[self.colname])


class MeanMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.mean(dataSlice[self.colname])


class MinMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.min(dataSlice[self.colname])


class RmsMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.std(dataSlice[self.colname])


class SumMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.sum(dataSlice[self.colname])

class CountMetric(SimpleScalarMetric):
    """Count the length of a simData column slice. """
    def run(self, dataSlice):
        return len(dataSlice[self.colname])a


