import numpy as np
from .baseMetric import BaseMetric

# Base class for simple metrics. 
class SimpleScalarMetric(BaseMetric):
    """This is the base class for the simplist metrics: ones that calculate one
       number on one column of data and return a scalar. 
    """
    def __init__(self, colname, *args, **kwargs):
        """Intantiate simple metric."""
        # Use base class init to register columns.
        super(SimpleScalarMetric, self).__init__(colname, *args, **kwargs)
        # Check incoming columns have only one value.
        if len(self.colNameList) > 1:
            raise Exception('Simple metrics should be passed single column. Got %s' %(colname))
        self.colname = self.colNameList[0]
        # Set return type.
        self.metricDtype = 'float'
        
        
    def run(self, dataSlice):
        raise NotImplementedError()


# Subclasses of simple metrics, that perform calculations at each gridpoint.

class Coaddm5Metric(SimpleScalarMetric):
    """Calculate the coadded m5 value at this gridpoint."""
    def __init__(self, m5col = '5sigma_modified', metricName = 'CoaddedM5', **kwargs):
        """Instantiate metric.
        
        m5col = the column name of the individual visit m5 data."""
        super(Coaddm5Metric, self).__init__(m5col, metricName=metricName, **kwargs)
    def run(self, dataSlice):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname]))) 


class MaxMetric(SimpleScalarMetric):
    """Calculate the maximum of a simData column slice."""
    def run(self, dataSlice):
        return np.max(dataSlice[self.colname]) 


class MeanMetric(SimpleScalarMetric):
    """Calculate the mean of a simData column slice."""
    def run(self, dataSlice):
        return np.mean(dataSlice[self.colname]) 

class MedianMetric(SimpleScalarMetric):
    """Calculate the median of a simData column slice."""
    def run(self, dataSlice):
        return np.median(dataSlice[self.colname])

    
class MinMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice):
        return np.min(dataSlice[self.colname]) 


class RmsMetric(SimpleScalarMetric):
    """Calculate the standard deviation of a simData column slice."""
    def run(self, dataSlice):
        return np.std(dataSlice[self.colname]) 


class SumMetric(SimpleScalarMetric):
    """Calculate the sum of a simData column slice."""
    def run(self, dataSlice):
        return np.sum(dataSlice[self.colname]) 

class CountMetric(SimpleScalarMetric):
    """Count the length of a simData column slice. """
    def run(self, dataSlice):
        return len(dataSlice[self.colname]) 

class CountRatioMetric(SimpleScalarMetric):
    """Count the length of a simData column slice and normalize by some value. """
    def __init__(self, col='expMJD', normVal=10., metricName = 'Ratio', **kwargs):
        """Instantiate metric.

        normVal = value to by which to divide the counts.
         """
        self.normVal=normVal
        super(CountRatioMetric, self).__init__(colname=col, metricName=metricName, **kwargs)
        self.units ='Ratio'
    def run(self, dataSlice):
        return len(dataSlice[self.colname])/self.normVal
