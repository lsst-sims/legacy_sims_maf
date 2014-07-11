import numpy as np
from .baseMetric import BaseMetric

# SIMPLE SCALAR METRICS are just that - they are metrics that operate on a single column of simData
#  and return a SINGLE SCALAR value.

# Base class for simple metrics. 
class SimpleScalarMetric(BaseMetric):
    """This is the base class for the simplist metrics: ones that calculate one
       number on one column of data and return a scalar. 
    """
    def __init__(self, colname, **kwargs):
        """Intantiate simple metric."""
        # Use base class init to register columns.
        super(SimpleScalarMetric, self).__init__(colname, metricDtype='float', **kwargs)
        # Check incoming columns have only one value.
        # (colNameArr is numpy array version of colname, set by BaseMetric).
        if len(self.colNameArr) > 1:
            raise Exception('Simple metrics should be passed single column. Got %s' %(colname))
        self.colname = self.colNameArr[0]
        
    def run(self, dataSlice, slicePoint=None):
        raise NotImplementedError()


# Subclasses of simple metrics, that perform calculations at each gridpoint.

class Coaddm5Metric(SimpleScalarMetric):
    """Calculate the coadded m5 value at this gridpoint."""
    def __init__(self, m5col = 'fivesigma_modified', metricName = 'CoaddedM5', **kwargs):
        """Instantiate metric.
        
        m5col = the column name of the individual visit m5 data."""
        super(Coaddm5Metric, self).__init__(m5col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname]))) 

        
class MaxMetric(SimpleScalarMetric):
    """Calculate the maximum of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.max(dataSlice[self.colname]) 


class MeanMetric(SimpleScalarMetric):
    """Calculate the mean of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.mean(dataSlice[self.colname]) 

class MedianMetric(SimpleScalarMetric):
    """Calculate the median of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.median(dataSlice[self.colname])

    
class MinMetric(SimpleScalarMetric):
    """Calculate the minimum of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.min(dataSlice[self.colname]) 

class FullRangeMetric(SimpleScalarMetric):
    """Calculate the range of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.max(dataSlice[self.colname])-np.min(dataSlice[self.colname]) 

class RmsMetric(SimpleScalarMetric):
    """Calculate the standard deviation of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.std(dataSlice[self.colname]) 

class SumMetric(SimpleScalarMetric):
    """Calculate the sum of a simData column slice."""
    def run(self, dataSlice, slicePoint=None):
        return np.sum(dataSlice[self.colname]) 

class CountMetric(SimpleScalarMetric):
    """Count the length of a simData column slice. """
    def run(self, dataSlice, slicePoint=None):
        return len(dataSlice[self.colname]) 

class RobustRmsMetric(SimpleScalarMetric):
    """Use the inter-quartile range of the data to estimate the RMS.  Robust since this calculation
    does not include outliers in the distribution"""
    def run(self, dataSlice, slicePoint=None):
        iqr = np.percentile(dataSlice[self.colname],75)-np.percentile(dataSlice[self.colname],25)
        rms = iqr/1.349 #approximation
        return rms
    
class BinaryMetric(SimpleScalarMetric):
    """Return 1 if there is data. """
    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size > 0:
            return 1
        else:
            return self.badval

class FracAboveMetric(SimpleScalarMetric):
    def __init__(self, colname, cutoff=0.5, **kwargs):
        super(FracAboveMetric, self).__init__(colname, **kwargs)
        self.cutoff = cutoff
    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.colname] >= self.cutoff)[0]
        fracAbove = np.size(good)/float(np.size(dataSlice[self.colname]))
        return fracAbove

class FracBelowMetric(SimpleScalarMetric):
    def __init__(self, colname, cutoff=0.5, **kwargs):
        super(FracBelowMetric, self).__init__(colname, **kwargs)
        self.cutoff = cutoff
    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.colname] <= self.cutoff)[0]
        fracBelow = np.size(good)/float(np.size(dataSlice[self.colname]))
        return fracBelow
