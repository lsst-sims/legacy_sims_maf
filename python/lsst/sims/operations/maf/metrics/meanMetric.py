# Example of a very basic metric. Calculates the Mean of the values at a gridpoint.
import numpy as np
from baseMetric import BaseMetric

class MeanMetric(BaseMetric):
    """Calculate the minimum of a simData column."""
    def __init__(self, colname, metricName = None):
        """Instantiate metric. 

        'colname' defines the single data column the metric will work on. """
        if hasattr(colname, '__iter__'):
            raise Exception('colname should be single column name: %s' %(colname))
        self.colname = colname   
        super(MeanMetric, self).__init__(self.colname, metricName)     
        return
            
    def run(self, dataSlice):
        return np.mean(dataSlice[self.colname])

