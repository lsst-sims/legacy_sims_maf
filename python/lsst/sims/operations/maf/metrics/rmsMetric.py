# Example of a very basic metric. Calculates the standard deviation of the values at a gridpoint.
import numpy as np
from baseMetric import BaseMetric

class RmsMetric(BaseMetric):
    """Calculate the minimum of a simData column."""
    def __init__(self, colname):
        """Instantiate metric. 

        'colname' defines the single data column the metric will work on. """
        if hasattr(colname, '__iter__'):
            raise Exception('colname should be single column name: %s' %(colname))
        self.colname = colname   
        super(RmsMetric, self).__init__(self.colname)     
        return
            
    def run(self, dataSlice):
        return np.std(dataSlice[self.colname])
