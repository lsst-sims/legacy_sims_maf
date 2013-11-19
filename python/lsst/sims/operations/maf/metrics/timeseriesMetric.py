# Example of a more complicated metric. 

import numpy as np
from scipy import fftpack
from baseMetric import BaseMetric

class FftMetric(BaseMetric):
    """Calculate the minimum of a simData column."""
    def __init__(self, expmjd='expmjd', metricName=None):
        """Instantiate metric.

        'expmjd' defines the column with the time of the visit."""
        self.expmjd = expmjd   
        super(FftMetric, self).__init__(self.expmjd, metricName)     
        # Update length of metric as we will save the first 10 FFT coefficients.
        self.metricLen = 10.0
        return
            
    def run(self, dataSlice):
        visitTimes = dataSlice[self.expmjd]
        
        return 
