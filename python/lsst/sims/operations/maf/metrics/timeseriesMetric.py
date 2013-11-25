# Example of a more complicated metric. 

import numpy as np
from scipy import fftpack
from .baseMetric import BaseMetric

class FftMetric(BaseMetric):
    """Calculate the fft transform of a set of exposure times."""
    def __init__(self, timesCol='expmjd', metricName='FftVisits'):
        """Instantiate metric.

        'expmjd' defines the column with the time of the visit."""
        self.times = timesCol   
        super(FftMetric, self).__init__(self.times, metricName)     
        self.reduceFuncs = {'Peak': self.reducePeak}
        return
            
    def run(self, dataSlice):
        dtimes = dataSlice[self.expmjd] - dataSlice[self.expmjd][0]
        fft = fftpack.rfft(dtimes)
        return fft[0:10]

    def reducePeak(self, fftValues):
        """Reduce fft coefficients by finding time of second highest peak. """
        pass
        

    
