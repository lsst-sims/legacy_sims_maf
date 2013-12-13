import numpy as np
from scipy import fftpack
from .complexMetrics import ComplexMetric

# Example class for multi-value (constant length vector) metrics. 
# Based on complexMetrics because want to use 'reduce' function dictionary implemented there.

class FftMetric(ComplexMetric):
    """Calculate a truncated FFT of the exposure times."""
    def __init__(self, timesCol='expmjd', metricName='FftMetric',
                 nCoeffs=100):
        """Instantiate metric.
        
        'timesCol' = column with the time of the visit (default expmjd), 
        'nCoeffs' = number of coefficients of the (real) FFT to keep."""
        self.times = timesCol   
        super(FftMetric, self).__init__([self.times,], metricName)
        # Set up length of return values.
        self.nCoeffs = nCoeffs
        return

    def run(self, dataSlice):
        fft = fftpack.rfft(dataSlice[self.times])
        return fft[0:self.nCoeffs]

    def reducePeak(self, fftCoeff):
        raise NotImplementedError()
        
