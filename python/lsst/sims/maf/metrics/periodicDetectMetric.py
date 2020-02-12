import numpy as np
from .baseMetric import BaseMetric
from lsst.sims.maf.utils import m52snr
import scipy

__all__ = ['PeriodicDetectMetric']


class PeriodicDetectMetric(BaseMetric):
    """Determine if we would be able to classify an object as periodic/non-uniform, using an F-test

    Parameters
    ----------

    period : float (2)
        The period of the star (days)
    amplitude : floar (0.1)
        The amplitude of the stellar variablility (mags).
    starMag : float (20.)
        The mean magnitude of the star (mags).
    sig_level : float (0.05)
        The value to use to compare to the p-value when deciding if we can reject the null hypothesis.

    Returns
    -------

    1 if we would detect star is variable, 0 if it is well-fit by a constant value.
    """
    def __init__(self, mjdCol='observationStartMJD', period=2., amplitude=0.1, m5Col='fiveSigmaDepth',
                 metricName='PeriodicDetectMetric', starMag=20, sig_level=0.05, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.period = period
        self.starMag = starMag
        self.amplitude = amplitude
        self.sig_level = sig_level

        super(PeriodicDetectMetric, self).__init__([mjdCol, m5Col], metricName=metricName,
                                                   units='Detected (1,0)', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        result = 0
        lc = self.amplitude*np.sin(dataSlice[self.mjdCol]*(np.pi*2)/self.period) + self.starMag
        n_pts = lc.size
        # If we had a correct model with phase, amplitude, period, zeropoint, then chi_squared/DoF would be ~1 with 4 free parameters.
        # The mean is one free parameter

        p1 = 1.
        p2 = 4.
        chi_sq_2 = 1.*(n_pts-p2)

        if n_pts > p2:
            snr = m52snr(lc, dataSlice[self.m5Col])
            delta_m = 2.5*np.log10(1.+1./snr)
            weights = 1./(delta_m**2)
            weighted_mean = np.sum(weights*lc)/np.sum(weights)
            chi_sq_1 = np.sum(((lc - weighted_mean)**2/delta_m**2))
            # Yes, I'm fitting magnitudes rather than flux. At least I feel kinda bad about it.
            # F-test for nested models:  https://en.wikipedia.org/wiki/F-test
            # https://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python/21503346
            f_numerator = (chi_sq_1 - chi_sq_2)/(p2-p1)
            f_denom = chi_sq_2/(n_pts-p2)
            f_val = f_numerator/f_denom

            # Has DoF (p2-p1, n-p2)
            p_value = scipy.stats.f.sf(f_val, p2-p1, n_pts-p2)
            if np.isfinite(p_value):
                if p_value < self.sig_level:
                    result = 1

        return result
