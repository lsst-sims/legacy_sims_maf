import numpy as np
from .baseMetric import BaseMetric
from lsst.sims.maf.utils.astrometryUtils import sigma_slope, m52snr,astrom_precision

class ProperMotionMetric(BaseMetric):
    """Calculate the uncertainty in the returned proper
    motion.  Assuming Gaussian errors.
    """

    def __init__(self, metricName='properMotion',
                 m5col='5sigma_modified', mjdcol='expMJD', units='mas/yr',
                 filtercol='filter', seeingcol='finSeeing', u=20.,
                 g=20., r=20., i=20., z=20., y=20., badval= -666,
                 stellarType=None, atm_err=0.01, normalize=False,
                 baseline=10., **kwargs):
        """ Instantiate metric.

        m5col = column name for inidivual visit m5
        mjdcol = column name for exposure time dates
        filtercol = column name for filter
        seeingcol = column name for seeing (assumed FWHM)
        u,g,r,i,z = mag of fiducial star in each filter
        atm_err = centroiding error due to atmosphere in arcsec
        normalize = Compare to the uncertainty that would result if half
        the observations were taken at the start of the survey and half
        at the end.  A 'perfect' survey will have a value close to unity,
        while a poorly scheduled survey will be close to zero.
        baseline = The length of the survey used for the normalization (years)
        """
        cols = [m5col, mjdcol,filtercol,seeingcol]
        if normalize:
            units = 'ratio'
        super(ProperMotionMetric, self).__init__(cols, metricName, units=units, **kwargs)
        # set return type
        self.seeingcol = seeingcol
        self.m5col = m5col
        self.metricDtype = 'float'
        self.units = units
        self.mags={'u':u, 'g':g,'r':r,'i':i,'z':z,'y':y}
        self.badval = badval
        self.atm_err = atm_err
        self.normalize = normalize
        self.baseline = baseline
        if stellarType != None:
            raise NotImplementedError('Spot to add colors for different stars')

    def run(self, dataslice):
        filters = np.unique(dataslice['filter'])
        precis = np.zeros(dataslice.size, dtype='float')
        for f in filters:
            observations = np.where(dataslice['filter'] == f)
            if np.size(observations[0]) < 2:
                precis[observations] = self.badval
            else:
                snr = m52snr(self.mags[f],
                   dataslice[self.m5col][observations])
                precis[observations] = astrom_precision(
                    dataslice[self.seeingcol][observations], snr)
                precis[observations] = np.sqrt(precis[observations]**2 + self.atm_err**2)
        good = np.where(precis != self.badval)
        result = sigma_slope(dataslice['expMJD'][good], precis[good])
        result = result*365.25*1e3 #convert to mas/yr
        if (self.normalize) & (good[0].size > 0):
            new_dates=dataslice['expMJD'][good]*0
            nDates = new_dates.size
            new_dates[nDates/2:] = self.baseline*365.25
            result = (sigma_slope(new_dates,  precis[good])*365.25*1e3)/result 
        # Observations that are very close together can still fail
        if np.isnan(result):
            result = self.badval 
        return result
        
