import numpy as np
from .baseMetric import BaseMetric
from lsst.sims.maf.utils.astrometryUtils import *

class ParallaxMetric(BaseMetric):
    """Calculate the uncertainty in a parallax measures given a serries of observations.
    """
    def __init__(self, metricName='parallax', m5col='fivesigma_modified',
                 mjdcol='expMJD', units = 'mas',
                 filtercol='filter', seeingcol='finSeeing',rmag=20.,
                 SedTemplate='flat', badval= -666,
                 atm_err=0.01, normalize=False,**kwargs):
        
        """ Instantiate metric.

        m5col = column name for inidivual visit m5
        mjdcol = column name for exposure time dates
        filtercol = column name for filter
        seeingcol = column name for seeing (assumed FWHM)
        rmag = mag of fiducial star in r filter.  Other filters are scaled using sedTemplate keyword
        atm_err = centroiding error due to atmosphere in arcsec
        normalize = Compare to a survey that has all observations with maximum parallax factor.
        An optimally scheduled survey would be expected to have a normalized value close to unity,
        and zero for a survey where the parallax can not be measured.

        return uncertainty in mas. Or normalized map as a fraction
        """
        cols = [m5col, mjdcol,filtercol,seeingcol, 'ra_pi_amp', 'dec_pi_amp']
        if normalize:
            units = 'ratio'
        super(ParallaxMetric, self).__init__(cols, metricName=metricName, units=units, **kwargs)
        # set return type
        self.m5col = m5col
        self.seeingcol = seeingcol
        self.filtercol = filtercol
        self.metricDtype = 'float'
        filters=['u','g','r','i','z','y']
        self.mags={}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            raise NotImplementedError('Spot to add colors for different stars')
        self.badval = badval
        self.atm_err = atm_err
        self.normalize = normalize
        
    def _final_sigma(self, position_errors, ra_pi_amp, dec_pi_amp):
        """Assume parallax in RA and DEC are fit independently, then combined.
        All inputs assumed to be arcsec """
        sigma_A = position_errors/ra_pi_amp
        sigma_B = position_errors/dec_pi_amp
        sigma_ra = np.sqrt(1./np.sum(1./sigma_A**2))
        sigma_dec = np.sqrt(1./np.sum(1./sigma_B**2))
        sigma = np.sqrt(1./(1./sigma_ra**2+1./sigma_dec**2))*1e3 #combine RA and Dec uncertainties, convert to mas
        return sigma
        
    def run(self, dataslice, slicePoint=None):
        filters = np.unique(dataslice[self.filtercol])
        snr = np.zeros(len(dataslice), dtype='float')
        # compute SNR for all observations
        for filt in filters:
            good = np.where(dataslice[self.filtercol] == filt)
            snr[good] = m52snr(self.mags[filt], dataslice[self.m5col][good])
        position_errors = np.sqrt(astrom_precision(dataslice[self.seeingcol], snr)**2+self.atm_err**2)
        sigma = self._final_sigma(position_errors,dataslice['ra_pi_amp'],dataslice['dec_pi_amp'] )
        if self.normalize:
            # Leave the dec parallax as zero since one can't have ra and dec maximized at the same time.
            sigma = self._final_sigma(position_errors,dataslice['ra_pi_amp']*0+1.,dataslice['dec_pi_amp']*0 )/sigma
        return sigma
        


class ProperMotionMetric(BaseMetric):
    """Calculate the uncertainty in the returned proper motion.  Assuming Gaussian errors.
    """
    def __init__(self, metricName='properMotion',
                 m5col='fivesigma_modified', mjdcol='expMJD', units='mas/yr',
                 filtercol='filter', seeingcol='finSeeing',  rmag=20.,
                 SedTemplate='flat', badval= -666,
                 atm_err=0.01, normalize=False,
                 baseline=10., **kwargs):
        """ Instantiate metric.

        m5col = column name for inidivual visit m5
        mjdcol = column name for exposure time dates
        filtercol = column name for filter
        seeingcol = column name for seeing (assumed FWHM)
        rmag = mag of fiducial star in r filter.  Other filters are scaled using sedTemplate keyword
        sedTemplate = template to use (currently only 'flat' is implamented)
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
        filters=['u','g','r','i','z','y']
        self.mags={}
        if SedTemplate == 'flat':
            for f in filters:
                self.mags[f] = rmag
        else:
            raise NotImplementedError('Spot to add colors for different stars')
        self.badval = badval
        self.atm_err = atm_err
        self.normalize = normalize
        self.baseline = baseline

    def run(self, dataslice, slicePoint=None):
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

## Check radius of observations to look for calibration effects.

def calcDist_cosines(RA1, Dec1, RA2, Dec2):
    #taken from simSelfCalib.py
    """Calculates distance on a sphere using spherical law of cosines.
    
    Give this function RA/Dec values in radians. Returns angular distance(s), in radians.
    Note that since this is all numpy, you could input arrays of RA/Decs."""
    # This formula can have rounding errors for case where distances are small.
    # Oh, the joys of wikipedia - http://en.wikipedia.org/wiki/Great-circle_distance 
    # For the purposes of these calculations, this is probably accurate enough.
    D = np.sin(Dec2)*np.sin(Dec1) + np.cos(Dec1)*np.cos(Dec2)*np.cos(RA2-RA1)
    D = np.arccos(D)
    return D

class RadiusObsMetric(BaseMetric):
    """find the radius in the focal plane. """

    def __init__(self, metricName='radiusObs', racol='fieldRA',deccol='fieldDec',
                 units='radians', **kwargs):
        cols = [racol,deccol]
        self.racol = racol
        self.deccol=deccol
        self.units=units
        super(RadiusObsMetric,self).__init__(cols,metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint):
        ra = slicePoint['ra']
        dec = slicePoint['dec']
        distances = calcDist_cosines(ra,dec, dataSlice[self.racol], dataSlice[self.deccol])
        return distances

    def reduceMean(self, distances):
        return np.mean(distances)
    def reduceRMS(self,distances):
        return np.std(distances)
    def reduceFullRange(self,distances):
        return np.max(distances)-np.min(distances)
    
