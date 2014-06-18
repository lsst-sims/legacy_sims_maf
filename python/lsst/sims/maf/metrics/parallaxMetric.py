import numpy as np
from .baseMetric import BaseMetric
from lsst.sims.maf.utils.astrometryUtils import m52snr, astrom_precision

class ParallaxMetric(BaseMetric):
    """Calculaute the uncertainty in a parallax measure
    given a serries of observations"""

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
        
    def run(self, dataslice, *args):
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
        
