from lsst.sims.maf.metrics import BaseMetric
import numpy as np
from lsst.sims.maf.utils import m52snr
from scipy.interpolate import interp1d

# Modifying from Knut Olson's fork at:
# https://github.com/knutago/sims_maf_contrib/blob/master/tutorials/CrowdingMetric.ipynb

class CrowdingMetric(BaseMetric):
    """
    Calculate whether the coadded depth in r has exceeded the confusion limit
    """
    def __init__(self, crowding_error=0.1, lumArea=10., seeingCol='finSeeing',
                 fiveSigCol='fiveSigmaDepth', units='mag', maps=['lumFuncMap'],
                 metricName='Crowding To Precision', **kwargs):
        """
        Parameters
        ----------
        crowding_error : float (0.1)
            mags?
        lumArea : float (10.)
            Area in square degrees. XXX-not sure what this is?

        Returns
        -------
        float
        The magnitude of a star which has a photometric error of `crowding_error`
        """
        cols=[seeingCol,fiveSigCol]
        self.crowding_error = crowding_error
        self.seeingCol = seeingCol
        self.fiveSigCol = fiveSigCol
        self.lumAreaArcsec = lumArea*3600.0**2

        super(CrowdingMetric, self).__init__(col=cols, maps=maps, units=units, metricName=metricName, **kwargs)


    def _compCrowdError(self, magVector, lumFunc, seeing, singleMag=None):
        """
        Compute the crowding error for each observation
        Need seeing to be a single value, or magVector and lumFunc should be single values
        """
        lumVector = 10**(-0.4*magVector)
        coeff=np.sqrt(np.pi/self.lumAreaArcsec)*seeing/2.
        myIntergral = (np.add.accumulate((lumVector**2*lumFunc)[::-1]))[::-1]
        temp = np.sqrt(myIntergral)/lumVector
        if singleMag is not None:
            interp = interp1d(magVector, temp)
            temp = interp(singleMag)

        crowdError = coeff*temp

        return crowdError

    def run(self, dataSlice, slicePoint=None):

        magVector = slicePoint['starMapBins'][1:]
        lumFunc = slicePoint['starLumFunc']

        crowdError =self._compCrowdError(magVector, lumFunc, seeing=min(dataSlice[self.seeingCol]) )

        # Locate at which point crowding error is greater than user-defined limit
        aboveCrowd = np.where(crowdError >= self.crowding_error)[0]

        if np.size(aboveCrowd) == 0:
            return max(magVector)
        else:
            crowdMag = magVector[max(aboveCrowd[0]-1,0)]
            return crowdMag

# Questions I have:
# 1) What is the definition of lumArea, and why is the default 10?
# 2) Is there a cite I can drop in the code so people can look up where the
#    equation came from (I made it pretty unreadable getting rid of the lop)
# 3) Why pick the best seeing case?  Are you just assuming that with the best seeing image you can do forced-photometry on the rest? What if the best seeing case happens to be in bright time?

class CrowdingMagUncertMetric(CrowdingMetric):
    """
    Given a stellar magnitude, calculate the uncertainty on the magnitude, using the crowding uncertainty if dominant
    """
    def __init__(self, rmag=20., bestResult=True, crowding_error=0.1, lumArea=10., seeingCol='finSeeing',
                 fiveSigCol='fiveSigmaDepth', maps=['lumFuncMap'], units='mag',
                 metricName='CrowdingMagUncert', **kwargs):
        """
        Parameters
        ----------
        rmag : float
            The magnitude of the star to consider
        best : bool
            Return the best result, otherwise the full vector
        """
        self.rmag = rmag
        self.best = bestResult
        super(CrowdingMagUncertMetric, self).__init__(crowding_error=crowding_error, lumArea=lumArea,
                                                      seeingCol=seeingCol,fiveSigCol=fiveSigCol,
                                                      maps=maps, units=units, metricName=metricName,
                                                      **kwargs)

    def run(self, dataSlice, slicePoint=None):

        magVector = slicePoint['starMapBins'][1:]
        lumFunc = slicePoint['starLumFunc']

        # Magnitude uncertainty given crowding
        dmagCrowd = self._compCrowdError(magVector, lumFunc,
                                         dataSlice[self.seeingCol], singleMag=self.rmag)

        # compute the magnitude uncertainty from the usual m5 depth
        snr = m52snr(self.rmag, dataSlice[self.fiveSigCol])
        # Magnitude uncertainty, given 5-sigma limiting depth
        dmagRegular = 2.5*np.log10(1.+1./snr)

        # Take the max, assume one is dominant.
        result = np.maximum(dmagCrowd,dmagRegular)
        # The best now that we've taken crowding into account
        if self.best:
            result = np.min(result)
        return result
