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
    def __init__(self, crowding_error=0.1, seeingCol='finSeeing',
                 fiveSigCol='fiveSigmaDepth', units='mag', maps=['StellarDensityMap'],
                 metricName='Crowding To Precision', **kwargs):
        """
        Parameters
        ----------
        crowding_error : float (0.1)
            The magnitude uncertainty from crowding. (mags)

        Returns
        -------
        float
        The magnitude of a star which has a photometric error of `crowding_error`
        """
        cols=[seeingCol,fiveSigCol]
        self.crowding_error = crowding_error
        self.seeingCol = seeingCol
        self.fiveSigCol = fiveSigCol
        self.lumAreaArcsec = 3600.0**2

        super(CrowdingMetric, self).__init__(col=cols, maps=maps, units=units, metricName=metricName, **kwargs)


    def _compCrowdError(self, magVector, lumFunc, seeing, singleMag=None):
        """
        Compute the crowding error for each observation

        Parameters
        ----------
        magVector : np.array
            Stellar magnitudes.
        lumFunc : np.array
            Stellar luminosity function.
        seeing : float
            The best seeing conditions. Assuming forced-photometry can use the best seeing conditions
            to help with confusion errors.
        singleMag : float (None)
            If singleMag is None, the crowding error is calculated for each mag in magVector. If
            singleMag is a float, the corwding error is interpolated to that single value.

        Returns
        -------
        np.array
            Magnitude uncertainties.


        Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
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

class CrowdingMagUncertMetric(CrowdingMetric):
    """
    Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
    """
    def __init__(self, rmag=20., seeingCol='finSeeing',
                 fiveSigCol='fiveSigmaDepth', maps=['StellarDensityMap'], units='mag',
                 metricName='CrowdingMagUncert', **kwargs):
        """
        Parameters
        ----------
        rmag : float
            The magnitude of the star to consider.

        Returns
        -------
        float
            The uncertainty in magnitudes caused by crowding for a star of rmag.
        """
        self.rmag = rmag
        super(CrowdingMagUncertMetric, self).__init__(seeingCol=seeingCol,fiveSigCol=fiveSigCol,
                                                      maps=maps, units=units, metricName=metricName,
                                                      **kwargs)

    def run(self, dataSlice, slicePoint=None):

        magVector = slicePoint['starMapBins'][1:]
        lumFunc = slicePoint['starLumFunc']
        # Magnitude uncertainty given crowding
        dmagCrowd = self._compCrowdError(magVector, lumFunc,
                                         dataSlice[self.seeingCol], singleMag=self.rmag)

        result = np.mean(dmagCrowd)
        return result
