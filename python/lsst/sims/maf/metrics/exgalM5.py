from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
from lsst.sims.photUtils import Sed

__all__ = ['ExgalM5']


class ExgalM5(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction.

    Uses photUtils
    """
    def __init__(self, m5Col='fiveSigmaDepth', units='mag', maps=['DustMap'],
                 lsstFilter='r', wavelen_min=None, wavelen_max=None, wavelen_step=1., **kwargs):
        """
        Args:
            m5Col (str): Column name that ('fiveSigmaDepth')
            units (str): units of the metric ('mag')
            maps (list): List of maps to use with the metric (['DustMap'])
            lsstFilter (str): Which LSST filter to calculate m5 for
            wavelen_min (float): Minimum wavength of your filter (None)
            wavelen_max (float): (None)
            wavelen_step (float): (1.)
            **kwargs:
        """

        waveMins={'u':330.,'g':403.,'r':552.,'i':691.,'z':818.,'y':950.}
        waveMaxes={'u':403.,'g':552.,'r':691.,'i':818.,'z':922.,'y':1070.}

        if lsstFilter is not None:
            wavelen_min = waveMins[lsstFilter]
            wavelen_max = waveMaxes[lsstFilter]

        self.m5Col = m5Col
        super(ExgalM5, self).__init__(col=[self.m5Col],
                                      maps=maps, units=units, **kwargs)

        testsed = Sed()
        testsed.setFlatSED(wavelen_min = wavelen_min,
                           wavelen_max = wavelen_max, wavelen_step = 1)
        self.a,self.b = testsed.setupCCMab()
        self.R_v = 3.1
        self.Coaddm5Metric = Coaddm5Metric(m5Col=m5Col)


    def run(self, dataSlice, slicePoint=None):
        """
        Compute the co-added m5 depth and then apply extinction to that magnitude.

        Args:
            dataSlice (np.array):
            slicePoint (dict):
        Returns:
             float that is the dust atennuated co-added m5-depth.
        """

        m5 = self.Coaddm5Metric.run(dataSlice)
        A_x = (self.a[0]+self.b[0]/self.R_v)*(self.R_v*slicePoint['ebv'])
        result = m5-A_x
        return result
