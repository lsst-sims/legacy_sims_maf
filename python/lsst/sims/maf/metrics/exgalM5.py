from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
from lsst.sims.photUtils import Sed

__all__ = ['ExgalM5', 'ExgalM5_cut']

class ExgalM5(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction.

    Uses photUtils
    """
    def __init__(self, m5Col='fiveSigmaDepth', units='mag',
                 lsstFilter='r', wavelen_min=None , wavelen_max=None , wavelen_step=1., **kwargs ):
        """
        Args:
            m5Col (str): Column name that ('fiveSigmaDepth')
            units (str): units of the metric ('mag')
            lsstFilter (str): Which LSST filter to calculate m5 for
            wavelen_min (float): Minimum wavength of your filter (None)
            wavelen_max (float): (None)
            wavelen_step (float): (1.)
            **kwargs:
        """
        maps = ['DustMap']
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
        self.a,self.b = testsed.setupCCM_ab()
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

    
class ExgalM5_cut(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction 
    and depth cuts

    A copy of ExgalM5 for use of FoMEmulator as a Summary Metric on this.
    """
    def __init__(self, m5Col='fiveSigmaDepth', units='mag',
                 lsstFilter='i', wavelen_min=None, wavelen_max=None, 
                 wavelen_step=1., extinction_cut=0.2, depth_cut=26, **kwargs):
        """
        Args: 
            m5Col (str): Column name that ('fiveSigmaDepth')
            units (str): units of the metric ('mag')
            lsstFilter (str): Which LSST filter to calculate m5 for
            wavelen_min (float): Minimum wavength of your filter (None)
            wavelen_max (float): (None)
            wavelen_step (float): (1.)
            **kwargs:
        """
        maps = ['DustMap']
        waveMins={'u':330., 'g':403., 'r':552., 'i':691., 'z':818., 'y':950.}
        waveMaxes={'u':403., 'g':552., 'r':691., 'i':818., 'z':922., 'y':1070.}

        if lsstFilter is not None:
            wavelen_min = waveMins[lsstFilter]
            wavelen_max = waveMaxes[lsstFilter]

        self.m5Col = m5Col
        super(ExgalM5_cut, self).__init__(col=[self.m5Col],
                                          maps=maps, 
                                          units=units, 
                                          **kwargs
                                         )

        testsed = Sed()
        testsed.setFlatSED(wavelen_min=wavelen_min,
                           wavelen_max=wavelen_max, 
                           wavelen_step=1)
        self.a,self.b = testsed.setupCCM_ab()
        self.R_v = 3.1
        self.Coaddm5Metric = Coaddm5Metric(m5Col=m5Col)
        
        self.extinction_cut = extinction_cut
        self.depth_cut = depth_cut


    def run(self, dataSlice, slicePoint=None):
        """
        Compute the co-added m5 depth and then apply extinction cut
        and depth cut to that magnitude.
            
        Args:
            dataSlice (ndarray): Values passed to metric by the slicer, 
                which the metric will use to calculate metric values 
                at each slicePoint.
            slicePoint (Dict): Dictionary of slicePoint metadata passed
                to each metric.
        Returns:
             float: the dust atennuated co-added m5-depth.
        """
        
        if slicePoint['ebv'] > self.extinction_cut:
            return self.badval

        m5 = self.Coaddm5Metric.run(dataSlice)
        A_x = (self.a[0] + self.b[0]/self.R_v) * (self.R_v*slicePoint['ebv'])
        result = m5-A_x
        if result < self.depth_cut:
            return self.badval
        else:
            return result

