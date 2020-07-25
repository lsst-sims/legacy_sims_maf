import numpy as np
from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
from lsst.sims.photUtils import Sed, Bandpass

__all__ = ['ExgalM5']


class ExgalM5(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction.

    Uses photUtils to calculate dust extinction.

    Parameters
    ----------
    m5Col : str, opt
        Column name for five sigma depth. Default 'fiveSigmaDepth'.
    unit : str, opt
        Label for units. Default 'mag'.
    lsstFilter : str, opt
        Filter name for which to calculate m5 depth. Default 'r'.
        This is used to set the wavelength range over which to calculate dust extinction.
        Overrides wavelen_min/wavelen_max/wavelen_step if specified.
    wavelen_min : float, opt
        If lsstFilter is not specified, this can be used to set the minimum wavelength for dust extinction.
    wavelen_max : float, opt
        If lsstFilter is not specified, this can be used to set the maximum wavelength for dust extinction.
    """
    def __init__(self, m5Col='fiveSigmaDepth', metricName='ExgalM5', units='mag',
                 lsstFilter='r', wavelen_min=None , wavelen_max=None , **kwargs):
        # Set the name for the dust map to use. This is gathered into the MetricBundle.
        maps = ['DustMap']
        # Set the default wavelength limits for the lsst filters. These are approximately correct.
        waveMins = {'u':330.,'g':403.,'r':552.,'i':691.,'z':818.,'y':950.}
        waveMaxes = {'u':403.,'g':552.,'r':691.,'i':818.,'z':922.,'y':1070.}
        if lsstFilter is not None:
            wavelen_min = waveMins[lsstFilter]
            wavelen_max = waveMaxes[lsstFilter]

        self.m5Col = m5Col
        super().__init__(col=[self.m5Col], maps=maps, metricName=metricName, units=units, **kwargs)

        # Set up internal values for the dust extinction.
        testsed = Sed()
        testsed.setFlatSED(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=1.0)
        testbandpass = Bandpass(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=1.0)
        testbandpass.setBandpass(wavelen=testsed.wavelen,
                                 sb=np.ones(len(testsed.wavelen)))
        self.R_v = 3.1
        self.ref_ebv = 1.0
        # Calculate non-dust-extincted magnitude
        flatmag = testsed.calcMag(testbandpass)
        # Add dust
        self.a, self.b = testsed.setupCCM_ab()
        testsed.addDust(self.a, self.b, ebv=self.ref_ebv, R_v=self.R_v)
        # Calculate difference due to dust when EBV=1.0 (m_dust = m_nodust - Ax, Ax > 0)
        self.Ax1 = testsed.calcMag(testbandpass) - flatmag
        # We will call Coaddm5Metric to calculate the coadded depth. Set it up here.
        self.Coaddm5Metric = Coaddm5Metric(m5Col=m5Col)

    def run(self, dataSlice, slicePoint):
        """
        Compute the co-added m5 depth and then apply dust extinction to that magnitude.
        """
        m5 = self.Coaddm5Metric.run(dataSlice)
        # Total dust extinction along this line of sight. Correct default A to this EBV value.
        A_x = self.Ax1 * slicePoint['ebv'] / self.ref_ebv
        return m5 - A_x

