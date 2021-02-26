import numpy as np
from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
from lsst.sims.photUtils import Dust_values

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
    """
    def __init__(self, m5Col='fiveSigmaDepth', metricName='ExgalM5', units='mag',
                 lsstFilter='r', **kwargs):
        # Set the name for the dust map to use. This is gathered into the MetricBundle.
        maps = ['DustMap']
        self.m5Col = m5Col
        super().__init__(col=[self.m5Col], maps=maps, metricName=metricName, units=units, **kwargs)
        # Set the default wavelength limits for the lsst filters. These are approximately correct.
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1
        # We will call Coaddm5Metric to calculate the coadded depth. Set it up here.
        self.Coaddm5Metric = Coaddm5Metric(m5Col=m5Col)

    def run(self, dataSlice, slicePoint):
        """
        Compute the co-added m5 depth and then apply dust extinction to that magnitude.
        """
        m5 = self.Coaddm5Metric.run(dataSlice)
        # Total dust extinction along this line of sight. Correct default A to this EBV value.
        A_x = self.Ax1 * slicePoint['ebv']
        return m5 - A_x
