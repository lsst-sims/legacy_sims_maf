from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
import numpy as np

__all__ = ['OptimalM5Metric']

class OptimalM5Metric(BaseMetric):
    """
    Compare the co-added depth of the survey to one where
    all the observations were taken on the meridian.
    """

    def __init__(self, m5Col='fiveSigmaDepth', optM5Col='m5Optimal',
                 normalize=False, **kwargs):
        if normalize:
            self.units = '% behind'
        else:
            self.units = 'N visits behind'
        super(OptimalM5Metric, self).__init__(col=[m5Col,optM5Col], units=self.units, **kwargs)
        self.m5Col = m5Col
        self.optM5Col = optM5Col
        self.normalize = normalize
        self.coaddRegular = Coaddm5Metric(m5Col=m5Col)
        self.coaddOptimal = Coaddm5Metric(m5Col=optM5Col)

    def run(self, dataSlice, slicePoint=None):
        """
        Find how many median depth exposures it would take to raise the
        co-added depth to the optimal meridian co-added depth.
        """
        regularDepth = self.coaddRegular.run(dataSlice)
        optimalDepth = self.coaddOptimal.run(dataSlice)
        medianSingle = np.median(dataSlice[self.m5Col])

        result = (10.**(0.8*optimalDepth)-10.**(0.8*regularDepth))/ \
                 (10.**(0.8*medianSingle))

        if self.normalize:
            result = result/np.size(dataSlice)*100.

        return result
