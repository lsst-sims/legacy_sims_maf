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
        """
        Parameters
        ----------
        m5Col : str ('fiveSigmaDepth')
            Column name that contains the five-sigma limiting depth of
            each observation
        optM5Col : str ('m5Optimal')
            The column name of the five-sigma-limiting depth if the
            observation had been taken on the meridian.
        normalize : bool (False)
            If False, metric returns how many more observations would need
            to be taken to reach the optimal depth.  If True, the number
            is normalized by the total number of observations already taken
            at that position.

        Returns
        --------
        out : float
            If normalize is True, output is the percentage of time the survey
            is behind an optimal meridian-scanning survey. So if a 10-year
            survey is at 20%, it would need to run for 12 years to reach the
            same depth as a 10-year meridian survey.
            If normalize is False (default), `out` is just the number of
            additional observations the survey needs to catch up to optimal.
        """
        if normalize:
            self.units = '% behind'
        else:
            self.units = 'N visits behind'
        super(OptimalM5Metric, self).__init__(col=[m5Col, optM5Col],
                                              units=self.units, **kwargs)
        self.m5Col = m5Col
        self.optM5Col = optM5Col
        self.normalize = normalize
        self.coaddRegular = Coaddm5Metric(m5Col=m5Col)
        self.coaddOptimal = Coaddm5Metric(m5Col=optM5Col)

    def run(self, dataSlice, slicePoint=None):

        regularDepth = self.coaddRegular.run(dataSlice)
        optimalDepth = self.coaddOptimal.run(dataSlice)
        medianSingle = np.median(dataSlice[self.m5Col])

        result = (10.**(0.8 * optimalDepth)-10.**(0.8 * regularDepth)) / \
                 (10.**(0.8 * medianSingle))

        if self.normalize:
            result = result/np.size(dataSlice)*100.

        return result
