import numpy as np
from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric

class BestSeeingCoaddMetric(BaseMetric):
    """Use the best seeing images to make a coadd """
    def __init__(self, metricName='BestSeeingCoaddMetric',
                 seeingCol = 'finSeeing', m5Col='fivesigma_modified',
                 percentile= 0.5, **kwargs):
        cols=[seeingCol, m5Col]
        units=''
        super(BestSeeingCoaddMetric, self).__init__(cols, metricName, units=units, **kwargs)
        self.metricDtype = 'object'
        self.m5Col = m5Col
        self.seeingCol = seeingCol
        self.percentile = percentile

    def run(self, dataSlice, slicePoint=None):
        goodSeeing = np.where(dataSlice[self.seeingCol] <=
                              np.percentile(dataSlice[self.seeingCol],
                                            self.percentile) )[0]
        m5Metric = Coaddm5Metric(m5col=self.m5Col)
        m5 = m5Metric.run(dataSlice[goodSeeing])
        meanSeeing = np.mean(dataSlice[self.seeingCol][goodSeeing])
        return {'m5':m5, 'meanSeeing':meanSeeing}
        
    def reduceM5(self, data):
        result = data['m5']
        return result

    def reduceMeanSeeing(self, data):
        result = data['meanSeeing']
        return result
