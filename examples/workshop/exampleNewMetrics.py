from lsst.sims.maf.metrics import BaseMetric, SimpleScalarMetric
import numpy as np

class SimplePercentileMetric(SimpleScalarMetric):
    def run(self, dataSlice, slicePoint=None):
        return np.percentile(dataSlice[self.colname], 95)

class PercentileMetric(SimpleScalarMetric):
    def __init__(self, colname, percentile=90, **kwargs):
        super(PercentileMetric, self).__init__(colname, **kwargs)
        self.percentile = percentile
    def run(self, dataSlice, slicePoint=None):
        pval = np.percentile(dataSlice[self.colname], self.percentile)
        return pval

class DifferenceMetric(BaseMetric):
    """
    Take the difference between two data columns and return the max value of the difference.
    """
    def __init__(self, colA, colB=None, **kwargs):
        self.colA = colA
        self.colB = colB
        super(DifferenceMetric, self).__init__([self.colA, self.colB], metricDtype='float', **kwargs)
    def run(self, dataSlice, slicePoint=None):
        if self.colB is not None:
            difference = dataSlice[self.colA] - dataSlice[self.colB]
            difference = np.abs(difference).max()
        else:
            difference = 0
        return difference
    
        
class CoaddedDepthBestSeeingMetric(BaseMetric):
    """
    Metric to calculate coadded limiting magnitude of images,
    using only visitFrac of the visits with best seeing.
    """
    def __init__(self, seeingCol='finSeeing', m5col='fivesigma_modified', visitFrac=0.5, **kwargs):
        """
        seeingCol = seeing column
        m5col = five sigma limiting magnitude column
        visitFrac = fraction of visits with best seeing to use.
        """
        self.seeingCol = seeingCol
        self.m5col = m5col
        self.visitFrac = visitFrac
        super(CoaddedDepthBestSeeingMetric, self).__init__([self.seeingCol, self.m5col],
                                                           metricDtype='float', units='mag',
                                                           **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Get the indexes of the dataSlice array, sorted by seeing values.
        seeingorder = np.argsort(dataSlice[self.seeingCol])
        # Translate visitFrac into number of visits to use.
        numvisits = self.visitFrac * len(seeingorder)
        if numvisits < 1:
            numvisits = 1
        else:
            numvisits = int(np.floor(numvisits))
        # Identify the visits we want to use.
        bestseeingvisits = seeingorder[:numvisits]
        # Calculate coadded depth of these visits.
        coaddm5 = 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5col][bestseeingvisits])))
        return coaddm5

class ComplexCoaddedDepthBestSeeingMetric(BaseMetric):
    """
    Metric to calculate the coadded limiting magnitude of a set
    of images, using only visitFrac of the visits with best seeing -- and to
    make a map both the resulting seeing and coadded depth values.
    """
    def __init__(self, seeingCol='finSeeing', m5col='fivesigma_modified', visitFrac=0.5, **kwargs):
        """
        seeingCol = seeing column
        m5col = five sigma limiting magnitude column
        visitFrac = fraction of visits with best seeing to use.
        """
        self.seeingCol = seeingCol
        self.m5col = m5col
        self.visitFrac = visitFrac
        super(ComplexCoaddedDepthBestSeeingMetric, self).__init__([self.seeingCol, self.m5col],
                                                           metricDtype='object', units='',
                                                           **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Identify visits with seeing better than visitFrac.
        seeingorderIdx = np.argsort(dataSlice[self.seeingCol])
        # Translate visitFrac into number of visits to use.
        numvisits = self.visitFrac * len(seeingorderIdx)
        if numvisits < 1:
            numvisits = 1
        else:
            numvisits = int(np.floor(numvisits))
        # Identify the visits we want to use.
        bestseeingvisitsIdx = seeingorderIdx[:numvisits]
        # Calculate coadded depth of these visits.
        coaddm5 = 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5col][bestseeingvisitsIdx])))
        # Calculate the mean of those bestseeing visits.
        meanseeing = np.mean(dataSlice[self.seeingCol][bestseeingvisitsIdx])
        return {'m5':m5, 'meanSeeing':meanSeeing}
        
    def reduceM5(self, data):
        return data['m5']
    def reduceMeanSeeing(self, data):
        return data['meanSeeing']
