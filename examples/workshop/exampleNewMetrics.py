from lsst.sims.maf.metrics import BaseMetric, SimpleScalarMetric

class PercentileMetric(SimpleScalarMetric):
    def __init__(self, colname, percentile=90, **kwargs):
        super(PercentileMetric, self).__init__(colname, **kwargs)
        self.percentile = percentile
    def run(self, dataSlice):
        pval = np.percentile(dataSlice[self.colname], self.percentile)
        return pval

class CoaddedDepthBestSeeingMetric(BaseMetric):
    """
    Metric to calculate coadded limiting magnitude of images,
    using only visitFrac of the visits with best seeing.
    """
    def __init__(seeingCol='finSeeing', m5col='fivesigma_modified', visitFrac=0.5, **kwargs):
        "
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

    def run(self, dataSlice, slicePoint):
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

# Add the following lines to the driver configuration script
#moduleDict = makeDict(['PATH_TO_WORKSHOP_DIR/exampleNewMetrics.py'])
#root.modules = moduleDict
    
#m1 = configureMetric('CoaddedDepthBestSeeingMetric', kwargs={'visitFrac':0.5})

