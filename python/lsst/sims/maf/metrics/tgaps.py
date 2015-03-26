import numpy as np
from .baseMetric import BaseMetric

class Tgaps(BaseMetric):
    """Histogram up all the time gaps """

    def __init__(self, timesCol='expMJD', allGaps=False, binMin=.5,
                 binMax=60.5, binsize=0.5, plotDict=None, units='days',
                 **kwargs):
        """
        Metric to measure the gaps between observations.  By default, only gaps
        between neighboring visits are computed.  If allGaps is set to true, all gaps are
        computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
        return [10,10,10] while allGaps returns [10,10,10,20,20,30])

        The gaps are binned into a histogram with properties set by the binMin, binMax, and binSize paramters.

        timesCol = column name for the exposure times.  Assumed to be in days.
        allGaps = should all observation gaps be computed (True), or
                  only gaps between consecutive observations (False, default)
        """
        if plotDict is None:
            plotDict = {}
        # If the user has not set the binmin/max/binsize for the plotDict,
        #   set it here so that the slicer uses the same bins.
        # (also set it here if the number of the bins set by user doesn't match).
        # Allow the user to set the bins explicitly though, in case they want a scale change.
        self.bins=np.arange(binMin, binMax+binsize,binsize)
        if ('binMin' in plotDict) and ('binMax' in plotDict) and ('binsize' in plotDict):
            plotbins = np.arange(plotDict['binMin'], plotDict['binMax'] + plotDict['binsize'],
                                 plotDict['binsize'])
            if plotbins.size != self.bins.size:
                warnings.warn('Overriding user set plotting bin parameters for tgaps, to match metric values')
                plotDict['binMin'] = binMin
                plotDict['binMax'] = binMax
                plotDict['binsize'] = binsize
        else:
            plotDict['binMin'] = binMin
            plotDict['binMax'] = binMax
            plotDict['binsize'] = binsize
        self.timesCol = timesCol
        super(Tgaps, self).__init__(col=[self.timesCol],
                                    metricDtype=object, plotDict=plotDict,
                                    units=units, **kwargs)
        self.allGaps = allGaps

    def run(self, dataSlice, slicePoint=None):
        if dataSlice.size < 2:
            return self.badval
        times = np.sort(dataSlice[self.timesCol])
        if self.allGaps:
            allDiffs = []
            for i in np.arange(1,times.size,1):
                allDiffs.append( (times-np.roll(times,i))[i:] )
            dts = np.concatenate(allDiffs)
        else:
            dts = np.diff(times)
        result, bins = np.histogram(dts, self.bins)
        return result

