import numpy as np
from .baseMetric import BaseMetric

__all__ = ['TgapsMetric']

class TgapsMetric(BaseMetric):
    """Histogram all the time gaps.


    Measure the gaps between observations.  By default, only gaps
    between neighboring visits are computed.  If allGaps is set to true, all gaps are
    computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
    return [10,10,10] while allGaps returns [10,10,10,20,20,30])

    The gaps are binned into a histogram with properties set by the bins array.

    timesCol : str, opt
        The column name for the exposure times.  Assumed to be in days.
    allGaps : bool, opt
        Should all observation gaps be computed (True), or
        only gaps between consecutive observations (False, default)

    Returns a histogram at each data point; these histograms can be combined and plotted using the
    'SummaryHistogram plotter'.
     """

    def __init__(self, timesCol='observationStartMJD', allGaps=False, bins=np.arange(0.5, 60.0, 0.5),
                 units='days', **kwargs):
        # Pass the same bins to the plotter.
        self.bins = bins
        self.timesCol = timesCol
        super(TgapsMetric, self).__init__(col=[self.timesCol], metricDtype='object', units=units, **kwargs)
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


class NVisitsPerNightMetric(BaseMetric):
    """Histogram the number of visits to a field per night.

     """

    def __init__(self, timesCol='observationStartMJD', allGaps=False, bins=np.arange(0.5, 60.0, 0.5),
                 units='days', **kwargs):
        """
        Metric to measure the gaps between observations.  By default, only gaps
        between neighboring visits are computed.  If allGaps is set to true, all gaps are
        computed (i.e., if there are observations at 10, 20, 30 and 40 the default will
        return [10,10,10] while allGaps returns [10,10,10,20,20,30])

        The gaps are binned into a histogram with properties set by the bins array.

        timesCol = column name for the exposure times.  Assumed to be in days.
        allGaps = should all observation gaps be computed (True), or
                  only gaps between consecutive observations (False, default)
        """
        # Pass the same bins to the plotter.
        self.bins = bins
        self.timesCol = timesCol
        super(TgapsMetric, self).__init__(col=[self.timesCol], metricDtype='object', units=units, **kwargs)
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