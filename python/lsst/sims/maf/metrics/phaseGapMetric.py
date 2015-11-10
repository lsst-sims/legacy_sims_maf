import numpy as np
from .baseMetric import BaseMetric

__all__ = ['PhaseGapMetric']

class PhaseGapMetric(BaseMetric):
    """
    Measure the maximum gap in phase coverage for observations of periodic variables.
    """
    def __init__(self, col='expMJD', nPeriods=5, periodMin=3., periodMax=35., nVisitsMin=3,
                 metricName='Phase Gap', **kwargs):
        """
        Construct an instance of a PhaseGapMetric class

        :param col: Name of the column to use for the observation times, commonly 'expMJD'
        :param nPeriods: Number of periods to test
        :param periodMin: Minimum period to test (days)
        :param periodMax: Maximimum period to test (days)
        :param nVistisMin: minimum number of visits necessary before looking for the phase gap
        """
        self.periodMin = periodMin
        self.periodMax = periodMax
        self.nPeriods = nPeriods
        self.nVisitsMin = nVisitsMin
        super(PhaseGapMetric, self).__init__(col, metricName=metricName, units='Fraction, 0-1', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        """
        Run the PhaseGapMetric.
        :param dataSlice: Data for this slice.
        :param slicePoint: Metadata for the slice (Optional as not used here).
        :return: a dictionary of the periods used here and the corresponding largest gaps.
        """
        if len(dataSlice) < self.nVisitsMin:
            return self.badval
        # Create 'nPeriods' evenly spaced periods within range of min to max.
        step = (self.periodMax-self.periodMin)/self.nPeriods
        if step == 0:
            periods = np.array([self.periodMin])
        else:
            periods = np.arange(self.nPeriods)
            periods = periods/np.max(periods)*(self.periodMax-self.periodMin)+self.periodMin
        maxGap = np.zeros(self.nPeriods, float)

        for i, period in enumerate(periods):
            # For each period, calculate the phases.
            phases = (dataSlice[self.colname] % period)/period
            phases = np.sort(phases)
            # Find the largest gap in coverage.
            gaps = np.diff(phases)
            start_to_end = np.array([1.0 - phases[-1] + phases[0]], float)
            gaps = np.concatenate([gaps, start_to_end])
            maxGap[i] = np.max(gaps)

        return {'periods':periods, 'maxGaps':maxGap}

    def reduceMeanGap(self, metricVal):
        """
        At each slicepoint, return the mean gap value.
        """
        return np.mean(metricVal['maxGaps'])

    def reduceMedianGap(self, metricVal):
        """
        At each slicepoint, return the median gap value.
        """
        return np.median(metricVal['maxGaps'])

    def reduceWorstPeriod(self, metricVal):
        """
        At each slicepoint, return the period with the largest phase gap.
        """
        worstP = metricVal['periods'][np.where(metricVal['maxGaps'] == metricVal['maxGaps'].max())]
        return worstP

    def reduceLargestGap(self, metricVal):
        """
        At each slicepoint, return the largest phase gap value.
        """
        return np.max(metricVal['maxGaps'])
