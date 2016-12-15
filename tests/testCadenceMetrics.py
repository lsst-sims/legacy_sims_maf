import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.utils.tests


class TestCadenceMetrics(unittest.TestCase):

    def testPhaseGapMetric(self):
        """
        Test the phase gap metric
        """
        data = np.zeros(10, dtype=zip(['observationStartMJD'], [float]))
        data['observationStartMJD'] += np.arange(10)*.25

        pgm = metrics.PhaseGapMetric(nPeriods=1, periodMin=0.5, periodMax=0.5)
        metricVal = pgm.run(data)

        meanGap = pgm.reduceMeanGap(metricVal)
        medianGap = pgm.reduceMedianGap(metricVal)
        worstPeriod = pgm.reduceWorstPeriod(metricVal)
        largestGap = pgm.reduceLargestGap(metricVal)

        assert(meanGap == 0.5)
        assert(medianGap == 0.5)
        assert(worstPeriod == 0.5)
        assert(largestGap == 0.5)

        pgm = metrics.PhaseGapMetric(nPeriods=2, periodMin=0.25, periodMax=0.5)
        metricVal = pgm.run(data)

        meanGap = pgm.reduceMeanGap(metricVal)
        medianGap = pgm.reduceMedianGap(metricVal)
        worstPeriod = pgm.reduceWorstPeriod(metricVal)
        largestGap = pgm.reduceLargestGap(metricVal)

        assert(meanGap == 0.75)
        assert(medianGap == 0.75)
        assert(worstPeriod == 0.25)
        assert(largestGap == 1.)

    def testSNMetric(self):
        """
        Test the SN Cadence Metric.
        """
        names = ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        types = [float, '|S1', float]
        data = np.zeros(700, dtype=zip(names, types))
        data['observationStartMJD'] = np.arange(0., 100., 1/7.)  # So, 100 days are well sampled in 2 filters
        data['filter'] = 'r'
        data['filter'][np.arange(0, 700, 2)] = 'g'
        data['fiveSigmaDepth'] = 30.
        slicePoint = {'sid': 0}
        metric = metrics.SupernovaMetric()
        result = metric.run(data, slicePoint)
        np.testing.assert_array_almost_equal(metric.reduceMedianMaxGap(result), 1/7.)
        assert(metric.reduceNsequences(result) == 10)
        assert((metric.reduceMedianNobs(result) < 561) & (metric.reduceMedianNobs(result) > 385))

    def testTemplateExists(self):
        """
        Test the TemplateExistsMetric.
        """
        names = ['finSeeing', 'observationStartMJD']
        types = [float, float]
        data = np.zeros(10, dtype=zip(names, types))
        data['finSeeing'] = [2., 2., 3., 1., 1., 1., 0.5, 1., 0.4, 1.]
        data['observationStartMJD'] = np.arange(10)
        slicePoint = {'sid': 0}
        # so here we have 4 images w/o good previous templates
        metric = metrics.TemplateExistsMetric(seeingCol='finSeeing')
        result = metric.run(data, slicePoint)
        assert(result == 6./10.)

    def testUniformityMetric(self):
        names = ['observationStartMJD']
        types = [float]
        data = np.zeros(100, dtype=zip(names, types))
        metric = metrics.UniformityMetric()
        result1 = metric.run(data)
        # If all the observations are on the 1st day, should be 1
        assert(result1 == 1)
        data['observationStartMJD'] = data['observationStartMJD']+365.25*10
        slicePoint = {'sid': 0}
        result2 = metric.run(data, slicePoint)
        # All on last day should also be 1
        assert(result1 == 1)
        # Make a perfectly uniform dist
        data['observationStartMJD'] = np.arange(0., 365.25*10, 365.25*10/100)
        result3 = metric.run(data, slicePoint)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.)
        # A single obseravtion should give a result of 1
        data = np.zeros(1, dtype=zip(names, types))
        result4 = metric.run(data, slicePoint)
        assert(result4 == 1)

    def testTGapMetric(self):
        names = ['observationStartMJD']
        types = [float]
        data = np.zeros(100, dtype=zip(names, types))
        # All 1-day gaps
        data['observationStartMJD'] = np.arange(100)

        metric = metrics.TgapsMetric(bins=np.arange(1, 100, 1))
        result1 = metric.run(data)
        # By default, should all be in first bin
        assert(result1[0] == data.size-1)
        assert(np.sum(result1) == data.size-1)
        data['observationStartMJD'] = np.arange(0, 200, 2)
        result2 = metric.run(data)
        assert(result2[1] == data.size-1)
        assert(np.sum(result2) == data.size-1)

        data = np.zeros(4, dtype=zip(names, types))
        data['observationStartMJD'] = [10, 20, 30, 40]
        metric = metrics.TgapsMetric(allGaps=True, bins=np.arange(1, 100, 10))
        result3 = metric.run(data)
        assert(result3[1] == 2)
        Ngaps = np.math.factorial(data.size-1)
        assert(np.sum(result3) == Ngaps)

    def testRapidRevisitMetric(self):
        data = np.zeros(100, dtype=zip(['observationStartMJD'], [float]))
        # Uniformly distribute time _differences_ between 0 and 100
        dtimes = np.arange(100)
        data['observationStartMJD'] = dtimes.cumsum()
        # Set up "rapid revisit" metric to look for visits between 5 and 25
        metric = metrics.RapidRevisitMetric(dTmin=5, dTmax=55, minNvisits=50)
        result = metric.run(data)
        # This should be uniform.
        self.assertTrue(result < 0.1)
        self.assertTrue(result >= 0)
        # Set up non-uniform distribution of time differences
        dtimes = np.zeros(100) + 5
        data['observationStartMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertTrue(result >= 0.5)
        dtimes = np.zeros(100) + 15
        data['observationStartMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertTrue(result >= 0.5)
        # Let's see how much dmax/result can vary
        resmin = 1
        resmax = 0
        for i in range(10000):
            dtimes = np.random.rand(100)
            data['observationStartMJD'] = dtimes.cumsum()
            metric = metrics.RapidRevisitMetric(dTmin=0.1, dTmax=0.8, minNvisits=50)
            result = metric.run(data)
            resmin = np.min([resmin, result])
            resmax = np.max([resmax, result])
        print "RapidRevisit .. range", resmin, resmax

    def testNRevisitsMetric(self):
        data = np.zeros(100, dtype=zip(['observationStartMJD'], [float]))
        dtimes = np.arange(100)/24./60.
        data['observationStartMJD'] = dtimes.cumsum()
        metric = metrics.NRevisitsMetric(dT=50.)
        result = metric.run(data)
        self.assertEqual(result, 50)
        metric = metrics.NRevisitsMetric(dT=50., normed=True)
        result = metric.run(data)
        self.assertEqual(result, 0.5)

    def testTransientMetric(self):
        names = ['observationStartMJD', 'fiveSigmaDepth', 'filter']
        types = [float, float, '|S1']

        ndata = 100
        dataSlice = np.zeros(ndata, dtype=zip(names, types))
        dataSlice['observationStartMJD'] = np.arange(ndata)
        dataSlice['fiveSigmaDepth'] = 25
        dataSlice['filter'] = 'g'

        metric = metrics.TransientMetric(surveyDuration=ndata/365.25)

        # Should detect everything
        assert(metric.run(dataSlice) == 1.)

        # Double to survey duration, should now only detect half
        metric = metrics.TransientMetric(surveyDuration=ndata/365.25*2)
        assert(metric.run(dataSlice) == 0.5)

        # Set half of the m5 of the observations very bright, so kill another half.
        dataSlice['fiveSigmaDepth'][0:ndata/2] = 20
        assert(metric.run(dataSlice) == 0.25)

        dataSlice['fiveSigmaDepth'] = 25
        # Demand lots of early observations
        metric = metrics.TransientMetric(peakTime=.5, nPrePeak=3, surveyDuration=ndata/365.25)
        assert(metric.run(dataSlice) == 0.)

        # Demand a reasonable number of early observations
        metric = metrics.TransientMetric(peakTime=2, nPrePeak=2, surveyDuration=ndata/365.25)
        assert(metric.run(dataSlice) == 1.)

        # Demand multiple filters
        metric = metrics.TransientMetric(nFilters=2, surveyDuration=ndata/365.25)
        assert(metric.run(dataSlice) == 0.)

        dataSlice['filter'] = ['r', 'g']*50
        assert(metric.run(dataSlice) == 1.)

        # Demad too many observation per light curve
        metric = metrics.TransientMetric(nPerLC=20, surveyDuration=ndata/365.25)
        assert(metric.run(dataSlice) == 0.)

        # Test both filter and number of LC samples
        metric = metrics.TransientMetric(nFilters=2, nPerLC=3, surveyDuration=ndata/365.25)
        assert(metric.run(dataSlice) == 1.)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
