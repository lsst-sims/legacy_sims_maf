import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics


class TestSimpleMetrics(unittest.TestCase):
    def setUp(self):
        dv = np.arange(0, 10, .5)
        self.dv = np.array(zip(dv), dtype=[('testdata', 'float')])
        
    def testMaxMetric(self):
        """Test max metric."""
        testmetric = metrics.MaxMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), self.dv['testdata'].max())

    def testMinMetric(self):
        """Test min metric."""
        testmetric = metrics.MinMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), self.dv['testdata'].min())

    def testMeanMetric(self):
        """Test mean metric."""
        testmetric = metrics.MeanMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), self.dv['testdata'].mean())

    def testMedianMetric(self):
        """Test median metric."""
        testmetric = metrics.MedianMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), np.median(self.dv['testdata']))

    def testFullRangeMetric(self):
        """Test full range metric."""
        testmetric = metrics.FullRangeMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), self.dv['testdata'].max()-self.dv['testdata'].min())

    def testCoaddm5Metric(self):
        """Test coaddm5 metric."""
        testmetric = metrics.Coaddm5Metric(m5Col='testdata')
        self.assertEqual(testmetric.run(self.dv), 1.25 * np.log10(np.sum(10.**(.8*self.dv['testdata']))))

    def testRmsMetric(self):
        """Test rms metric."""
        testmetric = metrics.RmsMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), np.std(self.dv['testdata']))

    def testSumMetric(self):
        """Test Sum metric."""
        testmetric = metrics.SumMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), self.dv['testdata'].sum())

    def testCountMetric(self):
        """Test count metric."""
        testmetric = metrics.CountMetric('testdata')
        self.assertEqual(testmetric.run(self.dv), np.size(self.dv['testdata']))

    def testRobustRmsMetric(self):
        """Test Robust RMS metric."""
        testmetric = metrics.RobustRmsMetric('testdata')
        rms_approx = (np.percentile(self.dv['testdata'], 75) - np.percentile(self.dv['testdata'], 25)) / 1.349
        self.assertEqual(testmetric.run(self.dv), rms_approx)

    def testFracAboveMetric(self):
        cutoff = 5.1
        testmetric=metrics.FracAboveMetric('testdata', cutoff = cutoff)
        self.assertEqual(testmetric.run(self.dv),
                         np.size(np.where(self.dv['testdata'] >= cutoff)[0])/float(np.size(self.dv)))

    def testFracBelowMetric(self):
        cutoff = 5.1
        testmetric=metrics.FracBelowMetric('testdata', cutoff = cutoff)
        self.assertEqual(testmetric.run(self.dv),
                         np.size(np.where(self.dv['testdata'] <= cutoff)[0])/float(np.size(self.dv)))

if __name__ == "__main__":
    unittest.main()
