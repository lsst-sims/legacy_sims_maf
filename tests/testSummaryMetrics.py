import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics


class TestSummaryMetrics(unittest.TestCase):
    def setUp(self):
        dv = np.arange(0, 1, .1)
        self.dv = np.array(zip(dv), dtype=[('testdata', 'float')])

    def testTableFractionMetric(self):
        """Test the table summary metric """
        metric = metrics.TableFractionMetric
        result = metric.run(self.dv)
        assert(np.max(result) == 1)
        assert(np.size(result) == 10)
        

    def testExactCompleteMetric(self):
        """Test ExamctCompleteMetric to be sure it pulls out the fraction of values that equal 1 """
        data = np.array([1.,1.,0.,0.], dtype=[('testdata','float')])
        metric = metrics.ExactCompleteMetric
        assert(metric.run(data) == np.size(np.where(data == 1.))/float(np.size(data)) )
        data = np.array([], dtype=[('testdata','float')])
        assert(metric.run(data) == metric.badval)
        
