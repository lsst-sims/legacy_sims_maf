import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics



class TestCadenceMetrics(unittest.TestCase):
                        
    def testSNMetric(self):
        """
        Test the SN Cadence Metric.
        """
        names = ['expMJD', 'filter', 'fiveSigmaDepth']
        types = [float,'|S1', float]
        data = np.zeros(700, dtype=zip(names,types))
        data['expMJD'] = np.arange(0.,100.,1/7.) # So, 100 days are well sampled in 2 filters
        data['filter']= 'r'
        data['filter'][np.arange(0,700,2)] = 'g'
        data['fiveSigmaDepth'] = 30.
        slicePoint = {'sid':0}
        metric = metrics.SupernovaMetric()
        result = metric.run(data, slicePoint)
        np.testing.assert_array_almost_equal(metric.reduceMedianMaxGap(result),  1/7.)
        assert(metric.reduceNsequences(result) == 10)
        assert((metric.reduceMedianNobs(result) <  561) & (metric.reduceMedianNobs(result) >  385) )

    def testTemplateExists(self):
        """
        Test the TemplateExistsMetric.
        """
        names = ['finSeeing', 'expMJD']
        types=[float,float]
        data = np.zeros(10,dtype=zip(names,types))
        data['finSeeing'] = [2.,2.,3.,1.,1.,1.,0.5,1.,0.4,1.]
        data['expMJD'] = np.arange(10)
        slicePoint = {'sid':0}
        # so here we have 4 images w/o good previous templates
        metric = metrics.TemplateExistsMetric()
        result = metric.run(data, slicePoint)
        assert(result == 6./10.)

    def testUniformityMetric(self):
        names = ['expMJD']
        types=[float]
        data = np.zeros(100, dtype=zip(names,types))
        metric = metrics.UniformityMetric()
        result1 = metric.run(data)
        # If all the observations are on the 1st day, should be 1
        assert(result1 == 1)
        data['expMJD'] = data['expMJD']+365.25*10
        slicePoint = {'sid':0}
        result2 = metric.run(data, slicePoint)
        # All on last day should also be 1
        assert(result1 == 1)
        # Make a perfectly uniform dist
        data['expMJD'] = np.arange(0.,365.25*10,365.25*10/100)
        result3 = metric.run(data, slicePoint)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.)
        # A single obseravtion should give a result of 1
        data = np.zeros(1, dtype=zip(names,types))
        result4 = metric.run(data, slicePoint)
        assert(result4 == 1)

if __name__ == '__main__':

    unittest.main()
