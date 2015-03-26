import matplotlib
matplotlib.use("Agg")
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


    def testTGapMetric(self):
        names = ['expMJD']
        types=[float]
        data = np.zeros(100, dtype=zip(names,types))
        # All 1-day gaps
        data['expMJD'] = np.arange(100)

        metric = metrics.Tgaps(binsize=1)
        result1 = metric.run(data)
        # By default, should all be in first bin
        assert(result1[0] == data.size-1)
        assert(np.sum(result1) == data.size-1)
        data['expMJD'] = np.arange(0,200,2)
        result2 =  metric.run(data)
        assert(result2[1] == data.size-1)
        assert(np.sum(result2) == data.size-1)

        metric = metrics.Tgaps(allGaps=True, binMax=200, binsize=1)
        result3 =  metric.run(data)
        assert(result3[1] == data.size-1)
        Ngaps = (data.size-1)*(data.size-1)/2.+(data.size-1)/2.
        assert(np.sum(result3) == Ngaps)

    def testRapidRevisitMetric(self):
        data = np.zeros(100, dtype=zip(['expMJD'], [float]))
        # Uniformly distribute time _differences_ between 0 and 100
        dtimes = np.arange(100)
        data['expMJD'] = dtimes.cumsum()
        # Set up "rapid revisit" metric to look for visits between 5 and 25
        metric = metrics.RapidRevisitMetric(dTmin=5, dTmax=55, minNvisits=50)
        result = metric.run(data)
        # This should be uniform.
        self.assertTrue(result < 0.1)
        self.assertTrue(result >= 0)
        # Set up non-uniform distribution of time differences
        dtimes = np.zeros(100) + 5
        data['expMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertTrue(result >= 0.5)
        dtimes = np.zeros(100) + 15
        data['expMJD'] = dtimes.cumsum()
        result = metric.run(data)
        self.assertTrue(result >= 0.5)
        # Let's see how much dmax/result can vary
        resmin = 1
        resmax = 0
        for i in range(10000):
            dtimes = np.random.rand(100)
            data['expMJD'] = dtimes.cumsum()
            metric = metrics.RapidRevisitMetric(dTmin=0.1, dTmax=0.8, minNvisits=50)
            result = metric.run(data)
            resmin = np.min([resmin, result])
            resmax = np.max([resmax, result])
        print "RapidRevisit .. range", resmin, resmax

    def testNRevisitsMetric(self):
        data = np.zeros(100, dtype=zip(['expMJD'], [float]))
        dtimes = np.arange(100)/24./60.
        data['expMJD'] = dtimes.cumsum()
        metric = metrics.NRevisitsMetric(dT=50.)
        result = metric.run(data)
        self.assertEqual(result, 50)
        metric = metrics.NRevisitsMetric(dT=50., normed=True)
        result = metric.run(data)
        self.assertEqual(result, 0.5)
        
if __name__ == '__main__':

    unittest.main()
