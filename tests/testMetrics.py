import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics

class TestBaseMetric(unittest.TestCase):
    def testMetricName(self):
        """Test that metric name is set appropriately automatically and explicitly"""
        # Test automatic setting of metric name
        testmetric = metrics.BaseMetric('testcol')
        self.assertEqual(testmetric.name, 'Base testcol')
        testmetric = metrics.BaseMetric(['testcol1', 'testcol2'])
        self.assertEqual(testmetric.name, 'Base testcol1, testcol2')
        # Test explicit setting of metric name
        testmetric = metrics.BaseMetric('testcol', metricName='Test')
        self.assertEqual(testmetric.name, 'Test')    
                        
    def testColRegistry(self):
        """Test column registry adds to class registry dictionary as expected"""
        cols = 'onecolumn'
        testmetric = metrics.BaseMetric(cols)
        # Class registry should have dictionary with key = metric class name
        self.assertEqual(testmetric.classRegistry.keys(), ['BaseMetric',])
        # Class registry should have dictionary with values = set of columns for metric class
        self.assertEqual(testmetric.classRegistry['BaseMetric'], set(['onecolumn']))
        cols = ['onecolumn', 'twocolumn']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.classRegistry['BaseMetric'], set(['onecolumn', 'twocolumn']))
        # Test with explicit metric name.
        testmetric = metrics.BaseMetric(cols, metricName='test')
        self.assertEqual(testmetric.classRegistry.keys(), ['BaseMetric',])
        # Test with additional (different) metric
        cols = 'twocolumn'
        testmetric2 = metrics.SimpleScalarMetric(cols)
        self.assertEqual(testmetric.classRegistry.keys(), ['SimpleScalarMetric', 'BaseMetric'])
        self.assertEqual(testmetric.classRegistry['SimpleScalarMetric'], set(['twocolumn']))
        # Test uniqueCols method to retrieve unique data columns from class registry
        self.assertEqual(testmetric.classRegistry.uniqueCols(), set(['onecolumn', 'twocolumn']))

    def testMetricDtype(self):
        """Test that base metric data value type set appropriately"""
        cols = 'onecolumn'
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.metricDtype, 'object')
    
    def testUnits(self):
        """Test unit setting (including units set by utils.getColInfo)"""
        cols = 'onecolumn'
        # Test for column not in colInfo, units not set by hand.
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, testmetric.name)
        # Test for column not in colInfo, units set by hand.
        testmetric = metrics.BaseMetric(cols, units='Test')
        self.assertEqual(testmetric.units, 'Test')
        # Test for column in colInfo (looking up units in colInfo)
        cols = 'seeing'
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec')
        # Test for column in colInfo but units overriden
        testmetric = metrics.BaseMetric(cols, units='Test')
        self.assertEqual(testmetric.units, 'Test')
        # Test for multiple columns not in colInfo
        cols = ['onecol', 'twocol']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, testmetric.name)
        # Test for multiple columns in colInfo
        cols = ['seeing', 'skybrightness_modified']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec mag/sq arcsec')
        # Test for multiple columns, only one in colInfo
        cols = ['seeing', 'twocol']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec ')
        # Test for multiple columns both in colInfo but repeated
        cols = ['seeing', 'seeing']
        testmetric = metrics.BaseMetric(cols)
        self.assertEqual(testmetric.units, 'arcsec arcsec')

    def testPlotParams(self):
        """Test plot parameter setting"""
        cols = 'onecolumn'
        # Don't set any plot parameters - is dictionary present and contains only _unit?
        #  (_unit is default unit)
        testmetric = metrics.BaseMetric(cols)
        self.assertTrue(isinstance(testmetric.plotParams, dict))
        self.assertEqual(testmetric.plotParams.keys(), ['_unit'])
        # Set some plot parameters - are they present in dictionary and dictionary contains only needed values?
        plotParams = {'title':'mytitle'}
        testmetric = metrics.BaseMetric(cols, plotParams=plotParams)
        self.assertTrue(isinstance(testmetric.plotParams, dict))
        self.assertEqual(set(testmetric.plotParams.keys()), set(['title', '_unit']))
        
    def testValidateData(self):
        """Test 'validateData' method"""
        testdata = np.zeros(10, dtype=[('testdata', 'float')])
        testmetric = metrics.BaseMetric(cols='testdata')
        self.assertTrue(testmetric.validateData(testdata))
        testmetric = metrics.BaseMetric(cols='nottestdata')
        self.assertRaises(ValueError, testmetric.validateData, testdata)

class TestSimpleMetrics(unittest.TestCase):
    def setUp(self):
        dv = np.arange(0, 10, .5)
        self.dv = np.array(zip(dv), dtype=[('testdata', 'float')])
        
    def testBaseSimpleScalar(self):
        """Test base simple scalar metric."""
        # Check that metric works as expected with single column.
        testmetric = metrics.SimpleScalarMetric(colname='testdata')
        self.assertEqual(testmetric.metricDtype, 'float')
        self.assertEqual(testmetric.colname, 'testdata')
        # Check that metric raises exception if given more than one column.
        self.assertRaises(Exception, metrics.SimpleScalarMetric, ['testdata1', 'testdata2'])

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
        testmetric = metrics.Coaddm5Metric(m5col='testdata')
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

    def testIdentityMetric(self):
        """Test identity metric."""
        testmetric = metrics.IdentityMetric('testdata')
        self.assertEqual(testmetric.run(self.dv[0]), self.dv[0]['testdata'])

class TestComplexMetrics(unittest.TestCase):
    def testBaseComplex(self):
        """Test the base class for complex metrics."""
        testmetric = metrics.ComplexMetric(cols='testdata')
        # Test that 'reduceFuncs' dictionary set up (although will be empty for 'base')
        self.assertEqual(testmetric.reduceFuncs.keys(), [])

    def testVisitPairs(self):
        """Test visit pairs metric."""
        # Set up some simple test data.
        tmin = 15.0/60./24.0
        tmax = 90./60./24.0
        tstart = 49406.00
        night = np.array([0, 0, 0, 0, 1, 1, 2, 3, 3], 'int')
        expmjd = np.array([tstart, tstart+tmin+tmin/10.0, tstart+tmin+tmin/2.0, tstart+tmax,
                           tstart, tstart+tmax, tstart, tstart, tstart+tmin], 'float')
        expmjd = expmjd + night
        testdata = np.core.records.fromarrays([expmjd, night], names=['expmjd', 'night'])
        # Set up metric.
        testmetric = metrics.VisitPairsMetric(timesCol='expmjd', nightsCol='night',
                                            deltaTmin=tmin, deltaTmax=tmax, nPairs=2, window=30)
        # Run metric for expected results.
        numpairs, nightpairs = testmetric.run(testdata)
        expected_nightpairs = np.array([0, 1, 3])
        expected_numpairs = np.array([5, 1, 1])
        np.testing.assert_equal(nightpairs, expected_nightpairs)
        np.testing.assert_equal(numpairs, expected_numpairs)
        # Test reduce methods.
        self.assertEqual(testmetric.reduceMedian((numpairs, nightpairs)), np.median(expected_numpairs))
        self.assertEqual(testmetric.reduceMean((numpairs, nightpairs)), np.mean(expected_numpairs))
        self.assertEqual(testmetric.reduceRms((numpairs, nightpairs)), np.std(expected_numpairs))
        self.assertEqual(testmetric.reduceNNightsWithPairs((numpairs, nightpairs)), len(expected_nightpairs))
        self.assertEqual(testmetric.reduceNNightsWithPairs((numpairs, nightpairs), nPairs=3), 1)
        self.assertEqual(testmetric.reduceNPairsInWindow((numpairs, nightpairs)), 7)
        self.assertEqual(testmetric.reduceNPairsInWindow((numpairs, nightpairs), window=2), 6)
        self.assertEqual(testmetric.reduceNNightsInWindow((numpairs, nightpairs)), 3)
        self.assertEqual(testmetric.reduceNNightsInWindow((numpairs, nightpairs), window=2), 2)

                                
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSimpleMetrics)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComplexMetrics)
    unittest.TextTestRunner(verbosity=2).run(suite)
