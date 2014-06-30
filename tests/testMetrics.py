import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics

class TestBaseMetric(unittest.TestCase):
    def testReduceDict(self):
        """Test that reduce dictionary is created."""
        testmetric = metrics.BaseMetric('testcol')
        self.assertEqual(testmetric.reduceFuncs.keys(), [])

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
        self.assertEqual(testmetric.units, '')
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
        self.assertEqual(testmetric.units, '')
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
        testmetric = metrics.BaseMetric(cols)
        self.assertTrue(isinstance(testmetric.plotParams, dict))
        self.assertEqual(testmetric.plotParams.keys(), ['units'])
        # Set some plot parameters - are they present in dictionary and dictionary contains only needed values?
        plotParams = {'title':'mytitle'}
        testmetric = metrics.BaseMetric(cols, plotParams=plotParams)
        self.assertTrue(isinstance(testmetric.plotParams, dict))
        self.assertEqual(set(testmetric.plotParams.keys()), set(['title', 'units']))
        
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
        np.testing.assert_equal(testmetric.run(self.dv), self.dv['testdata'])

    def testFracAboveMetric(self):
        cutoff = 5.1
        testmetric=metrics.FracAboveMetric('testdata', cutoff = cutoff)
        self.assertEqual(testmetric.run(self.dv), np.size(np.where(self.dv['testdata'] >= cutoff)[0])/float(np.size(self.dv)))

    def testFracBelowMetric(self):
        cutoff = 5.1
        testmetric=metrics.FracBelowMetric('testdata', cutoff = cutoff)
        self.assertEqual(testmetric.run(self.dv), np.size(np.where(self.dv['testdata'] <= cutoff)[0])/float(np.size(self.dv)))

    def testVisitGroups(self):
        """Test visit groups (solar system groups) metric."""
        # Set up some simple test data.
        tmin = 15.0/60./24.0
        tmax = 90./60./24.0
        tstart = 49406.00
        # The test data needs to have times for all observations: set up the times in two stages .. the night 
        night = np.array([0, 0, 0, 0, 0, 0, 
                          1, 1, 
                          2, 
                          3, 3, 
                          31, 31, 
                          32, 32, 
                          33, 33, 
                          34, 34, 34, 34,
                          35, 35, 35, 
                          36, 36, 36, 
                          37, 37, 37, 
                          38, 38, 38], 'int')
        #   .. and the time within the night.
        expmjd = np.array([tstart, tstart+tmin/10.0, tstart+tmin+tmin/2.0, tstart+tmin*2+tmax, tstart+2*tmax, tstart+2*tmax+tmin/2.0, #n0
                           tstart, tstart+tmax, # n1 .. only 2 obs but should make a pair
                           tstart, # n2 - only 1 obs, should not make a pair
                           tstart, tstart+tmin, # n3 .. only 2 obs but should make a pair
                           tstart, tstart+tmin, tstart, tstart+tmin, tstart, tstart+tmin, #n31/32/33 - a pair on each night
                           tstart, tstart+tmin/10.0, tstart+tmax*2, tstart+tmax*2+tmin/10.0, # n34 .. should make no pairs
                           tstart, tstart+tmin/10.0, tstart+tmax, # n35 should make 2.5 (too close at start)
                           tstart, tstart+tmax, tstart+tmax+tmin/10.0, #n36 should make 2.5 pairs (too close at end)
                           tstart, tstart+tmin, tstart+tmax, # n37 - 3 (regular 3)
                           tstart, tstart+tmax, tstart+tmax*2 # n38 - 3 (three, but long spacing)
                           ], 'float')
        expmjd = expmjd + night
        testdata = np.core.records.fromarrays([expmjd, night], names=['expmjd', 'night'])
        # Set up metric.
        testmetric = metrics.VisitGroupsMetric(timesCol='expmjd', nightsCol='night',
                                                deltaTmin=tmin, deltaTmax=tmax, minNVisits=2,
                                                window=5, minNNights=3)
        # and set up a copy, with a higher number of min visits per night 
        testmetric2 = metrics.VisitGroupsMetric(timesCol='expmjd', nightsCol='night', deltaTmin=tmin, deltaTmax=tmax, 
                                                minNVisits=3, window=5, minNNights=3)
        # Run metric for expected results.
        metricval = testmetric.run(testdata)
        # These are the expected results, based on the times above.
        expected_nights = np.array([0, 1, 3, 31, 32, 33, 35, 36, 37, 38])
        expected_numvisits = np.array([5.0, 2, 2, 2, 2, 2, 2.5, 2.5, 3, 3])
        np.testing.assert_equal(metricval['visits'], expected_numvisits)
        np.testing.assert_equal(metricval['nights'], expected_nights)
        # Test reduce methods.
        self.assertEqual(testmetric.reduceMedian(metricval), np.median(expected_numvisits))
        self.assertEqual(testmetric.reduceNNightsWithNVisits(metricval), len(expected_nights))
        self.assertEqual(testmetric2.reduceNNightsWithNVisits(metricval), 3)
        self.assertEqual(testmetric.reduceNVisitsInWindow(metricval), 11)
        self.assertEqual(testmetric2.reduceNNightsInWindow(metricval), 2)
        self.assertEqual(testmetric.reduceNLunations(metricval), 2)
        # Test with a longer (but simpler) date range.
        indnight = np.array([0, 1, 2, 3, 4, 5, 31, 32, 33, 34, 61, 62, 63, 121, 122, 123], 'int')
        indtimes = np.array([tstart, tstart+tmin, tstart+tmax], 'float')
        expmjd = []
        night = []
        for n in indnight:
            for t in indtimes:            
                expmjd.append(float(n + t))
                night.append(n)
        expmjd = np.array(expmjd)
        night = np.array(night)
        testdata = np.core.records.fromarrays([expmjd, night], names=['expmjd', 'night'])
        metricval = testmetric.run(testdata)
        self.assertEqual(testmetric.reduceNLunations(metricval), 4)
        self.assertEqual(testmetric.reduceMaxSeqLunations(metricval), 3)


if __name__ == "__main__":
    unittest.main()
