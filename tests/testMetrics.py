import numpy as np
import unittest
import lsst.sims.operations.maf.metrics as metrics

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
        # Set plot parameters - are they present in dictionary and dictionary contains only needed values?
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
    

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
