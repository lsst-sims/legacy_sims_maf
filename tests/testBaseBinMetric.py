import numpy as np
import unittest
import lsst.sims.operations.maf.binMetrics as binMetrics
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binners as binners

class TestSetupBaseBinMetric(unittest.TestCase):
    """Unit tests relating to setting up the baseBinMetric"""
    def setUp(self):
        self.testbbm = binMetrics.BaseBinMetric()
        self.m1 = metrics.MeanMetric('testdata', plotParams={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', plotParams={'units':'countunits',
                                                            'title':'count_title'})
        self.binner = binners.UniBinner()

    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.binner
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.binner = None
        
    def testInit(self):
        """Test init setup for baseBinMetric."""
        # Test metric Name list set up and empty
        self.assertEqual(self.testbbm.metricNames, [])
        # Test dictionaries set up but empty
        self.assertEqual(self.testbbm.metricObjs.keys(), [])
        self.assertEqual(self.testbbm.metricValues.keys(), [])
        self.assertEqual(self.testbbm.plotParams.keys(), [])
        self.assertEqual(self.testbbm.simDataName.keys(), [])
        self.assertEqual(self.testbbm.metadata.keys(), [])
        self.assertEqual(self.testbbm.comment.keys(), [])
        # Test that binner is set to None
        self.assertEqual(self.testbbm.binner, None)
        # Test that output file list is set to empty list
        self.assertEqual(self.testbbm.outputFiles, [])
        # Test that figformat is set to default (png)
        self.assertEqual(self.testbbm.figformat, 'png')
        # Test that can set figformat to alternate value
        testbbm2 = binMetrics.BaseBinMetric(figformat='eps')
        self.assertEqual(testbbm2.figformat, 'eps')

    def testSetBinner(self):
        """Test setBinner."""
        # Test can set binner (when bbm binner = None)
        self.testbbm.setBinner(self.binner)
        # Test can set/check binner (when = previous binner)
        binner2 = binners.UniBinner()
        self.assertTrue(self.testbbm.setBinner(binner, override=False))
        # Test can not set/override binner (when != previous binner)
        binner2 = binners.HealpixBinner(nside=16, verbose=False)
        self.assertFalse(self.testbbm.setBinner(binner2, override=False))
        # Unless you really wanted to..
        self.assertTrue(self.testbbm.setBinner(binner2, override=True))

    def testSetMetrics(self):
        """Test setting metrics."""
        self.testbbm.setMetrics([self.m1, self.m2])
        # Test metricNames list is as expected.
        self.assertEqual(self.testbbm.metricNames, ['Mean testdata', 'Count testdata'])
        # Test that dictionaries for metricObjs (which hold metric python objects) set
        self.assertEqual(self.testbbm.metricObjs.keys(), ['Mean testdata', 'Count testdata'])
        self.assertEqual(self.testbbm.metricObjs.values(), [self.m1, self.m2])
        # Test that plot parameters were passed through as expected
        self.assertEqual(self.testbbm.plotParams.keys(), ['Mean testdata', 'Count testdata'])
        self.assertEqual(self.testbbm.plotParams['Mean testdata'].keys(), ['units', '_unit'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].keys(), ['units', '_unit', 'title'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].values(),
                         ['countunits', 'Count testdata', 'count_title'])
        # Test that can set metrics using a single metric (not a list)
        testbbm2 = binMetrics.BaseBinMetric()
        testbbm2.setMetrics(self.m1)
        self.assertEqual(testbbm2.metricNames, ['Mean testdata',])
        # Test that if add an additional metric, the name is 'de-duped' as expected (and added)
        m3 = metrics.MeanMetric('testdata')
        testbbm2.setMetrics(m3)
        self.assertEqual(testbbm2.metricNames, ['Mean testdata', 'Mean testdata__0'])
        
    
        
        

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSetupBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)

        
