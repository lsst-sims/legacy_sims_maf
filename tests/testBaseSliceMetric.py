import numpy as np
import matplotlib
matplotlib.use('Agg')
import numpy.ma as ma
import warnings
import unittest
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.utils.tests as utilsTests

def makeDataValues(size=100, min=0., max=1., random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""    
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min()) 
    datavalues += min
    if random:
        randorder = np.random.rand(size)        
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    filtervalues = np.empty(size, dtype='str')
    half = int(size/2)
    filtervalues[0:half] = 'g'
    filtervalues[half:size] = 'r'
    ra = np.random.rand(size) * (360.0) * (np.pi/180.)
    dec = np.random.rand(size) * (-90.0) * (np.pi/180.)
    datavalues = np.core.records.fromarrays([datavalues, filtervalues, ra, dec],
                                            names=['testdata', 'filter', 'ra', 'dec'])    
    return datavalues


class TestSetupBaseSliceMetric(unittest.TestCase):
    """Unit tests relating to setting up the baseSliceMetric"""
    def setUp(self):
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotParams={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=50, r=50)
        self.slicer = slicers.UniSlicer()

    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.slicer = None
        
    def testInit(self):
        """Test init setup for baseSliceMetric."""
        # Test metric Name list set up and empty
        self.assertEqual(self.testbbm.metricNames, [])
        # Test dictionaries set up but empty
        self.assertEqual(self.testbbm.metricObjs.keys(), [])
        self.assertEqual(self.testbbm.metricValues.keys(), [])
        self.assertEqual(self.testbbm.plotParams.keys(), [])
        self.assertEqual(self.testbbm.simDataName.keys(), [])
        self.assertEqual(self.testbbm.sqlconstraint.keys(), [])
        self.assertEqual(self.testbbm.metadata.keys(), [])
        # Test that slicer is set to None
        self.assertEqual(self.testbbm.slicer, None)
        # Test that output file list is set to empty dict
        self.assertEqual(self.testbbm.outputFiles, {})
        # Test that figformat is set to default (pdf)
        self.assertEqual(self.testbbm.figformat, 'pdf')
        # Test that can set figformat to alternate value
        testbbm2 = sliceMetrics.BaseSliceMetric(figformat='eps')
        self.assertEqual(testbbm2.figformat, 'eps')

    def testSetSlicer(self):
        """Test setSlicer."""
        # Test can set slicer (when bbm slicer = None)
        self.testbbm.setSlicer(self.slicer)
        # Test can set/check slicer (when = previous slicer)
        slicer2 = slicers.UniSlicer()
        self.assertTrue(self.testbbm.setSlicer(slicer2, override=False))
        # Test can not set/override slicer (when != previous slicer)
        slicer2 = slicers.HealpixSlicer(nside=16, verbose=False)
        self.assertFalse(self.testbbm.setSlicer(slicer2, override=False))
        # Unless you really wanted to..
        self.assertTrue(self.testbbm.setSlicer(slicer2, override=True))

    def testSetMetrics(self):
        """Test setting metrics and de-dupe/dupe metric names."""
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])
        # Test metricNames list is as expected.
        self.assertEqual(self.testbbm.metricNames, ['Mean testdata', 'Count testdata', 'Completeness'])
        # Test that dictionaries for metricObjs (which hold metric python objects) set
        self.assertEqual(self.testbbm.metricObjs.keys(), ['Mean testdata', 'Count testdata', 'Completeness'])
        self.assertEqual(self.testbbm.metricObjs.values(), [self.m1, self.m2, self.m3])
        # Test that plot parameters were passed through as expected
        self.assertEqual(self.testbbm.plotParams.keys(), ['Mean testdata', 'Count testdata', 'Completeness'])
        self.assertEqual(self.testbbm.plotParams['Mean testdata'].keys(), ['units'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].keys(), ['units', 'title'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].values(),
                         ['countunits', 'count_title'])
        # Test that can set metrics using a single metric (not a list)
        testbbm2 = sliceMetrics.BaseSliceMetric()
        testbbm2.setMetrics(self.m1)
        self.assertEqual(testbbm2.metricNames, ['Mean testdata',])
        # Test that if add an additional metric, the name is 'de-duped' as expected (and added)
        m4 = metrics.MeanMetric('testdata')
        testbbm2.setMetrics(m4)
        self.assertEqual(testbbm2.metricNames, ['Mean testdata', 'Mean testdata__0'])
        # And that we can de-dupe name as expected.
        self.assertEqual(testbbm2._dupeMetricName('Mean testdata__0'), 'Mean testdata')
        
    def testValidateMetricData(self):
        """Test validation of metric data."""
        dv = makeDataValues()
        # Test that validates correctly if all columns present in data. 
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])
        self.assertTrue(self.testbbm.validateMetricData(dv))
        # Test that raises exception if not all columns present in data.
        m4 = metrics.MeanMetric('notTestData')
        self.testbbm.setMetrics(m4)
        self.assertRaises(Exception, self.testbbm.validateMetricData, dv)

class TestRunBaseSliceMetric(unittest.TestCase):        
    def setUp(self):
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotParams={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=500, r=500,
                                             plotParams={'xlabel':'Completeness'})
        self.metricNames = ['Mean testdata', 'Count testdata', 'Completeness']
        self.reduceNames = ['Completeness_u', 'Completeness_g', 'Completeness_r', 'Completeness_i',
                            'Completeness_z', 'Completeness_y', 'Completeness_Joint']
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.slicer = slicers.OneDSlicer('testdata', bins=np.arange(0, 1.25, .1))
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])

    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.slicer = None

    def testRunBins(self):
        """Test creating metric data values."""
        opsimname = 'opsim1000'
        sqlconstraint = 'created fake testdata'
        metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=opsimname, sqlconstraint=sqlconstraint, metadata=metadata)
        # Test that copied opsim name and sqlconstraint and metadata correctly for each metric name.
        for mname in self.metricNames:
            self.assertEqual(self.testbbm.simDataName[mname], opsimname)
            self.assertEqual(self.testbbm.sqlconstraint[mname], sqlconstraint)
            self.assertEqual(self.testbbm.metadata[mname], metadata)
        # Test that created metric data with expected number of data points.
        for mname in self.metricNames:
            self.assertEqual(len(self.testbbm.metricValues[mname]), len(self.slicer))
        # Test that metric data was masked where expected (last bin) due to no data in bin.
        lastbin = len(self.slicer) - 1
        for mname in self.metricNames:
            self.assertEqual(self.testbbm.metricValues[mname].mask[lastbin], True)

    def testReduce(self):
        """Test running reduce methods."""
        # Completeness metric has reduce methods, so check on those.
        opsimname = 'opsim1000'
        sqlconstraint = 'created fake testdata'
        metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=opsimname, sqlconstraint=sqlconstraint, metadata=metadata)
        self.testbbm.reduceAll()
        # Check that all metric data values expected exist.
        for m in self.metricNames:
            self.assertTrue(m in self.testbbm.metricValues.keys())
        for m in self.reduceNames:
            self.assertTrue(m in self.testbbm.metricValues.keys())
        # Check that simdata, sqlconstraint and metadatas were copied properly.
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.simDataName[m], opsimname)
            self.assertEqual(self.testbbm.sqlconstraint[m], sqlconstraint)
            self.assertEqual(self.testbbm.metadata[m], metadata)
        # Check that plot parameters were copied properly.
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.plotParams[m]['xlabel'], 'Completeness')
        # Check that mask carried through properly.
        lastbin = len(self.slicer) - 1
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.metricValues[m].mask[lastbin], True)
                

class TestReadWriteBaseSliceMetric(unittest.TestCase):        
    def setUp(self):
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotParams={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=500, r=500,
                                             plotParams={'xlabel':'Completeness'})
        self.metricNames = ['Mean testdata', 'Count testdata', 'Completeness']
        self.reduceNames = ['Completeness_u', 'Completeness_g', 'Completeness_r', 'Completeness_i',
                            'Completeness_z', 'Completeness_y', 'Completeness_Joint']
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.slicer = slicers.OneDSlicer('testdata', bins=np.arange(0, 1.25, .1))
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])
        self.opsimname = 'opsim1000'
        self.sqlconstraint = 'created fake testdata'
        self.metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        self.testbbm.reduceAll()
        self.outroot = 'testBaseSliceMetric'
        self.testbbm.writeAll(outDir='.', outfileRoot=self.outroot)
        self.expectedfiles = []
        for m in self.metricNames:
            filename = (self.outroot + ' ' + m + ' ' + self.metadata + ' ' +
                        self.slicer.slicerName[:4].upper() + '.npz')
            filename = filename.replace(' ', '_')
            self.expectedfiles.append(filename)
        for m in self.reduceNames:
            filename = (self.outroot + ' ' + m + ' ' + self.metadata + ' ' +
                        self.slicer.slicerName[:4].upper() + '.npz')
            filename = filename.replace(' ', '_')
            self.expectedfiles.append(filename)
                        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.slicer = None
        import os
        for f in self.expectedfiles:
            os.remove(f)        

    def testWrite(self):
        """Test writing data to disk (and test outfile name generation)."""
        import os
        filelist = os.listdir('.')
        for f in self.expectedfiles:
            self.assertTrue(f in filelist)

    def testRead(self):
        """Test reading data back from disk. """
        # Test with slicer already set up in sliceMetric.
        filename = (self.outroot + '_' + 'Completeness' + '_' + self.metadata + '_' +
                    self.slicer.slicerName[:4].upper() + '.npz')
        filename = filename.replace(' ', '_')
        self.testbbm.readMetricValues(filename)
        # Should be read in and de-duped.
        self.assertTrue('Completeness__0' in self.testbbm.metricValues)
        for m, n in zip(self.testbbm.metricValues['Completeness'].data,
                        self.testbbm.metricValues['Completeness__0'].data):
            np.testing.assert_equal(m, n)
        for m, n in zip(self.testbbm.metricValues['Completeness'].mask,
                        self.testbbm.metricValues['Completeness__0'].mask):
            self.assertEqual(m, n)
        # Test with new sliceMetric (with no slicer previously set up).
        testbbm2 = sliceMetrics.BaseSliceMetric()
        testbbm2.readMetricValues(filename)
        self.assertTrue('Completeness' in testbbm2.metricValues)
        for m, n in zip(self.testbbm.metricValues['Completeness'].data,
                        testbbm2.metricValues['Completeness'].data):
            np.testing.assert_equal(m, n)
        for m, n in zip(self.testbbm.metricValues['Completeness'].mask,
                        testbbm2.metricValues['Completeness'].mask):
            self.assertEqual(m, n)
        # Check if simDataName and sqlconstraint were restored as expected.
        self.assertEqual(testbbm2.simDataName['Completeness'], self.testbbm.simDataName['Completeness'])
        self.assertEqual(testbbm2.sqlconstraint['Completeness'], self.testbbm.sqlconstraint['Completeness'])
        # plot parameters not currently being written to disk
        #self.assertEqual(testbbm2.plotParams['Completeness']['xlabel'], 'Completeness')

    def testOutputFileKey(self):
        """Test that the output file dict is being generated as expected."""
        outkeys = self.testbbm.returnOutputFiles(verbose=False)
        # Check data in outkeys is valid
        for o in outkeys:
            self.assertEqual(outkeys[o]['metadata'], self.metadata)
            self.assertEqual(outkeys[o]['simDataName'], self.opsimname)
            self.assertEqual(outkeys[o]['sqlconstraint'], self.sqlconstraint)
            self.assertTrue(outkeys[o]['dataFile'].replace('./', '') in self.expectedfiles)
            self.assertEqual(outkeys[o]['slicerName'], self.slicer.slicerName)
            self.assertTrue((outkeys[o]['metricName'] in self.metricNames) or
                            (outkeys[o]['metricName'] in self.reduceNames)) 
        # Check data in outkeys is complete
        outkeysMetricNames = []
        for o in outkeys:
            outkeysMetricNames.append(o)
        for m in self.metricNames:
            self.assertTrue(m in outkeysMetricNames)
        for m in self.reduceNames:
            self.assertTrue(m in outkeysMetricNames)

class TestSummaryStatisticBaseSliceMetric(unittest.TestCase):
    def setUp(self):
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.testbbm.setMetrics([self.m1,])
        self.summaryStat = metrics.MeanMetric('metricData')
        self.opsimname = 'opsim1000'
        self.sqlconstraint = 'created fake testdata'
        self.metadata = 'testing fake data run'
        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.slicer = None

    def testSummaryStatistic(self):
        """Test summary statistic calculation."""
        # Try unislicer first: expect that summary statistic return will be simply the unislicer value.
        self.slicer = slicers.UniSlicer()
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'][0])
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', metrics.IdentityMetric('metricdata'))
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'][0])
        # Try oneD slicer: other slicers should behave similarly.
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.testbbm.setMetrics([self.m1,])
        self.slicer = slicers.OneDSlicer('testdata', bins=100)
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'].mean())
        # Test get warning if calculating summary statistics on 'object' data using simple scalar metric.
        fakemetricdata = ma.MaskedArray(data = np.empty(len(self.slicer), 'object'),
                                        mask = np.zeros(len(self.slicer), 'bool'),
                                        fill_value = self.slicer.badval)
        self.testbbm.metricValues['objecttest'] = fakemetricdata
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            summary = self.testbbm.computeSummaryStatistics('objecttest', self.summaryStat)
            self.assertTrue('objecttest' in str(w[-1].message))
            self.assertEqual(summary, None)
                            
        
class TestPlottingBaseSliceMetric(unittest.TestCase):
    def setUp(self):
        # Set up dictionary of all plotting parameters to test.
        self.plotParams = {'units': 'testunits',
                        'title': 'my test title',  # plot titles
                        'xlabel': 'my xlabel',  # plot x labels
                        'ylabel': 'my ylabel',  # plot y labels
                        # For 1-d slicer: set x min/max vals via bins OR by xMin/xMax, then y vals via yMin/yMax
                        # For spatial slicers: set hist x min/max vals via xMin/Max & number of bins via 'bins'
                        #   then for skymap, set colorbar min/max vals via xMin/xMax
                        'yMin': -0.5,
                        'yMax': 1.5,
                        'xMin': -0.5,  # histogram x minimum value for spatial slicer
                        'xMax': 1.5,   # histogram x maximum value for spatial slicer
                        # No way to set y value limits for spatial slicer histogram?
                        'bins': 50       # parameter for number of bins for spatial slicer histograms
                        }
        self.m1 = metrics.MeanMetric('testdata', metricName='Test labels', plotParams = self.plotParams)
        self.m2 = metrics.MeanMetric('testdata', metricName='Test defaults')
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.opsimname = 'opsim1000'
        self.sqlconstraint = 'created fake testdata'
        self.metadata = 'testing fake data run'
                        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.slicer = None

    def testPlotting(self):        
        """Test plotting."""
        import matplotlib.pyplot as plt    
        # Test OneDSlicer.
        bins = np.arange(0, 1.25, .1)
        self.slicer = slicers.OneDSlicer('testdata', bins=bins)
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.testbbm.setMetrics([self.m1, self.m2])
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname,
                               sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        # Test plotting oneDslicer, where we've set the plot parameters.
        fignums = self.testbbm.plotMetric(self.m1.name, savefig=False)
        fig = plt.figure(fignums['BinnedData'])
        ax = plt.gca()
        # Check x and y limits set from plot params.
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (self.plotParams['xMin'], self.plotParams['xMax']))
        ylims = plt.ylim()
        np.testing.assert_almost_equal(ylims, (self.plotParams['yMin'], self.plotParams['yMax']))
        # Check x and y labels
        self.assertEqual(ax.get_xlabel(), self.plotParams['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotParams['ylabel'])
        # Check title
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        # Test a spatial slicer.
        self.testbbm = sliceMetrics.BaseSliceMetric()
        self.testbbm.setMetrics([self.m1, ])
        self.slicer = slicers.HealpixSlicer(nside=4, spatialkey1='ra', spatialkey2='dec', verbose=False)
        self.slicer.setupSlicer(self.dv)
        self.testbbm.setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname,
                               sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        fignums = self.testbbm.plotMetric(self.m1.name, savefig=False)
        # Test histogram.
        fig = plt.figure(fignums['Histogram'])
        ax = plt.gca()
        # Check x limits.
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (self.plotParams['xMin'], self.plotParams['xMax']))
        # Check x and y labels.
        self.assertEqual(ax.get_xlabel(), self.plotParams['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotParams['ylabel'])
        # Check title.
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        # Test sky map.
        fig = plt.figure(fignums['SkyMap'])
        ax = plt.gca()
        # Not sure how to check clims of color bar.
        # Check title.
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        
def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TestSetupBaseSliceMetric)
    suites += unittest.makeSuite(TestRunBaseSliceMetric)
    suites += unittest.makeSuite(TestReadWriteBaseSliceMetric)
    suites += unittest.makeSuite(TestSummaryStatisticBaseSliceMetric)
    suites += unittest.makeSuite(TestPlottingBaseSliceMetric)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
