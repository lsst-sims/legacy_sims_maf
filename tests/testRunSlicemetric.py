import matplotlib
matplotlib.use("Agg")
import os
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


class TestSetupRunSliceMetric(unittest.TestCase):
    """Unit tests relating to setting up the baseSliceMetric"""
    def setUp(self):
        self.outDir = '.'
        self.figformat = 'pdf'
        self.dpi = 500
        self.testbbm = sliceMetrics.RunSliceMetric(outDir=self.outDir, figformat=self.figformat, dpi=self.dpi)
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotDict={'units':'meanunits'})
        self.m1iid = 0
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotDict={'units':'countunits',
                                                  'cbarFormat':'%d',
                                                  'title':'count_title'})
        self.m2iid = 1
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=50, r=50)
        self.m3iid = 2
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
        os.remove('resultsDb_sqlite.db')

    def testInit(self):
        """Test init setup for baseSliceMetric."""
        # Test dictionaries set up and empty.
        self.assertEqual(self.testbbm.metricNames.keys(), [])
        self.assertEqual(self.testbbm.metricObjs.keys(), [])
        self.assertEqual(self.testbbm.metricValues.keys(), [])
        self.assertEqual(self.testbbm.plotDicts.keys(), [])
        self.assertEqual(self.testbbm.simDataNames.keys(), [])
        self.assertEqual(self.testbbm.sqlconstraints.keys(), [])
        self.assertEqual(self.testbbm.metadatas.keys(), [])
        # Test that slicer is set to None
        self.assertEqual(self.testbbm.slicer, None)
        # Test that figformat is set
        self.assertEqual(self.testbbm.figformat, self.figformat)
        # Test that can set figformat to alternate value
        testbbm2 = sliceMetrics.RunSliceMetric(outDir='.', figformat='eps')
        self.assertEqual(testbbm2.figformat, 'eps')

    def testSetSlicer(self):
        """Test setSlicer."""
        # Test can set slicer (when bbm slicer = None)
        self.testbbm._setSlicer(self.slicer)
        # Test can set/check slicer (when = previous slicer)
        slicer2 = slicers.UniSlicer()
        self.assertTrue(self.testbbm._setSlicer(slicer2, override=False))
        # Test can not set/override slicer (when != previous slicer)
        slicer2 = slicers.HealpixSlicer(nside=16, verbose=False)
        self.assertFalse(self.testbbm._setSlicer(slicer2, override=False))
        # Unless you really wanted to..
        self.assertTrue(self.testbbm._setSlicer(slicer2, override=True))

    def testSetMetrics(self):
        """Test setting metrics."""
        self.testbbm._setMetrics([self.m1, self.m2, self.m3])
        # Test metricNames list is as expected.
        self.assertEqual(set(self.testbbm.metricNames.values()),
                         set(['Mean testdata', 'Count testdata', 'Completeness']))
        # Test that dictionaries for metricObjs (which hold metric python objects) set
        self.assertEqual(self.testbbm.metricObjs.values(), [self.m1, self.m2, self.m3])
        # Test that plot parameters were passed through as expected
        self.assertEqual(self.testbbm.plotDicts[self.m1iid].keys(), ['units'])
        self.assertEqual(self.testbbm.plotDicts[self.m2iid].keys(), ['units', 'cbarFormat', 'title'])
        self.assertEqual(self.testbbm.plotDicts[self.m2iid].values(),
                         ['countunits', '%d', 'count_title'])
        # Test that can set metrics using a single metric (not a list)
        testbbm2 = sliceMetrics.RunSliceMetric(outDir='.')
        testbbm2._setMetrics(self.m1)
        self.assertEqual(testbbm2.metricNames.values(), ['Mean testdata',])
        # Test that we can then add another metric.
        m4 = metrics.MeanMetric('testdata')
        testbbm2._setMetrics(m4)
        self.assertEqual(testbbm2.metricNames.values(), ['Mean testdata', 'Mean testdata'])

    def testValidateMetricData(self):
        """Test validation of metric data."""
        dv = makeDataValues()
        # Test that validates correctly if all columns present in data.
        self.testbbm._setMetrics([self.m1, self.m2, self.m3])
        self.assertTrue(self.testbbm.validateMetricData(dv))
        # Test that raises exception if not all columns present in data.
        m4 = metrics.MeanMetric('notTestData')
        self.testbbm._setMetrics(m4)
        self.assertRaises(Exception, self.testbbm.validateMetricData, dv)

class TestRunRunSliceMetric(unittest.TestCase):
    def setUp(self):
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotDict={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotDict={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=500, r=500,
                                             plotDict={'xlabel':'Completeness'})
        self.metricNames = ['Mean testdata', 'Count testdata', 'Completeness']
        self.iids = [0, 1, 2]
        self.reduceNames = ['Completeness_u', 'Completeness_g', 'Completeness_r', 'Completeness_i',
                            'Completeness_z', 'Completeness_y', 'Completeness_Joint']
        self.riids = [3, 4, 5, 6, 7, 8, 9]
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.slicer = slicers.OneDSlicer('testdata', bins=np.arange(0, 1.25, .1))
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm._setMetrics([self.m1, self.m2, self.m3])

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
        os.remove('resultsDb_sqlite.db')

    def testRunSlices(self):
        """Test creating metric data values."""
        opsimname = 'opsim1000'
        sqlconstraint = 'created fake testdata'
        metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=opsimname, sqlconstraint=sqlconstraint, metadata=metadata)
        # Test that copied opsim name and sqlconstraint and metadata correctly for each metric name.
        for iid in self.iids:
            self.assertEqual(self.testbbm.metricNames[iid], self.metricNames[iid])
            self.assertEqual(self.testbbm.simDataNames[iid], opsimname)
            self.assertEqual(self.testbbm.sqlconstraints[iid], sqlconstraint)
            self.assertEqual(self.testbbm.metadatas[iid], metadata)
        # Test that created metric data with expected number of data points.
        for iid in self.iids:
            self.assertEqual(len(self.testbbm.metricValues[iid]), len(self.slicer))
        # Test that metric data was masked where expected (last slicePoint) due to no data in slice.
        lastslice = len(self.slicer) - 1
        for iid in self.iids:
            self.assertEqual(self.testbbm.metricValues[iid].mask[lastslice], True)

    def testReduce(self):
        """Test running reduce methods."""
        # Completeness metric has reduce methods, so check on those.
        opsimname = 'opsim1000'
        sqlconstraint = 'created fake testdata'
        metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=opsimname, sqlconstraint=sqlconstraint, metadata=metadata)
        self.testbbm.reduceAll()
        # Check that all expected metric names exist.
        for mname in self.metricNames:
            self.assertTrue(mname in self.testbbm.metricNames.values())
        for rname in self.reduceNames:
            self.assertTrue(rname in self.testbbm.metricNames.values())
        # Check that simdata, sqlconstraint and metadatas were copied properly.
        for riid in self.riids:
            self.assertEqual(self.testbbm.simDataNames[riid], opsimname)
            self.assertEqual(self.testbbm.sqlconstraints[riid], sqlconstraint)
            self.assertEqual(self.testbbm.metadatas[riid], metadata)
        # Check that plot parameters were copied properly.
        for riid in self.riids:
            self.assertEqual(self.testbbm.plotDicts[riid]['xlabel'], 'Completeness')
        # Check that mask carried through properly.
        lastslice = len(self.slicer) - 1
        for riid in self.riids:
            self.assertEqual(self.testbbm.metricValues[riid].mask[lastslice], True)


class TestReadWriteRunSliceMetric(unittest.TestCase):
    def setUp(self):
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotDict={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotDict={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=500, r=500,
                                             plotDict={'xlabel':'Completeness'})
        self.metricNames = ['Mean testdata', 'Count testdata', 'Completeness']
        self.iids = [0, 1, 2]
        self.reduceNames = ['Completeness_u', 'Completeness_g', 'Completeness_r', 'Completeness_i',
                            'Completeness_z', 'Completeness_y', 'Completeness_Joint']
        self.riids = [3, 4, 5, 6, 7, 8, 9]
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.slicer = slicers.OneDSlicer('testdata', bins=np.arange(0, 1.25, .1))
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm._setMetrics([self.m1, self.m2, self.m3])
        self.opsimname = 'opsim1000'
        self.sqlconstraint = 'created fake testdata'
        self.metadata = 'testing fake data run'
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint,
                               metadata=self.metadata)
        self.testbbm.reduceAll()
        self.outroot = 'testRunSliceMetric'
        self.testbbm.writeAll(outfileRoot=self.outroot)
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
        self.expectedfiles.append('resultsDb_sqlite.db')

    def tearDown(self):
        self.testbbm.resultsDb.close()
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.slicer
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.slicer = None
        for f in self.expectedfiles:
            os.remove(f)

    def testWrite(self):
        """Test writing data to disk (and test outfile name generation)."""
        filelist = os.listdir('.')
        for f in self.expectedfiles:
            self.assertTrue(f in filelist)

    def testRead(self):
        """Test reading data back from disk. """
        # Test with slicer already set up in sliceMetric.
        filename = (self.outroot + '_' + 'Completeness' + '_' + self.metadata + '_' +
                    self.slicer.slicerName[:4].upper() + '.npz')
        filename = filename.replace(' ', '_')
        oldiid = self.testbbm.findIids(metricName='Completeness')[0]
        newiid = self.testbbm.iid_next
        self.testbbm.readMetricData(filename)
        # Should be read in.
        for m, n in zip(self.testbbm.metricValues[oldiid].data,
                        self.testbbm.metricValues[newiid].data):
            np.testing.assert_equal(m, n)
        for m, n in zip(self.testbbm.metricValues[oldiid].mask,
                        self.testbbm.metricValues[newiid].mask):
            self.assertEqual(m, n)
        # Test with new sliceMetric (with no slicer previously set up).
        testbbm2 = sliceMetrics.RunSliceMetric(outDir='.')
        testbbm2.readMetricData(filename)
        for m, n in zip(self.testbbm.metricValues[oldiid].data,
                        testbbm2.metricValues[0].data):
            np.testing.assert_equal(m, n)
        for m, n in zip(self.testbbm.metricValues[oldiid].mask,
                        testbbm2.metricValues[0].mask):
            self.assertEqual(m, n)
        # Check if simDataName and sqlconstraint were restored as expected.
        self.assertEqual(testbbm2.simDataNames[0], self.testbbm.simDataNames[oldiid])
        self.assertEqual(testbbm2.sqlconstraints[0], self.testbbm.sqlconstraints[oldiid])
        # plot parameters not currently being written to disk
        #self.assertEqual(testbbm2.plotDict['Completeness']['xlabel'], 'Completeness')

class TestSummaryStatisticRunSliceMetric(unittest.TestCase):
    def setUp(self):
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotDict={'units':'meanunits'})
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.testbbm._setMetrics([self.m1,])
        self.iid = 0
        self.summaryStat = metrics.MeanMetric('metricdata')
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
        os.remove('resultsDb_sqlite.db')

    def testSummaryStatistic(self):
        """Test summary statistic calculation."""
        # Try unislicer first: expect that summary statistic return will be simply the unislicer value.
        self.slicer = slicers.UniSlicer()
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint,
                               metadata=self.metadata)
        summary = self.testbbm.computeSummaryStatistics(self.iid, self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues[self.iid][0])
        summary = self.testbbm.computeSummaryStatistics(self.iid, metrics.IdentityMetric('metricdata'))
        self.assertEqual(summary, self.testbbm.metricValues[self.iid][0])
        # Try oneD slicer: other slicers should behave similarly.
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.testbbm._setMetrics([self.m1,])
        self.slicer = slicers.OneDSlicer('testdata', bins=100)
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname, sqlconstraint=self.sqlconstraint,
                               metadata=self.metadata)
        summary = self.testbbm.computeSummaryStatistics(self.iid, self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues[self.iid].mean())

class TestPlottingRunSliceMetric(unittest.TestCase):
    def setUp(self):
        # Set up dictionary of all plotting parameters to test.
        self.plotDict = {'units': 'testunits',
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
        self.m1 = metrics.MeanMetric('testdata', metricName='Test labels', plotDict = self.plotDict)
        self.m1iid = 0
        self.m2 = metrics.MeanMetric('testdata', metricName='Test defaults')
        self.m2iid = 1
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
        os.remove('resultsDb_sqlite.db')

    def testPlotting(self):
        """Test plotting."""
        import matplotlib.pyplot as plt
        # Test OneDSlicer.
        bins = np.arange(0, 1.25, .1)
        self.slicer = slicers.OneDSlicer('testdata', bins=bins)
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.testbbm._setMetrics([self.m1, self.m2])
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname,
                               sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        # Test plotting oneDslicer, where we've set the plot parameters.
        fignums = self.testbbm.plotMetric(self.m1iid, savefig=False)
        fig = plt.figure(fignums['BinnedData'])
        ax = plt.gca()
        # Check x and y limits set from plot args.
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (self.plotDict['xMin'], self.plotDict['xMax']))
        ylims = plt.ylim()
        np.testing.assert_almost_equal(ylims, (self.plotDict['yMin'], self.plotDict['yMax']))
        # Check x and y labels
        self.assertEqual(ax.get_xlabel(), self.plotDict['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotDict['ylabel'])
        # Check title
        self.assertEqual(ax.get_title(), self.plotDict['title'])
        # Test a spatial slicer.
        self.testbbm = sliceMetrics.RunSliceMetric(outDir='.')
        self.testbbm._setMetrics([self.m1, ])
        self.slicer = slicers.HealpixSlicer(nside=4, lonCol='ra', latCol='dec', verbose=False)
        self.slicer.setupSlicer(self.dv)
        self.testbbm._setSlicer(self.slicer)
        self.testbbm.runSlices(self.dv, simDataName=self.opsimname,
                               sqlconstraint=self.sqlconstraint, metadata=self.metadata)
        fignums = self.testbbm.plotMetric(self.m1iid, savefig=False)
        # Test histogram.
        fig = plt.figure(fignums['Histogram'])
        ax = plt.gca()
        # Check x limits.
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (self.plotDict['xMin'], self.plotDict['xMax']))
        # Check x and y labels.
        self.assertEqual(ax.get_xlabel(), self.plotDict['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotDict['ylabel'])
        # Check title.
        self.assertEqual(ax.get_title(), self.plotDict['title'])
        # Test sky map.
        fig = plt.figure(fignums['SkyMap'])
        ax = plt.gca()
        # Not sure how to check clims of color bar.
        # Check title.
        self.assertEqual(ax.get_title(), self.plotDict['title'])

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TestSetupRunSliceMetric)
    suites += unittest.makeSuite(TestRunRunSliceMetric)
    suites += unittest.makeSuite(TestReadWriteRunSliceMetric)
    suites += unittest.makeSuite(TestSummaryStatisticRunSliceMetric)
    suites += unittest.makeSuite(TestPlottingRunSliceMetric)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
