import numpy as np
import unittest
import lsst.sims.maf.binMetrics as binMetrics
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binners as binners

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


class TestSetupBaseBinMetric(unittest.TestCase):
    """Unit tests relating to setting up the baseBinMetric"""
    def setUp(self):
        self.testbbm = binMetrics.BaseBinMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.m2 = metrics.CountMetric('testdata', metricName='Count testdata',
                                      plotParams={'units':'countunits', 'title':'count_title'})
        self.m3 = metrics.CompletenessMetric('filter', metricName='Completeness', g=50, r=50)
        self.binner = binners.UniBinner()

    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
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
        self.assertTrue(self.testbbm.setBinner(binner2, override=False))
        # Test can not set/override binner (when != previous binner)
        binner2 = binners.HealpixBinner(nside=16, verbose=False)
        self.assertFalse(self.testbbm.setBinner(binner2, override=False))
        # Unless you really wanted to..
        self.assertTrue(self.testbbm.setBinner(binner2, override=True))

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
        self.assertEqual(self.testbbm.plotParams['Mean testdata'].keys(), ['units', '_units'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].keys(), ['units', '_units', 'title'])
        self.assertEqual(self.testbbm.plotParams['Count testdata'].values(),
                         ['countunits', 'Count testdata', 'count_title'])
        # Test that can set metrics using a single metric (not a list)
        testbbm2 = binMetrics.BaseBinMetric()
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

class TestRunBaseBinMetric(unittest.TestCase):        
    def setUp(self):
        self.testbbm = binMetrics.BaseBinMetric()
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
        self.binner = binners.OneDBinner('testdata')
        self.binner.setupBinner(self.dv, bins=np.arange(0, 1.25, .1))
        self.testbbm.setBinner(self.binner)
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])

    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.binner
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.binner = None

    def testRunBins(self):
        """Test creating metric data values."""
        opsimname = 'opsim1000'
        metadata = 'created fake testdata'
        comment = 'testing fake data run'
        self.testbbm.runBins(self.dv, simDataName=opsimname, metadata=metadata, comment=comment)
        # Test that copied opsim name and metadata and comment correctly for each metric name.
        for mname in self.metricNames:
            self.assertEqual(self.testbbm.simDataName[mname], opsimname)
            self.assertEqual(self.testbbm.metadata[mname], metadata)
            self.assertEqual(self.testbbm.comment[mname], comment)
        # Test that created metric data with expected number of data points.
        for mname in self.metricNames:
            self.assertEqual(len(self.testbbm.metricValues[mname]), len(self.binner))
        # Test that metric data was masked where expected (last bin) due to no data in bin.
        lastbin = len(self.binner) - 1
        for mname in self.metricNames:
            self.assertEqual(self.testbbm.metricValues[mname].mask[lastbin], True)

    def testReduce(self):
        """Test running reduce methods."""
        # Completeness metric has reduce methods, so check on those.
        opsimname = 'opsim1000'
        metadata = 'created fake testdata'
        comment = 'testing fake data run'
        self.testbbm.runBins(self.dv, simDataName=opsimname, metadata=metadata, comment=comment)
        self.testbbm.reduceAll()
        # Check that all metric data values expected exist.
        for m in self.metricNames:
            self.assertTrue(m in self.testbbm.metricValues.keys())
        for m in self.reduceNames:
            self.assertTrue(m in self.testbbm.metricValues.keys())
        # Check that simdata, metadata and comments were copied properly.
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.simDataName[m], opsimname)
            self.assertEqual(self.testbbm.metadata[m], metadata)
            self.assertEqual(self.testbbm.comment[m], comment)
        # Check that plot parameters were copied properly.
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.plotParams[m]['xlabel'], 'Completeness')
        # Check that mask carried through properly.
        lastbin = len(self.binner) - 1
        for m in self.reduceNames:
            self.assertEqual(self.testbbm.metricValues[m].mask[lastbin], True)
                

class TestReadWriteBaseBinMetric(unittest.TestCase):        
    def setUp(self):
        self.testbbm = binMetrics.BaseBinMetric()
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
        self.binner = binners.OneDBinner('testdata')
        self.binner.setupBinner(self.dv, bins=np.arange(0, 1.25, .1))
        self.testbbm.setBinner(self.binner)
        self.testbbm.setMetrics([self.m1, self.m2, self.m3])
        self.opsimname = 'opsim1000'
        self.metadata = 'created fake testdata'
        self.comment = 'testing fake data run'
        self.testbbm.runBins(self.dv, simDataName=self.opsimname, metadata=self.metadata, comment=self.comment)
        self.testbbm.reduceAll()
        self.outroot = 'testBaseBinMetric'
        self.testbbm.writeAll(outDir='.', outfileRoot=self.outroot)
        self.expectedfiles = []
        for m in self.metricNames:
            filename = (self.outroot + ' ' + m + ' ' + self.comment + ' ' +
                        self.binner.binnerName[:4].upper() + '.npz')
            filename = filename.replace(' ', '_')
            self.expectedfiles.append(filename)
        for m in self.reduceNames:
            filename = (self.outroot + ' ' + m + ' ' + self.comment + ' ' +
                        self.binner.binnerName[:4].upper() + '.npz')
            filename = filename.replace(' ', '_')
            self.expectedfiles.append(filename)
                        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.m2
        del self.m3
        del self.binner
        self.testbbm = None
        self.m1 = None
        self.m2 = None
        self.binner = None
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
        # Test with binner already set up in binMetric.
        filename = (self.outroot + '_' + 'Completeness' + '_' + self.comment + '_' +
                    self.binner.binnerName[:4].upper() + '.npz')
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
        # Test with new binMetric (with no binner previously set up).
        testbbm2 = binMetrics.BaseBinMetric()
        testbbm2.readMetricValues(filename)
        self.assertTrue('Completeness' in testbbm2.metricValues)
        for m, n in zip(self.testbbm.metricValues['Completeness'].data,
                        testbbm2.metricValues['Completeness'].data):
            np.testing.assert_equal(m, n)
        for m, n in zip(self.testbbm.metricValues['Completeness'].mask,
                        testbbm2.metricValues['Completeness'].mask):
            self.assertEqual(m, n)
        # Check if simDataName and metadata were restored as expected.
        self.assertEqual(testbbm2.simDataName['Completeness'], self.testbbm.simDataName['Completeness'])
        self.assertEqual(testbbm2.metadata['Completeness'], self.testbbm.metadata['Completeness'])
        # plot parameters not currently being written to disk
        #self.assertEqual(testbbm2.plotParams['Completeness']['xlabel'], 'Completeness')

    def testOutputFileKey(self):
        """Test that the output file key is being generated as expected."""
        outkeys = self.testbbm.returnOutputFiles(verbose=False)
        # Check data in outkeys is valid
        for o in outkeys:
            self.assertEqual(o['comment'], self.comment)
            self.assertEqual(o['simDataName'], self.opsimname)
            self.assertEqual(o['metadata'], self.metadata)
            self.assertTrue(o['filename'].replace('./', '') in self.expectedfiles)
            self.assertEqual(o['binner'], self.binner.binnerName)
            self.assertTrue((o['metricName'] in self.metricNames) or (o['metricName'] in self.reduceNames)) 
        # Check data in outkeys is complete
        outkeysMetricNames = []
        for o in outkeys:
            outkeysMetricNames.append(o['metricName'])
        for m in self.metricNames:
            self.assertTrue(m in outkeysMetricNames)
        for m in self.reduceNames:
            self.assertTrue(m in outkeysMetricNames)

class TestSummaryStatisticBaseBinMetric(unittest.TestCase):
    def setUp(self):
        self.testbbm = binMetrics.BaseBinMetric()
        self.m1 = metrics.MeanMetric('testdata', metricName='Mean testdata',
                                     plotParams={'units':'meanunits'})
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.testbbm.setMetrics([self.m1,])
        self.summaryStat = metrics.MeanMetric('metricData')
        self.opsimname = 'opsim1000'
        self.metadata = 'created fake testdata'
        self.comment = 'testing fake data run'
        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.binner
        self.testbbm = None
        self.m1 = None
        self.binner = None

    def testSummaryStatistic(self):
        """Test summary statistic calculation."""
        # Try unibinner first: expect that summary statistic return will be simply the unibinner value.
        self.binner = binners.UniBinner()
        self.binner.setupBinner(self.dv)
        self.testbbm.setBinner(self.binner)
        self.testbbm.runBins(self.dv, simDataName=self.opsimname, metadata=self.metadata, comment=self.comment)
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'][0])
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', metrics.IdentityMetric('metricdata'))
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'][0])
        # Try oneD binner
        self.testbbm = binMetrics.BaseBinMetric()
        self.testbbm.setMetrics([self.m1,])
        self.binner = binners.OneDBinner('testdata')
        self.binner.setupBinner(self.dv, nbins=100)
        self.testbbm.setBinner(self.binner)
        self.testbbm.runBins(self.dv, simDataName=self.opsimname, metadata=self.metadata, comment=self.comment)
        summary = self.testbbm.computeSummaryStatistics('Mean testdata', self.summaryStat)
        self.assertEqual(summary, self.testbbm.metricValues['Mean testdata'].mean())
        # Other binners (spatial binner, etc) should be similar to oneD binner.

            
class TestPlottingBaseBinMetric(unittest.TestCase):
    def setUp(self):
        # Set up dictionary of all plotting parameters to test.
        self.plotParams = {'_units': 'testunits',
                        'title': 'my test title',  # plot titles
                        'xlabel': 'my xlabel',  # plot x labels
                        'ylabel': 'my ylabel',  # plot y labels
                        # For 1-d binner: set hist x min/max vals via bins when setting, then y vals via plotMin/Max
                        # For spatial binners: set hist x min/max vals via histMin/Max & number of bins via 'bins'
                        #   then for skymap, set colorbar min/max vals via plotMin/Max
                        'plotMin': -0.5,  # plot minimum values for skymap clims and y value for 1-d histograms
                        'plotMax': 1.5,   # plot maximum values for skymap clims and y value for 1-d histograms
                        'histMin': -0.5,  # histogram x minimum value for spatial binner
                        'histMax': 1.5,   # histogram x maximum value for spatial binner
                        # No way to set y value limits for spatial binner histogram?
                        'bins': 50       # parameter for number of bins for spatial binner histograms
                        }
        self.m1 = metrics.MeanMetric('testdata', metricName='Test labels', plotParams = self.plotParams)
        self.dv = makeDataValues(size=1000, min=0, max=1)
        self.opsimname = 'opsim1000'
        self.metadata = 'created fake testdata'
        self.comment = 'testing fake data run'
                        
    def tearDown(self):
        del self.testbbm
        del self.m1
        del self.binner
        self.testbbm = None
        self.m1 = None
        self.binner = None

    def testPlotting(self):        
        """Test plotting."""
        import matplotlib.pyplot as plt    
        # Test OneDBinner.
        self.binner = binners.OneDBinner('testdata')
        bins = np.arange(0, 1.25, .1)
        self.testbbm = binMetrics.BaseBinMetric()
        self.testbbm.setMetrics([self.m1,])
        self.binner.setupBinner(self.dv, bins=bins)
        self.testbbm.setBinner(self.binner)
        self.testbbm.runBins(self.dv, simDataName=self.opsimname, metadata=self.metadata, comment=self.comment)
        fignums = self.testbbm.plotMetric(self.m1.name, savefig=False)
        fig = plt.figure(fignums['hist'])
        ax = plt.gca()
        # Check x and y limits
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (bins.min(), bins.max()))
        ylims = plt.ylim()
        np.testing.assert_almost_equal(ylims, (self.plotParams['plotMin'], self.plotParams['plotMax']))
        # Check x and y labels
        self.assertEqual(ax.get_xlabel(), self.plotParams['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotParams['ylabel'])
        # Check title
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        # Test a spatial binner.
        self.testbbm = binMetrics.BaseBinMetric()
        self.testbbm.setMetrics([self.m1, ])
        self.binner = binners.HealpixBinner(nside=4, spatialkey1='ra', spatialkey2='dec', verbose=False)
        self.binner.setupBinner(self.dv)
        self.testbbm.setBinner(self.binner)
        self.testbbm.runBins(self.dv, simDataName=self.opsimname, metadata=self.metadata, comment=self.comment)
        fignums = self.testbbm.plotMetric(self.m1.name, savefig=False)
        # Test histogram.
        fig = plt.figure(fignums['hist'])
        ax = plt.gca()
        # Check x limits.
        xlims = plt.xlim()
        np.testing.assert_almost_equal(xlims, (self.plotParams['histMin'], self.plotParams['histMax']))
        # Check x and y labels.
        self.assertEqual(ax.get_xlabel(), self.plotParams['xlabel'])
        self.assertEqual(ax.get_ylabel(), self.plotParams['ylabel'])
        # Check title.
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        # Test sky map.
        fig = plt.figure(fignums['sky'])
        ax = plt.gca()
        # Not sure how to check clims of color bar.
        # Check title.
        self.assertEqual(ax.get_title(), self.plotParams['title'])
        
        
        
                                                                    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSetupBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRunBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReadWriteBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSummaryStatisticBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPlottingBaseBinMetric)
    unittest.TextTestRunner(verbosity=2).run(suite)        
