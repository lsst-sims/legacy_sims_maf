import os, warnings
import unittest
import lsst.utils.tests as utilsTests
import numpy as np
import lsst.sims.maf.db as db
import shutil

class TestResultsDb(unittest.TestCase):
    def setUp(self):
        self.outDir = 'Out'
        self.metricName = 'Count ExpMJD'
        self.slicerName = 'OneDSlicer'
        self.runName = 'fakeopsim'
        self.sqlconstraint = ''
        self.metadata = 'Dithered'
        self.metricDataFile = 'testmetricdatafile.npz'
        self.plotType = 'BinnedData'
        self.plotName = 'testmetricplot_BinnedData.png'
        self.summaryStatName1 = 'Mean'
        self.summaryStatValue1 = 20
        self.summaryStatName2 = 'Median'
        self.summaryStatValue2 = 18
        self.summaryStatName3 = 'TableFrac'
        self.summaryStatValue3 = np.empty(10, dtype=[('name', '|S12'), ('value', float)])
        for i in range(10):
            self.summaryStatValue3['name'] = 'test%d' %(i)
            self.summaryStatValue3['value'] = i
        self.displayDict = {'group':'seeing', 'subgroup':'all', 'order':1, 'caption':'lalalalal'}
        
    def testDbCreation(self):
        # Test default sqlite file created (even if outDir doesn't exist)
        resultsdb = db.ResultsDb(outDir=self.outDir)
        self.assertTrue(os.path.isfile(os.path.join(self.outDir, 'resultsDb_sqlite.db')))
        resultsdb.close()
        # Test can pick custom name in directory that exists.
        sqlitefilename = os.path.join(self.outDir, 'testDb_sqlite.db')
        resultsdb = db.ResultsDb(resultsDbAddress = 'sqlite:///' + sqlitefilename)
        self.assertTrue(os.path.isfile(sqlitefilename))
        resultsdb.close()
        # Test that get appropriate exception if directory doesn't exist.
        sqlitefilename = os.path.join(self.outDir + 'test', 'testDb_sqlite.db')
        self.assertRaises(ValueError, db.ResultsDb, resultsDbAddress='sqlite:///' + sqlitefilename)            
        
    def testAddData(self):
        resultsDb = db.ResultsDb(outDir=self.outDir)
        # Add metric. 
        metricId = resultsDb.addMetric(self.metricName, self.slicerName, self.runName, self.sqlconstraint,
                                        self.metadata, self.metricDataFile)
        # Add plot.
        resultsDb.addPlot(metricId, self.plotType, self.plotName)
        # Add normal summary statistics. 
        resultsDb.addSummaryStat(metricId, self.summaryStatName1, self.summaryStatValue1)
        resultsDb.addSummaryStat(metricId, self.summaryStatName2, self.summaryStatValue2)
        # Add something like tableFrac summary statistic.
        resultsDb.addSummaryStat(metricId, self.summaryStatName3, self.summaryStatValue3)
        # Test get warning when try to add a non-conforming summary stat.
        teststat = np.empty(10, dtype=[('col', '|S12'), ('value', float)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.addSummaryStat(metricId, 'testfail', teststat)
            self.assertTrue("not save" in str(w[-1].message))
        # Test get warning when try to add a string (non-conforming) summary stat.
        teststat = 'teststring'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resultsDb.addSummaryStat(metricId, 'testfail', teststat)
            self.assertTrue("not save" in str(w[-1].message))
        

    def tearDown(self):
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)


class TestUseResultsDb(unittest.TestCase):
    def setUp(self):
        self.outDir = 'Out'
        self.metricName = 'Count ExpMJD'
        self.slicerName = 'OneDSlicer'
        self.runName = 'fakeopsim'
        self.sqlconstraint = ''
        self.metadata = 'Dithered'
        self.metricDataFile = 'testmetricdatafile.npz'
        self.plotType = 'BinnedData'
        self.plotName = 'testmetricplot_BinnedData.png'
        self.summaryStatName1 = 'Mean'
        self.summaryStatValue1 = 20
        self.summaryStatName2 = 'Median'
        self.summaryStatValue2 = 18
        self.resultsDb = db.ResultsDb(outDir=self.outDir)
        self.metricId = self.resultsDb.addMetric(self.metricName, self.slicerName, self.runName, self.sqlconstraint,
                                            self.metadata, self.metricDataFile)
        self.resultsDb.addPlot(self.metricId, self.plotType, self.plotName)
        self.resultsDb.addSummaryStat(self.metricId, self.summaryStatName1, self.summaryStatValue1)
        self.resultsDb.addSummaryStat(self.metricId, self.summaryStatName2, self.summaryStatValue2)

    def getIds(self):
        mids = self.resultsDb.getMetricIds()
        self.assertEqual(mids[0], self.metricId)
        
    def showSummary(self):
        self.resultsDb.getSummaryStats()
        
    def tearDown(self):
        self.resultsDb.close()
        if os.path.isdir(self.outDir):
            shutil.rmtree(self.outDir)

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TestResultsDb)
    suites += unittest.makeSuite(TestUseResultsDb)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
