import matplotlib
matplotlib.use("Agg")
import os, warnings
import unittest
import lsst.utils.tests as utilsTests
import lsst.sims.maf.db as db
import shutil

class TestTrackingDb(unittest.TestCase):
    def setUp(self):
        self.opsimRun = 'testopsim'
        self.opsimComment = 'opsimcomment'
        self.mafComment = 'mafcomment'
        self.mafDir = 'mafdir'
        self.trackingDb = 'trackingDb_sqlite.db'
        self.mafDate = '1/1/11'
        self.opsimDate = '1/1/11'

    def tearDown(self):
        if os.path.isfile(self.trackingDb):
            os.remove(self.trackingDb)

    def testTrackingDbCreation(self):
        """Test tracking database creation."""
        trackingdb = db.TrackingDb(database = self.trackingDb)
        self.assertTrue(os.path.isfile(self.trackingDb))
        trackingdb.close()

    def testAddRun(self):
        """Test adding a run to the tracking database."""
        trackingdb = db.TrackingDb(database = self.trackingDb)
        trackId = trackingdb.addRun(opsimRun = self.opsimRun, opsimComment = self.opsimComment,
                                    mafComment = self.mafComment, mafDir = self.mafDir,
                                    mafDate = self.mafDate, opsimDate = self.opsimDate)
        tdb = db.Database(self.trackingDb,
                            dbTables={'runs':['runs', 'mafRunId']})
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Try adding this run again. Should just return previous trackId without adding.
        trackId2 = trackingdb.addRun(opsimRun = self.opsimRun, opsimComment = self.opsimComment,
                                     mafComment = self.mafComment, mafDir = self.mafDir,
                                     mafDate = self.mafDate, opsimDate = self.opsimDate)
        self.assertEqual(trackId, trackId2)
        # Test will add run, if we use 'override=True'. Also works to use None's.
        trackId3 = trackingdb.addRun(opsimRun = None, opsimComment=None, mafComment=None,
                                     mafDir = self.mafDir, override=True,
                                     mafDate = self.mafDate, opsimDate = self.opsimDate)
        self.assertNotEqual(trackId, trackId3)
        trackingdb.close()

    def testDelRun(self):
        """Test removing a run from the tracking database."""
        trackingdb = db.TrackingDb(database = self.trackingDb)
        tdb = db.Database(self.trackingDb,
                          dbTables={'runs':['runs', 'mafRunId']})
        # Add a run.
        trackId = trackingdb.addRun(opsimRun = self.opsimRun, opsimComment = self.opsimComment,
                                    mafComment = self.mafComment, mafDir = self.mafDir,
                                    mafDate = self.mafDate, opsimDate = self.opsimDate)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Test removal works.
        trackingdb.delRun(trackId)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertTrue(len(res) == 0)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, trackId)

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(TestTrackingDb)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
