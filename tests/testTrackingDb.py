import matplotlib
matplotlib.use("Agg")
import os
import warnings
import unittest
import lsst.sims.maf.db as db
import shutil
import lsst.utils.tests


class TestTrackingDb(unittest.TestCase):

    def setUp(self):
        self.opsimRun = 'testopsim'
        self.opsimComment = 'opsimcomment'
        self.mafComment = 'mafcomment'
        self.mafDir = 'mafdir'
        self.trackingDb = 'trackingDb_sqlite.db'
        self.mafDate = '1/1/11'
        self.opsimDate = '1/1/11'
        self.dbFile = None

    def tearDown(self):
        if os.path.isfile(self.trackingDb):
            os.remove(self.trackingDb)

    def testTrackingDbCreation(self):
        """Test tracking database creation."""
        trackingdb = db.TrackingDb(database=self.trackingDb)
        self.assertTrue(os.path.isfile(self.trackingDb))
        trackingdb.close()

    def testAddRun(self):
        """Test adding a run to the tracking database."""
        trackingdb = db.TrackingDb(database=self.trackingDb)
        trackId = trackingdb.addRun(opsimRun=self.opsimRun, opsimComment=self.opsimComment,
                                    mafComment=self.mafComment, mafDir=self.mafDir,
                                    mafDate=self.mafDate, opsimDate=self.opsimDate,
                                    dbFile=self.dbFile)
        tdb = db.Database(database=self.trackingDb,
                          dbTables={'runs': ['runs', 'mafRunId']})
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Try adding this run again. Should just return previous trackId without adding.
        trackId2 = trackingdb.addRun(opsimRun=self.opsimRun, opsimComment=self.opsimComment,
                                     mafComment=self.mafComment, mafDir=self.mafDir,
                                     mafDate=self.mafDate, opsimDate=self.opsimDate,
                                     dbFile=self.dbFile)
        self.assertEqual(trackId, trackId2)
        # Test will add run, if we use 'override=True'. Also works to use None's.
        trackId3 = trackingdb.addRun(opsimRun=None, opsimComment=None, mafComment=None,
                                     mafDir=self.mafDir, override=True,
                                     mafDate=self.mafDate, opsimDate=self.opsimDate,
                                     dbFile=self.dbFile)
        self.assertNotEqual(trackId, trackId3)
        trackingdb.close()

    def testDelRun(self):
        """Test removing a run from the tracking database."""
        trackingdb = db.TrackingDb(database=self.trackingDb)
        tdb = db.Database(database=self.trackingDb,
                          dbTables={'runs': ['runs', 'mafRunId']})
        # Add a run.
        trackId = trackingdb.addRun(opsimRun=self.opsimRun, opsimComment=self.opsimComment,
                                    mafComment=self.mafComment, mafDir=self.mafDir,
                                    mafDate=self.mafDate, opsimDate=self.opsimDate,
                                    dbFile=self.dbFile)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Test removal works.
        trackingdb.delRun(trackId)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertTrue(len(res) == 0)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, trackId)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
