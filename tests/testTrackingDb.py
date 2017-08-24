import os
import unittest
import tempfile
import shutil
import lsst.sims.maf.db as db
import lsst.utils.tests


ROOT = os.path.abspath(os.path.dirname(__file__))


class TestTrackingDb(unittest.TestCase):

    def setUp(self):
        self.opsimRun = 'testopsim'
        self.opsimGroup = 'test'
        self.opsimComment = 'opsimcomment'
        self.mafComment = 'mafcomment'
        self.mafDir = 'mafdir'
        self.trackingDb = tempfile.mktemp(dir=ROOT,
                                          prefix='trackingDb_sqlite',
                                          suffix='.db')
        self.mafVersion = '1.0'
        self.mafDate = '2017-01-01'
        self.opsimVersion = '4.0'
        self.opsimDate = '2017-02-01'
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
        trackId = trackingdb.addRun(opsimGroup=self.opsimGroup, opsimRun=self.opsimRun,
                                    opsimComment=self.opsimComment,
                                    opsimVersion=self.opsimVersion, opsimDate=self.opsimDate,
                                    mafComment=self.mafComment, mafDir=self.mafDir,
                                    mafVersion=self.mafVersion, mafDate=self.mafDate,
                                    dbFile=self.dbFile)
        tdb = db.Database(database=self.trackingDb,
                          dbTables={'runs': ['runs', 'mafRunId']})
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Try adding this run again. Should return previous trackId.
        trackId2 = trackingdb.addRun(mafDir=self.mafDir)
        self.assertEqual(trackId, trackId2)
        # Test will add additional run, with new trackId.
        trackId3 = trackingdb.addRun(mafDir='test2')
        self.assertNotEqual(trackId, trackId3)
        trackingdb.close()

    def testDelRun(self):
        """Test removing a run from the tracking database."""
        trackingdb = db.TrackingDb(database=self.trackingDb)
        tdb = db.Database(database=self.trackingDb,
                          dbTables={'runs': ['runs', 'mafRunId']})
        # Add a run.
        trackId = trackingdb.addRun(mafDir=self.mafDir)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertEqual(res['mafRunId'][0], trackId)
        # Test removal works.
        trackingdb.delRun(trackId)
        res = tdb.queryDatabase('runs', 'select * from runs')
        self.assertTrue(len(res) == 0)
        # Test cannot remove run which does not exist.
        self.assertRaises(Exception, trackingdb.delRun, trackId)
        trackingdb.close()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
