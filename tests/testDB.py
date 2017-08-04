import matplotlib
matplotlib.use("Agg")
import os
import unittest
import numpy as np
import lsst.sims.maf.db as db
import lsst.utils.tests
from lsst.sims.utils.CodeUtilities import sims_clean_up


class TestDb(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.database = os.path.join(os.getenv('SIMS_MAF_DIR'),
                                     'tests', 'pontus_1150.db')
        self.driver = 'sqlite'

    def tearDown(self):
        del self.driver
        del self.database

    def testBaseDatabase(self):
        """Test base database class."""
        # Test instantiation connects to expected tables.
        basedb = db.Database(database=self.database, driver=self.driver)
        expectedTables = ['Config', 'ScheduledDowntime', 'SlewMaxSpeeds',
                          'Field', 'Session', 'SummaryAllProps', 'ObsExposures',
                          'SlewActivities', 'TargetExposures', 'ObsHistory',
                          'SlewFinalState', 'TargetHistory', 'ObsProposalHistory',
                          'SlewHistory', 'TargetProposalHistory', 'Proposal',
                          'SlewInitialState', 'UnscheduledDowntime']
        self.assertEqual(set(basedb.tableNames),
                         set(expectedTables))
        # Test general query with a simple query.
        query = 'select fieldId, ra, dec from Field where dec>0 limit 3'
        data = basedb.query_arbitrary(query)
        self.assertEqual(len(data), 3)
        # Test query columns with a simple query.
        data = basedb.query_columns('Field', colnames=['fieldId', 'ra', 'dec'], numLimit=3)
        self.assertEqual(data.dtype.names, ('fieldId', 'ra', 'dec'))
        self.assertEqual(len(data), 3)

    def testSqliteFileNotExists(self):
        """Test that db gives useful error message if db file doesn't exist."""
        self.assertRaises(IOError, db.Database, 'thisdatabasedoesntexist_sqlite.db')

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
