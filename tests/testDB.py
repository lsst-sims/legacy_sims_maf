import matplotlib
matplotlib.use("Agg")
import os
import unittest
import numpy as np
import lsst.sims.maf.db as db

class TestDb(unittest.TestCase):
    def setUp(self):
        self.database = os.path.join(os.getenv('SIMS_MAF_DIR'),
                                    'tests', 'opsimblitz1_1133_sqlite.db')
        self.driver = 'sqlite'

    def tearDown(self):
        del self.driver
        del self.database

    def testTable(self):
        """Test that we can connect to a DB table and pull data."""
        # Make a connection
        table = db.Table('Summary', 'obsHistID', database=self.database, driver=self.driver)
        # Query a particular column.
        data = table.query_columns_Array(colnames=['finSeeing'])
        self.assertTrue(isinstance(data, np.ndarray))
        self.assertTrue('finSeeing' in data.dtype.names)
        # Check error is raised if grabbing a non-existent column
        self.assertRaises(ValueError, table.query_columns_Array, colnames=['notRealName'])
        # Check that can apply a sql constraint.
        constraint = 'filter = "r" and finSeeing < 1.0'
        data = table.query_columns_Array(colnames=['finSeeing', 'filter'], constraint=constraint)
        maxseeing = data['finSeeing'].max()
        self.assertTrue(maxseeing <= 1.0)
        filter = np.unique(data['filter'])
        self.assertEqual(filter, 'r')

    def testBaseDatabase(self):
        """Test base database class."""
        # Test instantation with no dbTables info (and no defaults).
        basedb = db.Database(database=self.database, driver=self.driver)
        self.assertEqual(basedb.tables, None)
        # Test instantiation with some tables.
        basedb = db.Database(database=self.database, driver=self.driver,
                             dbTables={'obsHistTable':['ObsHistory', 'obsHistID'],
                                       'fieldTable':['Field', 'fieldID'],
                                       'obsHistoryProposalTable':['Obshistory_Proposal',
                                       'obsHistory_propID']})
        self.assertEqual(set(basedb.tables.keys()),
                         set(['obsHistTable',
                              'obsHistoryProposalTable', 'fieldTable']))
        # Test general query with a simple query.
        query = 'select fieldID, fieldRA, fieldDec from Field where fieldDec>0'
        data = basedb.queryDatabase('fieldTable', query)
        self.assertEqual(data.dtype.names, ('fieldID', 'fieldRA', 'fieldDec'))

    def testSqliteFileNotExists(self):
        """Test that db gives useful error message if db file doesn't exist."""
        self.assertRaises(IOError, db.Database, 'thisdatabasedoesntexist_sqlite.db')

    def testArbitraryQuery(self):
        """
        Test that an arbitrary query can be executed.
        No attempt is made to validat the results.
        """
        table = db.Table('Summary', 'obsHistID', database=self.database, driver=self.driver)
        query = 'select count(expMJD), filter from ObsHistory, ObsHistory_Proposal'
        query += ' where obsHistID = ObsHistory_obsHistID group by Proposal_propID, filter'
        results = table.execute_arbitrary(query)

        #This is a specific case that gave me trouble when refactoring DBObject
        #Something about the fact that the database was stored in unicode
        #tripped up numpy.rec.fromrecords().  This test will verify that the
        #problem has not recurred
        query = 'select sessionID, version, sessionDate, runComment from Session'
        dtype=np.dtype([('id',int),('version',str,256),('date',str,256),('comment',str,256)])
        results = table.execute_arbitrary(query,dtype=dtype)
        self.assertTrue(isinstance(results[0][0],int))
        self.assertTrue(isinstance(results[0][1],str))
        self.assertTrue(isinstance(results[0][2],str))
        self.assertTrue(isinstance(results[0][3],str))
        self.assertEqual(results[0][0],1133)
        self.assertEqual(results[0][1],'3.1')
        self.assertEqual(results[0][2],'2014-07-11 17:02:08')
        self.assertEqual(results[0][3],'all DD + regular survey for 1 lunation')

if __name__ == "__main__":
    unittest.main()
