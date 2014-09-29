import os
import unittest
import numpy as np
import lsst.sims.maf.db as db

class TestDb(unittest.TestCase):
    def setUp(self):
        filepath = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/')
        self.dbAddress = 'sqlite:///' + filepath + 'opsimblitz1_1133_sqlite.db'

    def tearDown(self):
        self.dbAddress = None
        
    def testTable(self):
        """Test that we can connect to a DB table and pull data."""
        # Make a connection
        table = db.Table('Summary', 'obsHistID', self.dbAddress)
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
        basedb = db.Database(self.dbAddress)
        self.assertEqual(basedb.tables, None)
        # Test instantiation with some tables.
        basedb = db.Database(self.dbAddress, dbTables={'obsHistTable':['ObsHistory', 'obsHistID'],
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
        self.assertRaises(IOError, db.Database, 'sqlite:///thisdatabasedoesntexist_sqlite.db')

    def testArbitraryQuery(self):
        """
        Test that an arbitrary query can be executed.
        No attempt is made to validat the results.
        """
        table = db.Table('Summary', 'obsHistID', self.dbAddress)
        query = 'select count(expMJD), filter from ObsHistory, ObsHistory_Proposal'
        query += ' where obsHistID = ObsHistory_obsHistID group by Proposal_propID, filter'
        results = table.execute_arbitrary(query)

if __name__ == "__main__":
    unittest.main()
