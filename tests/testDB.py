import numpy as np
import unittest
import os
import lsst.sims.operations.maf.db as db

class TestDB(unittest.TestCase):
    def setUp(self):
        filepath = os.environ['SIMS_OPERATIONS_MAF_DIR']+'/examples/'
        self.dbAddress = 'sqlite:///'+filepath+'opsim_small.sqlite'
        self.tableName = 'opsim_small'


    def testTable(self):
        """Test that we can connect to a DB and pull data """
        # Make a connection
        table = db.Table(self.tableName, 'obsHistID', self.dbAddress )
        # Pull the data
        data1 = table.query_columns_RecArray()
        # Make sure it is a numpy array
        assert(type(data1).__name__ == 'ndarray' )
        data1 = table.query_columns_RecArray(colnames=['finSeeing'])
        # Check error is raised if grabbing a non-existent column
        self.assertRaises(ValueError,table.query_columns_RecArray,colnames=['notRealName'] )
        
        
    def testDatabase(self):
        """Need an example full-database with all tables to test."""
        pass

if __name__ == "__main__":
    unittest.main()
