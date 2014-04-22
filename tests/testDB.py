import numpy as np
import unittest
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.maf.db as db
    from lsst.sims.maf.utils.getData import getDbAddress, fetchSimData, fetchFieldsFromOutputTable, fetchFieldsFromFieldTable

class TestDB(unittest.TestCase):
    def setUp(self):
        filepath = os.environ['SIMS_MAF_DIR']+'/examples/'
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

    def testGetData(self):
        """Test the GetData utils"""
        dbA = getDbAddress(dbLoginFile = os.environ['SIMS_MAF_DIR']+'/tests/dbLogin')
        assert(dbA == 'sqlite:///opsim.sqlite' )

        simdata = fetchSimData(self.tableName, self.dbAddress, "filter = 'r'", ['expMJD'])
        simdata_nod =  fetchSimData(self.tableName, self.dbAddress, "filter = 'r'", ['expMJD'], distinctExpMJD=False)
        assert(simdata.size <= simdata_nod.size) # The test DB is actually already distinct on ExpMJD...
        
        fields1 = fetchFieldsFromOutputTable(self.tableName, self.dbAddress, "filter = 'r'" )
        cols = ['fieldID', 'fieldRA',  'fieldDec']
        for col in cols:
            assert(col in fields1.dtype.names)

        # Need full DB example to test fetchFieldsFromFieldTable
        
    def testDatabase(self):
        """Need an example full-database with all tables to test."""
        pass

if __name__ == "__main__":
    unittest.main()
