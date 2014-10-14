import matplotlib
matplotlib.use("Agg")
import os
import unittest
import numpy as np
import lsst.sims.maf.db as db
import lsst.sims.maf.utils.outputUtils as out

class TestOpsimDb(unittest.TestCase):
    """Test opsim specific database class."""
    def setUp(self):
        filepath = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/')
        self.dbAddress = 'sqlite:///' + filepath + 'opsimblitz1_1133_sqlite.db'
        self.oo = db.OpsimDatabase(self.dbAddress)

    def tearDown(self):
        del self.oo
        self.dbAddress = None
        self.oo = None
        
    def testOpsimDbSetup(self):
        """Test opsim specific database class setup/instantiation."""
        # Test tables were connected to.
        self.assertTrue(isinstance(self.oo.tables, dict))
        self.assertEqual(self.oo.dbTables['summaryTable'][0], 'Summary')
        # Test can override default table name/id keys if needed.
        oo = db.OpsimDatabase(self.dbAddress, dbTables={'summaryTable':['ObsHistory', 'obsHistID']})
        self.assertEqual(oo.dbTables['summaryTable'][0], 'ObsHistory')

    def testOpsimDbMetricData(self):
        """Test queries for sim data. """
        data = self.oo.fetchMetricData(['finSeeing',], 'filter="r" and finSeeing<1.0')
        self.assertEqual(data.dtype.names, ('obsHistID', 'finSeeing'))
        self.assertTrue(data['finSeeing'].max() <= 1.0)

    def testOpsimDbPropID(self):
        """Test queries for prop ID"""
        propids, wfd, dd, propID2Name = self.oo.fetchPropIDs()
        self.assertTrue(len(propids) > 0)
        self.assertTrue(len(wfd) > 0)
        self.assertTrue(len(dd) > 0)
        for w in wfd:
            self.assertTrue(w in propids)
        for d in dd:
            self.assertTrue(d in propids)
        
    def testOpsimDbFields(self):
        """Test queries for field data."""
        # Fetch field data for all fields.
        dataAll = self.oo.fetchFieldsFromFieldTable()
        self.assertEqual(dataAll.dtype.names, ('fieldID', 'fieldRA', 'fieldDec'))
        # Fetch field data for all fields requested by a particular propid.
        propids, wfd, dd, propID2Name = self.oo.fetchPropIDs()
        propid = propids[0]
        dataProp1 = self.oo.fetchFieldsFromFieldTable(propID=propid)
        # Fetch field data for all fields requested by all proposals.
        dataPropAll = self.oo.fetchFieldsFromFieldTable(propID=propids)
        self.assertTrue(dataProp1.size < dataPropAll.size)
        # And check that did not return multiple copies of the same field.
        self.assertEqual(len(dataPropAll['fieldID']), len(np.unique(dataPropAll['fieldID'])))
        
    def testOpsimDbRunLength(self):
        """Test query for length of opsim run."""
        nrun = self.oo.fetchRunLength()
        self.assertEqual(nrun, 0.0794)    

    def testOpsimDbSimName(self):
        """Test query for opsim name."""
        simname = self.oo.fetchOpsimRunName()
        self.assertTrue(isinstance(simname, str))
        self.assertEqual(simname, 'opsimblitz1_1133')

    def testOpsimDbSeeingColName(self):
        """Test query to pull out column name for seeing (seeing or finSeeing)."""
        seeingcol = self.oo.fetchSeeingColName()
        self.assertTrue(seeingcol, 'finSeeing')

    '''
    def testOpsimDbConfig(self):
        """Test generation of config data. """
        configsummary, configdetails = self.oo.fetchConfig()
        self.assertTrue(isinstance(configsummary, dict))
        self.assertTrue(isinstance(configdetails, dict))
        self.assertEqual(set(configsummary.keys()), set(['Version', 'RunInfo', 'Proposals', 'keyorder']))
        propids, wfd, dd = self.oo.fetchPropIDs()
        propidsSummary = []
        for propname in configsummary['Proposals']:
            if propname != 'keyorder':            
                propidsSummary.append(configsummary['Proposals'][propname]['PropID'])
        self.assertEqual(set(propidsSummary), set(propids))
        out.printDict(configsummary, 'Summary')
        out.printDict(configdetails, 'Details')
    '''
       
if __name__ == "__main__":
    unittest.main()
