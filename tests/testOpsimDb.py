import matplotlib
matplotlib.use("Agg")
import os
import unittest
import numpy as np
import lsst.sims.maf.db as db
import lsst.sims.maf.utils.outputUtils as out
import lsst.utils.tests


class TestOpsimDb(unittest.TestCase):
    """Test opsim specific database class."""

    def setUp(self):
        self.database = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests',
                                     'pontus_1150.db')
        self.oo = db.OpsimDatabase(database=self.database)

    def tearDown(self):
        del self.oo
        del self.database
        self.oo = None

    def testOpsimDbSetup(self):
        """Test opsim specific database class setup/instantiation."""
        # Test tables were connected to.
        self.assertTrue(isinstance(self.oo.tables, dict))
        self.assertEqual(self.oo.dbTables['SummaryAllProps'][0], 'SummaryAllProps')
        # Test can override default table name/id keys if needed.
        oo = db.OpsimDatabase(database=self.database,
                              dbTables={'SummaryAllProps': ['ObsHistory', 'obsHistId']})
        self.assertEqual(oo.dbTables['SummaryAllProps'][0], 'ObsHistory')

    def testOpsimDbMetricData(self):
        """Test queries for sim data. """
        data = self.oo.fetchMetricData(['seeingFwhmEff', ], 'filter="r" and seeingFwhmEff<1.0')
        self.assertEqual(data.dtype.names, ('observationId', 'seeingFwhmEff'))
        self.assertTrue(data['seeingFwhmEff'].max() <= 1.0)

    @unittest.skip("14 Dec 2016 -- need to update fetchPropInfo for new OpSim")
    def testOpsimDbPropID(self):
        """Test queries for prop ID"""
        propids, propTags = self.oo.fetchPropInfo()
        self.assertTrue(len(propids.keys()) > 0)
        self.assertTrue(len(propTags['WFD']) > 0)
        self.assertTrue(len(propTags['DD']) >= 0)
        for w in propTags['WFD']:
            self.assertTrue(w in propids)
        for d in propTags['DD']:
            self.assertTrue(d in propids)

    def testOpsimDbFields(self):
        """Test queries for field data."""
        # Fetch field data for all fields.
        dataAll = self.oo.fetchFieldsFromFieldTable()
        self.assertEqual(dataAll.dtype.names, ('fieldId', 'ra', 'dec'))
        # Fetch field data for all fields requested by a particular propid.

    def testOpsimDbRunLength(self):
        """Test query for length of opsim run."""
        nrun = self.oo.fetchRunLength()
        self.assertEqual(nrun, 10.)

    def testOpsimDbSimName(self):
        """Test query for opsim name."""
        simname = self.oo.fetchOpsimRunName()
        self.assertTrue(isinstance(simname, str))
        self.assertEqual(simname, 'pontus_1150')

    def testOpsimDbSeeingColName(self):
        """Test query to pull out column name for seeing (seeing or finSeeing)."""
        seeingcol = self.oo.fetchSeeingColName()
        self.assertTrue(seeingcol, 'seeingFwhmEff')

    def testOpsimDbConfig(self):
        """Test generation of config data. """
        configsummary, configdetails = self.oo.fetchConfig()
        self.assertTrue(isinstance(configsummary, dict))
        self.assertTrue(isinstance(configdetails, dict))
        #  self.assertEqual(set(configsummary.keys()), set(['Version', 'RunInfo', 'Proposals', 'keyorder']))
        propids, proptags = self.oo.fetchPropInfo()
#        propidsSummary = []
#        for propname in configsummary['Proposals']:
#            if propname != 'keyorder':
#                propidsSummary.append(configsummary['Proposals'][propname]['propID'])
#        self.assertEqual(set(propidsSummary), set(propids))
#        out.printDict(configsummary, 'Summary')


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
