import matplotlib
matplotlib.use("Agg")
import os
import unittest
import numpy as np
import lsst.sims.maf.db as db
import lsst.sims.maf.utils.outputUtils as out
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.sims.utils.CodeUtilities import sims_clean_up
from builtins import str


class TestOpsimDb(unittest.TestCase):
    """Test opsim specific database class."""

    @classmethod
    def tearDownClass(cls):
        sims_clean_up()

    def setUp(self):
        self.database = os.path.join(getPackageDir('sims_data'), 'OpSimData',
                                     'opsimblitz1_1133_sqlite.db')
        self.oo = db.OpsimDatabase(database=self.database)

    def tearDown(self):
        del self.oo
        del self.database
        self.oo = None

    def testOpsimDbSetup(self):
        """Test opsim specific database class setup/instantiation."""
        # Test tables were connected to.
        self.assertTrue(isinstance(self.oo.tables, dict))
        self.assertEqual(self.oo.dbTables['Summary'][0], 'Summary')
        # Test can override default table name/id keys if needed.
        oo = db.OpsimDatabase(database=self.database,
                              dbTables={'Summary': ['ObsHistory', 'obsHistID']})
        self.assertEqual(oo.dbTables['Summary'][0], 'ObsHistory')

    def testOpsimDbMetricData(self):
        """Test queries for sim data. """
        data = self.oo.fetchMetricData(['finSeeing', ], 'filter="r" and finSeeing<1.0')
        self.assertEqual(data.dtype.names, ('obsHistID', 'finSeeing'))
        self.assertTrue(data['finSeeing'].max() <= 1.0)

    def testOpsimDbPropID(self):
        """Test queries for prop ID"""
        propids, propTags = self.oo.fetchPropInfo()
        self.assertTrue(len(list(propids.keys())) > 0)
        self.assertTrue(len(propTags['WFD']) > 0)
        self.assertTrue(len(propTags['DD']) > 0)
        for w in propTags['WFD']:
            self.assertTrue(w in propids)
        for d in propTags['DD']:
            self.assertTrue(d in propids)

    def testOpsimDbFields(self):
        """Test queries for field data."""
        # Fetch field data for all fields.
        dataAll = self.oo.fetchFieldsFromFieldTable()
        self.assertEqual(dataAll.dtype.names, ('fieldID', 'fieldRA', 'fieldDec'))
        # Fetch field data for all fields requested by a particular propid.
        propids, proptags = self.oo.fetchPropInfo()
        propid = list(propids.keys())[0]
        dataProp1 = self.oo.fetchFieldsFromFieldTable(propID=propid)
        # Fetch field data for all fields requested by all proposals.
        dataPropAll = self.oo.fetchFieldsFromFieldTable(propID=list(propids.keys()))
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

    def testOpsimDbConfig(self):
        """Test generation of config data. """
        configsummary, configdetails = self.oo.fetchConfig()
        self.assertTrue(isinstance(configsummary, dict))
        self.assertTrue(isinstance(configdetails, dict))
        self.assertEqual(set(configsummary.keys()), set(['Version', 'RunInfo', 'Proposals', 'keyorder']))
        propids, proptags = self.oo.fetchPropInfo()
        propidsSummary = []
        for propname in configsummary['Proposals']:
            if propname != 'keyorder':
                propidsSummary.append(configsummary['Proposals'][propname]['propID'])
        self.assertEqual(set(propidsSummary), set(propids))
        out.printDict(configsummary, 'Summary')


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
