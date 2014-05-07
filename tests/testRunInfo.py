import unittest
import os
from lsst.sims.maf.utils.runInfo import fetchPropIDs,fetchNFields, scaleStretchDesign

class TestDB(unittest.TestCase):
    def setUp(self):
        filepath = os.environ['SIMS_MAF_DIR']+'/examples/'
        self.dbAddress = 'sqlite:///'+filepath+'full_small_sqlite.db'

    def testfetchProp(self):
        propIDs,wfdIDs,ddIDs = fetchPropIDs(self.dbAddress)
        assert(propIDs == [55])
        assert(wfdIDs == [55])
        assert(ddIDs == [])

    def testfetchN(self):
        nFields = fetchNFields(self.dbAddress, [55])
        assert(nFields == [3425])

    def testScale(self):
        nvisitDesign, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessDesign, seeingDesign = scaleStretchDesign(self.dbAddress)
        #XXX-need to decide what these values should be compared to.

if __name__ == "__main__":
    unittest.main()
