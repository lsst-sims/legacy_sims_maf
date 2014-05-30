import unittest
from lsst.sims.maf.utils.opsimUtils import scaleStretchDesign

class TestOpsimUtils(unittest.TestCase):
    def testScale(self):
        for runLength in ([1, 5, 10]):
            nvisitDesign, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessDesign, seeingDesign = scaleStretchDesign(runLength)
        #XXX-need to decide what these values should be compared to.

if __name__ == "__main__":
    unittest.main()
