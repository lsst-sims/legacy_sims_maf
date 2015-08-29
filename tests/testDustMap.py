import unittest
import numpy as np
from lsst.sims.photUtils import EBV


class TestDustMap(unittest.TestCase):

    def testCreate(self):
        """ Test that we can create the dustmap"""

        # Test that we can load without error
        dustmap = EBV.EBVbase()
        dustmap.load_ebvMapNorth()
        dustmap.load_ebvMapSouth()

        # Test the interpolation
        ra = np.array([0.,0.,np.radians(30.)])
        dec = np.array([0.,np.radians(30.),np.radians(-30.)])

        ebvMap = dustmap.calculateEbv(equatorialCoordinates=np.array([ra,dec]),
                                      interp=False)

if __name__ == "__main__":
    unittest.main()
