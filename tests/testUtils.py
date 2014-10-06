import unittest
import numpy as np

import lsst.sims.maf.utils as utils

class TestUtils(unittest.TestCase):
    
    def testStellarMags(self):

        stellarTypes = ['O','B','A','F','G','K','M']
        for st in stellarTypes:
            mags = utils.stellarMags(st)
            mags2 = utils.stellarMags(st, rmag=20.)
            for key in mags:
                self.assertLess(mags[key], mags2[key])
        
        self.assertRaises(ValueError, utils.stellarMags, 'ack')


if __name__ == "__main__":
    unittest.main()
