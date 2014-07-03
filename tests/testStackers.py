import numpy as np
import lsst.sims.maf.stackers as stackers
import unittest

class TestMoreMetrics(unittest.TestCase):

    def testNormAirmass(self):
        """ test the normalized airmass stacker"""
        data = np.zeros(600, dtype=zip(['airmass','fieldDec'],[float,float]))
        data['airmass'] = np.random.rand(600)
        data['fieldDec'] = np.random.rand(600)*np.pi-np.pi/2.
        stacker = stackers.NormAirmassStacker()
        data = stacker.run(data)
        for i in np.arange(data.size):
            assert(data['normairmass'][i] <= data['airmass'][i])

    def testParallaxFactor(self):
        """test the parallax factor """
        pass


    def testRandomDither(self):
        """test the random dither pattern """
        data = np.zeros(600, dtype=zip(['fieldRA','fieldDec'],[float,float]))
        data['fieldRA'] = np.random.rand(600)*2*np.pi
        data['fieldDec'] = np.random.rand(600)*np.pi-np.pi/2.

        maxDither = .5
        stacker = stackers.RandomDitherStacker(maxDither=maxDither)
        data = stacker.run(data)
        for i in np.arange(data.size):
            assert( np.abs(data['fieldRA'][i]-data['randomRADither'][i])
                    <= np.radians(maxDither))
            assert( np.abs(data['fieldDec'][i]-data['randomDecDither'][i])
                    <= np.radians(maxDither))
            
    
if __name__ == '__main__':

    unittest.main()
