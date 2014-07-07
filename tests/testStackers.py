import numpy as np
import lsst.sims.maf.stackers as stackers
import unittest

class TestStackerClasses(unittest.TestCase):

    def testNormAirmass(self):
        """ test the normalized airmass stacker"""
        data = np.zeros(600, dtype=zip(['airmass','fieldDec'],[float,float]))
        data['airmass'] = np.random.rand(600)
        data['fieldDec'] = np.random.rand(600)*np.pi-np.pi/2.
        stacker = stackers.NormAirmassStacker()
        data = stacker.run(data)
        for i in np.arange(data.size):
            assert(data['normairmass'][i] <= data['airmass'][i])
        assert(np.min(data['normairmass']-data['airmass']) < 0)

    def testParallaxFactor(self):
        """test the parallax factor """
        data = np.zeros(600, dtype=zip(['fieldRA','fieldDec', 'expMJD'],
                                       [float,float,float]))
        data['fieldRA'] = data['fieldRA']+.1
        data['fieldDec'] = data['fieldDec']-.1
        data['expMJD'] = np.arange(data.size)+49000.
        stacker = stackers.ParallaxFactorStacker()
        data = stacker.run(data)
        assert(max(np.abs(data['ra_pi_amp'])) < 1.1)
        assert(max(np.abs(data['dec_pi_amp'])) < 1.1)
        assert( np.max(data['ra_pi_amp']**2+data['dec_pi_amp']**2) < 1.1)
        assert(min(np.abs(data['ra_pi_amp'])) > 0.)
        assert(min(np.abs(data['dec_pi_amp'])) > 0.)

    def testRandomDither(self):
        """test the random dither pattern """
        maxDither = .5
        data = np.zeros(600, dtype=zip(['fieldRA','fieldDec'],[float,float]))
        data['fieldRA'] = np.random.rand(600)*(2*np.pi - 2.*maxDither)+maxDither
        data['fieldDec'] = np.random.rand(600)*np.pi-np.pi/2.
        stacker = stackers.RandomDitherStacker(maxDither=maxDither)
        data = stacker.run(data)
        for i in np.arange(data.size):
            diff = np.abs(data['fieldRA'][i]-data['randomRADither'][i])
            assert( diff <= np.radians(maxDither))
            assert( np.abs(data['fieldDec'][i]-data['randomDecDither'][i])
                    <= np.radians(maxDither))
        assert(np.min(data['fieldDec']-data['randomDecDither']) < 0)
        assert(np.max(data['fieldDec']-data['randomDecDither']) > 0)
            
    
if __name__ == '__main__':

    unittest.main()
