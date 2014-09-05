import numpy as np
import warnings
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
            


    def testHAStacker(self):
        """Test the Hour Angle stacker"""
        data = np.zeros(100, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = np.arange(100)/99.*np.pi*2
        stacker = stackers.HourAngleStacker()
        data = stacker.run(data)
        # Check that data is always wrapped
        assert(np.max(data['HA']) < 12.)
        assert(np.min(data['HA']) > -12.)

        # Check that HA is zero if lst == RA
        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data = stacker.run(data)
        assert(data['HA'] == 0.)

        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = 2.
        data['fieldRA'] = 2.
        data = stacker.run(data)
        assert(data['HA'] == 0.)

        # Check a value
        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = 0.
        data['fieldRA'] = np.pi/2.
        data = stacker.run(data)
        np.testing.assert_almost_equal(data['HA'], -6.)

        
                    
        
if __name__ == '__main__':

    unittest.main()
