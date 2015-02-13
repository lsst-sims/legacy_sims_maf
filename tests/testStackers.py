import matplotlib
matplotlib.use("Agg")
import numpy as np
import warnings
import lsst.sims.maf.stackers as stackers
import unittest
from lsst.sims.catalogs.generation.db import haversine

class TestStackerClasses(unittest.TestCase):


    def testNormAirmass(self):
        """
        Test the normalized airmass stacker.
        """
        data = np.zeros(600, dtype=zip(['airmass','fieldDec'],[float,float]))
        data['airmass'] = np.random.rand(600)
        data['fieldDec'] = np.random.rand(600)*np.pi-np.pi/2.
        stacker = stackers.NormAirmassStacker()
        data = stacker.run(data)
        for i in np.arange(data.size):
            self.assertLessEqual(data['normairmass'][i], data['airmass'][i])
        self.assertLess(np.min(data['normairmass']-data['airmass']), 0)

    def testParallaxFactor(self):
        """
        Test the parallax factor.
        """
        data = np.zeros(600, dtype=zip(['fieldRA','fieldDec', 'expMJD'],
                                       [float,float,float]))
        data['fieldRA'] = data['fieldRA']+.1
        data['fieldDec'] = data['fieldDec']-.1
        data['expMJD'] = np.arange(data.size)+49000.
        stacker = stackers.ParallaxFactorStacker()
        data = stacker.run(data)
        self.assertLess(max(np.abs(data['ra_pi_amp'])), 1.1)
        self.assertLess(max(np.abs(data['dec_pi_amp'])), 1.1)
        self.assertLess(np.max(data['ra_pi_amp']**2+data['dec_pi_amp']**2), 1.1)
        self.assertGreater(min(np.abs(data['ra_pi_amp'])), 0.)
        self.assertGreater(min(np.abs(data['dec_pi_amp'])), 0.)

    def testRandomDither(self):
        """
        Test the random dither pattern.
        """
        maxDither = .5
        data = np.zeros(600, dtype=zip(['fieldRA','fieldDec'],[float,float]))
        # Restrict dithers to area where wraparound is not a problem for comparisons.
        data['fieldRA'] = np.random.rand(600)*(np.pi) + np.pi/2.0
        data['fieldDec'] = np.random.rand(600)*np.pi/2.0 - np.pi/4.0
        stacker = stackers.RandomDitherStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['fieldRA'] - data['randomRADither'])*np.cos(data['fieldDec'])
        diffsdec = data['fieldDec'] - data['randomDecDither']
        # Check dithers within expected range.
        for diffra, diffdec, ra, dec in zip(diffsra, diffsdec, data['fieldRA'], data['fieldDec']):
            self.assertLessEqual(np.abs(diffra), np.radians(maxDither))
            self.assertLessEqual(np.abs(diffdec), np.radians(maxDither))
        distances = haversine(data['fieldRA'], data['fieldDec'], data['randomRADither'],
                              data['randomDecDither'])
        for d in distances:
            self.assertLessEqual(d, np.radians(maxDither))
        # Check dithers not all the same and go positive and negative.
        self.assertGreater(diffsra.max(), 0)
        self.assertGreater(diffsdec.max(), 0)
        self.assertLess(diffsra.min(), 0)
        self.assertLess(diffsdec.min(), 0)

    def testNightlyRandomDither(self):
        """
        Test the per-night random dither pattern.
        """
        maxDither = 0.5
        ndata = 600
        data = np.zeros(ndata, dtype=zip(['fieldRA', 'fieldDec', 'night'], [float, float, int]))
        data['fieldRA'] = np.random.rand(ndata)*(np.pi) + np.pi/2.0
        data['fieldDec'] = np.random.rand(ndata)*np.pi/2.0 - np.pi/4.0
        data['night'] = np.floor(np.random.rand(ndata)*10).astype('int')
        stacker = stackers.NightlyRandomDitherStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['fieldRA'] - data['nightlyRandomRADither'])*np.cos(data['fieldDec'])
        diffsdec = data['fieldDec'] - data['nightlyRandomDecDither']
        for diffra, diffdec, ra, dec in zip(diffsra, diffsdec, data['fieldRA'], data['fieldDec']):
            self.assertLessEqual(np.abs(diffra), np.radians(maxDither))
            self.assertLessEqual(np.abs(diffdec), np.radians(maxDither))
        distances = haversine(data['fieldRA'], data['fieldDec'], data['nightlyRandomRADither'],
                              data['nightlyRandomDecDither'])
        for d in distances:
            self.assertLessEqual(d, np.radians(maxDither))
        # Check dithers not all the same and go positive and negative.
        self.assertGreater(diffsra.max(), 0)
        self.assertGreater(diffsdec.max(), 0)
        self.assertLess(diffsra.min(), 0)
        self.assertLess(diffsdec.min(), 0)
        # Check that dithers on the same night are the same.
        nights = np.unique(data['night'])
        for n in nights:
            match = np.where(data['night'] == n)[0]
            rarange = np.abs(np.diff(diffsra[match]))
            decrange = np.abs(np.diff(diffsdec[match]))
            for r, d in zip(rarange, decrange):
                self.assertAlmostEqual(r, 0)
                self.assertAlmostEqual(d, 0)

    def testHAStacker(self):
        """Test the Hour Angle stacker"""
        data = np.zeros(100, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = np.arange(100)/99.*np.pi*2
        stacker = stackers.HourAngleStacker()
        data = stacker.run(data)
        # Check that data is always wrapped
        self.assertLess(np.max(data['HA']), 12.)
        self.assertGreater(np.min(data['HA']), -12.)
        # Check that HA is zero if lst == RA
        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data = stacker.run(data)
        self.assertEqual(data['HA'], 0.)
        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = 2.
        data['fieldRA'] = 2.
        data = stacker.run(data)
        self.assertEqual(data['HA'],0.)
        # Check a value
        data = np.zeros(1, dtype=zip(['lst','fieldRA'], [float,float]))
        data['lst'] = 0.
        data['fieldRA'] = np.pi/2.
        data = stacker.run(data)
        np.testing.assert_almost_equal(data['HA'], -6.)


if __name__ == '__main__':

    unittest.main()
