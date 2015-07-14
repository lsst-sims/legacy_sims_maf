import matplotlib
matplotlib.use("Agg")
import numpy as np
import warnings
import lsst.sims.maf.stackers as stackers
from lsst.sims.utils import galacticFromEquatorial
import unittest

class TestStackerClasses(unittest.TestCase):

    def testEQ(self):
        """
        Test that stackers can be compared
        """
        s1 = stackers.ParallaxFactorStacker()
        s2 = stackers.ParallaxFactorStacker()
        assert(s1 == s2)

        s1 = stackers.RandomDitherStacker()
        s2 = stackers.RandomDitherStacker()
        assert(s1 == s2)

        # Test if they have numpy array atributes
        s1.ack = np.arange(10)
        s2.ack = np.arange(10)
        assert(s1 == s2)

        # Change the array and test
        s1.ack += 1
        assert(s1 != s2)

        s2 = stackers.RandomDitherStacker(decCol='blah')
        assert(s1 != s2)

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
        # Set seed so the test is stable
        np.random.seed(42)
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
        # Set seed so the test is stable
        np.random.seed(42)
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

    def testFilterColorStacker(self):
        """Test the filter color stacker."""
        data = np.zeros(60, dtype=zip(['filter'],['|S1']))
        data['filter'][0:10] = 'u'
        data['filter'][10:20] = 'g'
        data['filter'][20:30] = 'r'
        data['filter'][30:40] = 'i'
        data['filter'][40:50] = 'z'
        data['filter'][50:60] = 'y'
        stacker = stackers.FilterColorStacker()
        data = stacker.run(data)
        # Check if re-run stacker raises warning (adding column twice).
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = stacker.run(data)
            assert len(w) > 1
            assert "already present in simData" in str(w[-1].message)
        # Check if use non-recognized filter raises exception.
        data = np.zeros(600, dtype=zip(['filter'],['|S1']))
        data['filter'] = 'q'
        self.assertRaises(IndexError, stacker.run, data)

    def testGalacticStacker(self):
        """
        Test the galactic coordinate stacker
        """
        ra,dec = np.meshgrid(np.arange(0,2.*np.pi, 0.1), np.arange(-np.pi,np.pi, 0.1) )
        ra = np.ravel(ra)
        dec=np.ravel(dec)
        data = np.zeros(ra.size, dtype=zip(['ra','dec'],[float]*2))
        data['ra'] += ra
        data['dec'] += dec
        stacker = stackers.GalacticStacker(raCol='ra',decCol='dec')
        newData = stacker.run(data)
        expectedL, expectedB = galacticFromEquatorial(ra, dec)
        np.array_equal(newData['gall'], expectedL )
        np.array_equal(newData['galb'], expectedB)

        # Check that we have all the quadrants populated
        q1 = np.where((newData['gall'] < np.pi) & (newData['galb'] < 0.) )[0]
        q2 = np.where((newData['gall'] < np.pi) & (newData['galb'] > 0.) )[0]
        q3 = np.where((newData['gall'] > np.pi) & (newData['galb'] < 0.) )[0]
        q4 = np.where((newData['gall'] > np.pi) & (newData['galb'] > 0.) )[0]
        assert(q1.size > 0)
        assert(q2.size > 0)
        assert(q3.size > 0)
        assert(q4.size > 0)


if __name__ == '__main__':

    unittest.main()
