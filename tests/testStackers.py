import numpy as np
import matplotlib
import warnings
import unittest
import lsst.utils.tests
import lsst.sims.maf.stackers as stackers
from lsst.sims.utils import _galacticFromEquatorial, calcLmstLast, Site, _altAzPaFromRaDec, ObservationMetaData

matplotlib.use("Agg")


class TestStackerClasses(unittest.TestCase):

    def testEQ(self):
        """
        Test that stackers can be compared
        """
        s1 = stackers.ParallaxFactorStacker()
        s2 = stackers.ParallaxFactorStacker()
        assert(s1 == s2)

        s1 = stackers.RandomDitherFieldPerVisitStacker()
        s2 = stackers.RandomDitherFieldPerVisitStacker()
        assert(s1 == s2)

        # Test if they have numpy array atributes
        s1.ack = np.arange(10)
        s2.ack = np.arange(10)
        assert(s1 == s2)

        # Change the array and test
        s1.ack += 1
        assert(s1 != s2)

        s2 = stackers.RandomDitherFieldPerVisitStacker(decCol='blah')
        assert(s1 != s2)

    def testNormAirmass(self):
        """
        Test the normalized airmass stacker.
        """
        data = np.zeros(600, dtype=zip(
            ['airmass', 'dec_rad'], [float, float]))
        data['airmass'] = np.random.rand(600)
        data['dec_rad'] = np.random.rand(600) * np.pi - np.pi / 2.
        stacker = stackers.NormAirmassStacker()
        data = stacker.run(data)
        for i in np.arange(data.size):
            self.assertLessEqual(data['normairmass'][i], data['airmass'][i])
        self.assertLess(np.min(data['normairmass'] - data['airmass']), 0)

    def testParallaxFactor(self):
        """
        Test the parallax factor.
        """
        data = np.zeros(600, dtype=zip(['ra_rad', 'dec_rad', 'observationStartMJD'],
                                       [float, float, float]))
        data['ra_rad'] = data['ra_rad'] + .1
        data['dec_rad'] = data['dec_rad'] - .1
        data['observationStartMJD'] = np.arange(data.size) + 49000.
        stacker = stackers.ParallaxFactorStacker()
        data = stacker.run(data)
        self.assertLess(max(np.abs(data['ra_pi_amp'])), 1.1)
        self.assertLess(max(np.abs(data['dec_pi_amp'])), 1.1)
        self.assertLess(
            np.max(data['ra_pi_amp']**2 + data['dec_pi_amp']**2), 1.1)
        self.assertGreater(min(np.abs(data['ra_pi_amp'])), 0.)
        self.assertGreater(min(np.abs(data['dec_pi_amp'])), 0.)

    def _tDitherRange(self, diffsra, diffsdec, ra, dec, maxDither):
        self.assertTrue(np.all(np.abs(diffsra) <= np.radians(maxDither)))
        self.assertTrue(np.all(np.abs(diffsdec) <= np.radians(maxDither)))
        offsets = np.sqrt(diffsra**2 + diffsdec**2)
        self.assertLessEqual(offsets.max(), np.radians(maxDither))
        self.assertGreater(diffsra.max(), 0)
        self.assertGreater(diffsdec.max(), 0)
        self.assertLess(diffsra.min(), 0)
        self.assertLess(diffsdec.min(), 0)

    def _tDitherPerNight(self, diffsra, diffsdec, ra, dec, nights):
        n = np.unique(nights)
        for ni in n:
            match = np.where(nights == ni)[0]
            dra_on_night = np.abs(np.diff(diffsra[match]))
            ddec_on_night = np.abs(np.diff(diffsdec[match]))
            if dra_on_night.max() > 0.0005:
                print ni
                m = np.where(dra_on_night > 0.0005)[0]
                print diffsra[match][m]
                print ra[match][m]
                print dec[match][m]
                print dra_on_night[m]
            self.assertAlmostEqual(dra_on_night.max(), 0)
            self.assertAlmostEqual(ddec_on_night.max(), 0)

    def testRandomDither(self):
        """
        Test the random dither pattern.
        """
        maxDither = .5
        data = np.zeros(600, dtype=zip(
            ['ra_rad', 'dec_rad'], [float, float]))
        # Set seed so the test is stable
        np.random.seed(42)
        # Restrict dithers to area where wraparound is not a problem for
        # comparisons.
        data['ra_rad'] = np.random.rand(600) * (np.pi) + np.pi / 2.0
        data['dec_rad'] = np.random.rand(600) * np.pi / 2.0 - np.pi / 4.0
        stacker = stackers.RandomDitherFieldPerVisitStacker(
            maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['ra_rad'] - data['randomDitherFieldPerVisitRa']
                   ) * np.cos(data['dec_rad'])
        diffsdec = data['dec_rad'] - data['randomDitherFieldPerVisitDec']
        # Check dithers within expected range.
        self._tDitherRange(diffsra, diffsdec, data[
                           'ra_rad'], data['dec_rad'], maxDither)

    def testRandomDitherPerNight(self):
        """
        Test the per-night random dither pattern.
        """
        maxDither = 0.5
        ndata = 600
        # Set seed so the test is stable
        np.random.seed(42)
        data = np.zeros(ndata, dtype=zip(
            ['ra_rad', 'dec_rad', 'fieldId', 'night'], [float, float, int, int]))
        data['ra_rad'] = np.random.rand(ndata) * (np.pi) + np.pi / 2.0
        data['dec_rad'] = np.random.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['fieldId'] = np.floor(np.random.rand(ndata) * ndata)
        data['night'] = np.floor(np.random.rand(ndata) * 10).astype('int')
        stacker = stackers.RandomDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['ra_rad'] - data['randomDitherPerNightRa']
                   ) * np.cos(data['dec_rad'])
        diffsdec = data['dec_rad'] - data['randomDitherPerNightDec']
        self._tDitherRange(diffsra, diffsdec, data[
                           'ra_rad'], data['dec_rad'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['ra_rad'], data[
                              'dec_rad'], data['night'])

    def testSpiralDitherPerNight(self):
        """
        Test the per-night spiral dither pattern.
        """
        maxDither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        np.random.seed(42)
        data = np.zeros(ndata, dtype=zip(
            ['ra_rad', 'dec_rad', 'fieldId', 'night'], [float, float, int, int]))
        data['ra_rad'] = np.random.rand(ndata) * (np.pi) + np.pi / 2.0
        data['ra_rad'] = np.zeros(ndata) + np.pi / 2.0
        data['dec_rad'] = np.random.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['dec_rad'] = np.zeros(ndata)
        data['fieldId'] = np.floor(np.random.rand(ndata) * ndata)
        data['night'] = np.floor(np.random.rand(ndata) * 20).astype('int')
        stacker = stackers.SpiralDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['ra_rad'] - data['spiralDitherPerNightRa']
                   ) * np.cos(data['dec_rad'])
        diffsdec = data['dec_rad'] - data['spiralDitherPerNightDec']
        self._tDitherRange(diffsra, diffsdec, data[
                           'ra_rad'], data['dec_rad'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['ra_rad'], data[
                              'dec_rad'], data['night'])

    def testHexDitherPerNight(self):
        """
        Test the per-night hex dither pattern.
        """
        maxDither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        np.random.seed(42)
        data = np.zeros(ndata, dtype=zip(
            ['ra_rad', 'dec_rad', 'fieldId', 'night'], [float, float, int, int]))
        data['ra_rad'] = np.random.rand(ndata) * (np.pi) + np.pi / 2.0
        data['dec_rad'] = np.random.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['fieldId'] = np.floor(np.random.rand(ndata) * ndata)
        data['night'] = np.floor(np.random.rand(ndata) * 217).astype('int')
        stacker = stackers.HexDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['ra_rad'] - data['hexDitherPerNightRa']
                   ) * np.cos(data['dec_rad'])
        diffsdec = data['dec_rad'] - data['hexDitherPerNightDec']
        self._tDitherRange(diffsra, diffsdec, data[
                           'ra_rad'], data['dec_rad'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['ra_rad'],
                              data['dec_rad'], data['night'])

    def testHAStacker(self):
        """Test the Hour Angle stacker"""
        data = np.zeros(100, dtype=zip(['lst', 'ra_rad'], [float, float]))
        data['lst'] = np.arange(100) / 99. * np.pi * 2
        stacker = stackers.HourAngleStacker()
        data = stacker.run(data)
        # Check that data is always wrapped
        self.assertLess(np.max(data['HA']), 12.)
        self.assertGreater(np.min(data['HA']), -12.)
        # Check that HA is zero if lst == RA
        data = np.zeros(1, dtype=zip(['lst', 'ra_rad'], [float, float]))
        data = stacker.run(data)
        self.assertEqual(data['HA'], 0.)
        data = np.zeros(1, dtype=zip(['lst', 'ra_rad'], [float, float]))
        data['lst'] = 2.
        data['ra_rad'] = 2.
        data = stacker.run(data)
        self.assertEqual(data['HA'], 0.)
        # Check a value
        data = np.zeros(1, dtype=zip(['lst', 'ra_rad'], [float, float]))
        data['lst'] = 0.
        data['ra_rad'] = np.pi / 2.
        data = stacker.run(data)
        np.testing.assert_almost_equal(data['HA'], -6.)

    def testPAStacker(self):
        """ Test the parallacticAngleStacker"""
        data = np.zeros(100, dtype=zip(
            ['expMJD', 'dec_rad', 'ra_rad', 'lst'], [float] * 4))
        data['expMJD'] = np.arange(100) * .2 + 50000
        site = Site(name='LSST')
        data['lst'], last = calcLmstLast(data['expMJD'], site.longitude_rad)
        data['lst'] = data['lst']*np.pi/12.
        stacker = stackers.ParallacticAngleStacker()
        data = stacker.run(data)
        # Check values are in good range
        assert(data['PA'].max() <= np.pi)
        assert(data['PA'].min() >= -np.pi)

        # Check compared to the util
        check_pa = []
        for ra, dec, mjd in zip(data['ra_rad'], data['dec_rad'], data['expMJD']):
            alt, az, pa = _altAzPaFromRaDec(ra, dec,
                                            ObservationMetaData(mjd=mjd, site=site))

            check_pa.append(pa)
        np.testing.assert_array_almost_equal(data['PA'], check_pa, decimal=2)

    def testFilterColorStacker(self):
        """Test the filter color stacker."""
        data = np.zeros(60, dtype=zip(['filter'], ['|S1']))
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
        data = np.zeros(600, dtype=zip(['filter'], ['|S1']))
        data['filter'] = 'q'
        self.assertRaises(IndexError, stacker.run, data)

    def testGalacticStacker(self):
        """
        Test the galactic coordinate stacker
        """
        ra, dec = np.meshgrid(np.arange(0, 2. * np.pi, 0.1),
                              np.arange(-np.pi, np.pi, 0.1))
        ra = np.ravel(ra)
        dec = np.ravel(dec)
        data = np.zeros(ra.size, dtype=zip(['ra', 'dec'], [float] * 2))
        data['ra'] += ra
        data['dec'] += dec
        s = stackers.GalacticStacker(raCol='ra', decCol='dec')
        newData = s.run(data)
        expectedL, expectedB = _galacticFromEquatorial(ra, dec)
        np.testing.assert_array_equal(newData['gall'], expectedL)
        np.testing.assert_array_equal(newData['galb'], expectedB)

        # Check that we have all the quadrants populated
        q1 = np.where((newData['gall'] < np.pi) & (newData['galb'] < 0.))[0]
        q2 = np.where((newData['gall'] < np.pi) & (newData['galb'] > 0.))[0]
        q3 = np.where((newData['gall'] > np.pi) & (newData['galb'] < 0.))[0]
        q4 = np.where((newData['gall'] > np.pi) & (newData['galb'] > 0.))[0]
        assert(q1.size > 0)
        assert(q2.size > 0)
        assert(q3.size > 0)
        assert(q4.size > 0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
