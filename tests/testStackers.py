from __future__ import print_function
from builtins import str
from builtins import zip
import numpy as np
import matplotlib
import warnings
import unittest
import lsst.utils.tests
import lsst.sims.maf.stackers as stackers
from lsst.sims.utils import _galacticFromEquatorial, calcLmstLast, Site, _altAzPaFromRaDec, \
    ObservationMetaData
from lsst.sims.survey.fields import FieldsDatabase

matplotlib.use("Agg")


class TestStackerClasses(unittest.TestCase):

    def testAddCols(self):
        """Test that we can add columns as expected.
        """
        data = np.zeros(90, dtype=list(zip(['alt'], [float])))
        data['alt'] = np.arange(0, 90)
        stacker = stackers.ZenithDistStacker(altCol='alt', degrees=True)
        newcol = stacker.colsAdded[0]
        # First - are the columns added if they are not there.
        data, cols_present = stacker._addStackerCols(data)
        self.assertEqual(cols_present, False)
        self.assertIn(newcol, data.dtype.names)
        # Next - if they are present, is that information passed back?
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data, cols_present = stacker._addStackerCols(data)
            self.assertEqual(cols_present, True)

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
        rng = np.random.RandomState(232)
        data = np.zeros(600, dtype=list(zip(
            ['airmass', 'fieldDec'], [float, float])))
        data['airmass'] = rng.random_sample(600)
        data['fieldDec'] = rng.random_sample(600) * np.pi - np.pi / 2.
        data['fieldDec'] = np.degrees(data['fieldDec'])
        stacker = stackers.NormAirmassStacker(degrees=True)
        data = stacker.run(data)
        for i in np.arange(data.size):
            self.assertLessEqual(data['normairmass'][i], data['airmass'][i])
        self.assertLess(np.min(data['normairmass'] - data['airmass']), 0)

    def testParallaxFactor(self):
        """
        Test the parallax factor.
        """

        data = np.zeros(600, dtype=list(zip(['fieldRA', 'fieldDec', 'observationStartMJD'],
                                       [float, float, float])))
        data['fieldRA'] = data['fieldRA'] + .1
        data['fieldDec'] = data['fieldDec'] - .1
        data['observationStartMJD'] = np.arange(data.size) + 49000.
        stacker = stackers.ParallaxFactorStacker(degrees=True)
        data = stacker.run(data)
        self.assertLess(max(np.abs(data['ra_pi_amp'])), 1.1)
        self.assertLess(max(np.abs(data['dec_pi_amp'])), 1.1)
        self.assertLess(
            np.max(data['ra_pi_amp']**2 + data['dec_pi_amp']**2), 1.1)
        self.assertGreater(min(np.abs(data['ra_pi_amp'])), 0.)
        self.assertGreater(min(np.abs(data['dec_pi_amp'])), 0.)

    def _tDitherRange(self, diffsra, diffsdec, ra, dec, maxDither):
        self.assertLessEqual(np.abs(diffsra).max(), maxDither)
        self.assertLessEqual(np.abs(diffsdec).max(), maxDither)
        offsets = np.sqrt(diffsra**2 + diffsdec**2)
        self.assertLessEqual(offsets.max(), maxDither)
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
                print(ni)
                m = np.where(dra_on_night > 0.0005)[0]
                print(diffsra[match][m])
                print(ra[match][m])
                print(dec[match][m])
                print(dra_on_night[m])
            self.assertAlmostEqual(dra_on_night.max(), 0)
            self.assertAlmostEqual(ddec_on_night.max(), 0)

    def testRandomDither(self):
        """
        Test the random dither pattern.
        """
        maxDither = .5
        data = np.zeros(600, dtype=list(zip(
            ['fieldRA', 'fieldDec'], [float, float])))
        # Set seed so the test is stable
        rng = np.random.RandomState(42)
        # Restrict dithers to area where wraparound is not a problem for
        # comparisons.
        data['fieldRA'] = np.degrees(rng.random_sample(600) * (np.pi) + np.pi / 2.0)
        data['fieldDec'] = np.degrees(rng.random_sample(600) * np.pi / 2.0 - np.pi / 4.0)
        stacker = stackers.RandomDitherFieldPerVisitStacker(
            maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['fieldRA'] - data['randomDitherFieldPerVisitRa']
                   ) * np.cos(np.radians(data['fieldDec']))
        diffsdec = data['fieldDec'] - data['randomDitherFieldPerVisitDec']
        # Check dithers within expected range.
        self._tDitherRange(diffsra, diffsdec, data[
                           'fieldRA'], data['fieldDec'], maxDither)

    def testRandomDitherPerNight(self):
        """
        Test the per-night random dither pattern.
        """
        maxDither = 0.5
        ndata = 600
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(ndata, dtype=list(zip(
            ['fieldRA', 'fieldDec', 'fieldId', 'night'], [float, float, int, int])))
        data['fieldRA'] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data['fieldDec'] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['fieldId'] = np.floor(rng.rand(ndata) * ndata)
        data['night'] = np.floor(rng.rand(ndata) * 10).astype('int')
        stacker = stackers.RandomDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (np.radians(data['fieldRA']) - np.radians(data['randomDitherPerNightRa'])
                   ) * np.cos(np.radians(data['fieldDec']))
        diffsdec = np.radians(data['fieldDec']) - np.radians(data['randomDitherPerNightDec'])
        self._tDitherRange(diffsra, diffsdec, data[
                           'fieldRA'], data['fieldDec'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['fieldRA'], data[
                              'fieldDec'], data['night'])

    def testSpiralDitherPerNight(self):
        """
        Test the per-night spiral dither pattern.
        """
        maxDither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(ndata, dtype=list(zip(
            ['fieldRA', 'fieldDec', 'fieldId', 'night'], [float, float, int, int])))
        data['fieldRA'] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data['fieldRA'] = np.zeros(ndata) + np.pi / 2.0
        data['fieldDec'] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['fieldDec'] = np.zeros(ndata)
        data['fieldId'] = np.floor(rng.rand(ndata) * ndata)
        data['night'] = np.floor(rng.rand(ndata) * 20).astype('int')
        stacker = stackers.SpiralDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['fieldRA'] - data['spiralDitherPerNightRa']
                   ) * np.cos(np.radians(data['fieldDec']))
        diffsdec = data['fieldDec'] - data['spiralDitherPerNightDec']
        self._tDitherRange(diffsra, diffsdec, data[
                           'fieldRA'], data['fieldDec'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['fieldRA'], data[
                              'fieldDec'], data['night'])

    def testHexDitherPerNight(self):
        """
        Test the per-night hex dither pattern.
        """
        maxDither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(ndata, dtype=list(zip(
            ['fieldRA', 'fieldDec', 'fieldId', 'night'], [float, float, int, int])))
        data['fieldRA'] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data['fieldDec'] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data['fieldId'] = np.floor(rng.rand(ndata) * ndata)
        data['night'] = np.floor(rng.rand(ndata) * 217).astype('int')
        stacker = stackers.HexDitherPerNightStacker(maxDither=maxDither)
        data = stacker.run(data)
        diffsra = (data['fieldRA'] - data['hexDitherPerNightRa']
                   ) * np.cos(np.radians(data['fieldDec']))
        diffsdec = data['fieldDec'] - data['hexDitherPerNightDec']
        self._tDitherRange(diffsra, diffsdec, data[
                           'fieldRA'], data['fieldDec'], maxDither)
        # Check that dithers on the same night are the same.
        self._tDitherPerNight(diffsra, diffsdec, data['fieldRA'],
                              data['fieldDec'], data['night'])

    def testRandomRotDitherPerFilterChangeStacker(self):
        """
        Test the rotational dither stacker.
        """
        maxDither = 90
        filt = np.array(['r', 'r', 'r', 'g', 'g', 'g', 'r', 'r'])
        rotTelPos = np.array([0, 0, 1, 0, .5, 1, 0, 180], float)
        odata = np.zeros(len(filt), dtype=list(zip(['filter', 'rotTelPos'], [(np.str_, 1), float])))
        odata['filter'] = filt
        odata['rotTelPos'] = rotTelPos
        stacker = stackers.RandomRotDitherPerFilterChangeStacker(maxDither=maxDither, degrees=True,
                                                                 randomSeed=99)
        data = stacker.run(odata)
        randomDithers = data['randomDitherPerFilterChangeRotTelPos']
        rotOffsets = rotTelPos - randomDithers
        self.assertEqual(rotOffsets[0], 0)
        offsetChanges = np.where(rotOffsets[1:] != rotOffsets[:-1])[0]
        filtChanges = np.where(filt[1:] != filt[:-1])[0]
        # Don't count last offset change because this was just value to force min/max limit.
        np.testing.assert_array_equal(offsetChanges[:-1], filtChanges)
        self.assertLessEqual(randomDithers.max(), 90.0)
        stacker = stackers.RandomRotDitherPerFilterChangeStacker(maxDither=maxDither,
                                                                 degrees=True, maxRotAngle = 30,
                                                                 randomSeed=19231)
        data = stacker.run(odata)
        randomDithers = data['randomDitherPerFilterChangeRotTelPos']
        self.assertEqual(randomDithers.max(), 30.0)

    def testHAStacker(self):
        """Test the Hour Angle stacker"""
        data = np.zeros(100, dtype=list(zip(['observationStartLST', 'fieldRA'], [float, float])))
        data['observationStartLST'] = np.arange(100) / 99. * np.pi * 2
        stacker = stackers.HourAngleStacker(degrees=True)
        data = stacker.run(data)
        # Check that data is always wrapped
        self.assertLess(np.max(data['HA']), 12.)
        self.assertGreater(np.min(data['HA']), -12.)
        # Check that HA is zero if lst == RA
        data = np.zeros(1, dtype=list(zip(['observationStartLST', 'fieldRA'], [float, float])))
        data = stacker.run(data)
        self.assertEqual(data['HA'], 0.)
        data = np.zeros(1, dtype=list(zip(['observationStartLST', 'fieldRA'], [float, float])))
        data['observationStartLST'] = 20.
        data['fieldRA'] = 20.
        data = stacker.run(data)
        self.assertEqual(data['HA'], 0.)
        # Check a value
        data = np.zeros(1, dtype=list(zip(['observationStartLST', 'fieldRA'], [float, float])))
        data['observationStartLST'] = 0.
        data['fieldRA'] = np.degrees(np.pi / 2.)
        data = stacker.run(data)
        np.testing.assert_almost_equal(data['HA'], -6.)

    def testPAStacker(self):
        """ Test the parallacticAngleStacker"""
        data = np.zeros(100, dtype=list(zip(
            ['observationStartMJD', 'fieldDec', 'fieldRA', 'observationStartLST'], [float] * 4)))
        data['observationStartMJD'] = np.arange(100) * .2 + 50000
        site = Site(name='LSST')
        data['observationStartLST'], last = calcLmstLast(data['observationStartMJD'], site.longitude_rad)
        data['observationStartLST'] = data['observationStartLST']*180./12.
        stacker = stackers.ParallacticAngleStacker(degrees=True)
        data = stacker.run(data)
        # Check values are in good range
        assert(data['PA'].max() <= 180)
        assert(data['PA'].min() >= -180)

        # Check compared to the util
        check_pa = []
        ras = np.radians(data['fieldRA'])
        decs = np.radians(data['fieldDec'])
        for ra, dec, mjd in zip(ras, decs, data['observationStartMJD']):
            alt, az, pa = _altAzPaFromRaDec(ra, dec,
                                            ObservationMetaData(mjd=mjd, site=site))

            check_pa.append(pa)
        check_pa = np.degrees(check_pa)
        np.testing.assert_array_almost_equal(data['PA'], check_pa, decimal=0)

    def testFilterColorStacker(self):
        """Test the filter color stacker."""
        data = np.zeros(60, dtype=list(zip(['filter'], ['<U1'])))
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
        data = np.zeros(600, dtype=list(zip(['filter'], ['<U1'])))
        data['filter'] = 'q'
        self.assertRaises(IndexError, stacker.run, data)

    def testGalacticStacker(self):
        """
        Test the galactic coordinate stacker
        """
        ra, dec = np.degrees(np.meshgrid(np.arange(0, 2. * np.pi, 0.1),
                                         np.arange(-np.pi, np.pi, 0.1)))
        ra = np.ravel(ra)
        dec = np.ravel(dec)
        data = np.zeros(ra.size, dtype=list(zip(['ra', 'dec'], [float] * 2)))
        data['ra'] += ra
        data['dec'] += dec
        s = stackers.GalacticStacker(raCol='ra', decCol='dec')
        newData = s.run(data)
        expectedL, expectedB = _galacticFromEquatorial(np.radians(ra), np.radians(dec))
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

    def testOpSimFieldStacker(self):
        """
        Test the OpSimFieldStacker
        """
        rng = np.random.RandomState(812351)
        s = stackers.OpSimFieldStacker(raCol='ra', decCol='dec', degrees=False)

        # First sanity check. Make sure the center of the fields returns the right field id
        opsim_fields_db = FieldsDatabase()

        # Returned RA/Dec coordinates in degrees
        field_id, ra, dec = opsim_fields_db.get_id_ra_dec_arrays("select * from Field;")

        data = np.array(list(zip(np.radians(ra),
                                 np.radians(dec))),
                        dtype=list(zip(['ra', 'dec'], [float, float])))
        new_data = s.run(data)

        np.testing.assert_array_equal(field_id, new_data['fieldId'])

        # Cherry picked a set of coordinates that should belong to a certain list of fields. These coordinates
        # are not exactly at the center of fields, but close enough that they should be classified as belonging to
        # them.
        ra_inside_2548 = (10. + 1. / 60 + 6.59 / 60. / 60.) * np.pi / 12.  # 10:01:06.59
        dec_inside_2548 = np.radians(-1. * (2. + 8. / 60. + 27.6 / 60. / 60.))  # -02:08:27.6

        ra_inside_8 = (8. + 49. / 60 + 19.83 / 60. / 60.) * np.pi / 12.  # 08:49:19.83
        dec_inside_8 = np.radians(-1. * (85. + 19. / 60. + 04.7 / 60. / 60.))  # -85:19:04.7

        ra_inside_1253 = (9. + 16. / 60 + 13.67 / 60. / 60.) * np.pi / 12.  # 09:16:13.67
        dec_inside_1253 = np.radians(-1. * (30. + 23. / 60. + 41.4 / 60. / 60.))  # -30:23:41.4

        data = np.zeros(3, dtype=list(zip(['ra', 'dec'], [float, float])))
        field_id = np.array([2548, 8, 1253], dtype=int)
        data['ra'] = np.array([ra_inside_2548, ra_inside_8, ra_inside_1253])
        data['dec'] = np.array([dec_inside_2548, dec_inside_8, dec_inside_1253])

        new_data = s.run(data)

        np.testing.assert_array_equal(field_id, new_data['fieldId'])

        # Now let's generate a set of random coordinates and make sure they are all assigned a fieldID.
        data = np.array(list(zip(rng.rand(600) * 2. * np.pi,
                                 rng.rand(600) * np.pi - np.pi / 2.)),
                        dtype=list(zip(['ra', 'dec'], [float, float])))

        new_data = s.run(data)

        self.assertTrue(np.all(new_data['fieldId'] > 0))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
