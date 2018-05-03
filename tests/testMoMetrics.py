import numpy as np
import pandas as pd
import unittest
import lsst.sims.maf.metrics as metrics


class TestMoMetrics1(unittest.TestCase):

    def setUp(self):
        # Set up some ssoObs data to test the metrics on.
        # Note that ssoObs is a numpy recarray.
        # The expected set of columns in ssoObs is:
        cols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
                'visitExpTime', 'FWHMgeom', 'fiveSigmaDepth', 'solarElong',
                'delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt', 'phase', 'solarelon',
                'velocity', 'magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect']
        # And stackers will often add
        addCols = ['appMag', 'magLimit', 'snr', 'vis']

        # Test metrics using ssoObs for a particular object.
        times = np.array([0.1, 0.2, 0.3,
                          1.1, 1.3,
                          5.1,
                          7.1, 7.2, 7.3,
                          10.1, 10.2, 10.3,
                          13.1, 13.5], dtype='float')
        ssoObs = np.recarray([len(times)], dtype=([('time', '<f8'), ('ra', '<f8'), ('dec', '<f8'),
                                                ('appMag', '<f8'), ('expMJD', '<f8'), ('night', '<f8'), ('magLimit', '<f8'),
                                                ('SNR', '<f8'), ('vis', '<f8')]))

        ssoObs['time'] = times
        ssoObs['expMJD'] = times
        ssoObs['night'] = np.floor(times)
        ssoObs['ra'] = np.arange(len(times))
        ssoObs['dec'] = np.arange(len(times))
        ssoObs['appMag'] = np.zeros(len(times), dtype='float') + 24.0
        ssoObs['magLimit'] = np.zeros(len(times), dtype='float') + 25.0
        ssoObs['SNR'] = np.zeros(len(times), dtype='float') + 5.0
        ssoObs['vis'] = np.zeros(len(times), dtype='float') + 1
        ssoObs['vis'][0:5] = 0
        self.ssoObs = ssoObs
        self.orb = None
        self.Hval = 8.0

    def testNObsMetric(self):
        nObsMetric = metrics.NObsMetric(snrLimit=5)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs['time']))
        nObsMetric = metrics.NObsMetric(snrLimit=10)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, 0)
        nObsMetric = metrics.NObsMetric(snrLimit=None)
        nObs = nObsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs['time']) - 5)

    def testNObsNoSinglesMetric(self):
        nObsNoSinglesMetric = metrics.NObsNoSinglesMetric(snrLimit=5)
        nObs = nObsNoSinglesMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nObs, len(self.ssoObs['time'])-1)

    def testNNightsMetric(self):
        nNightsMetric = metrics.NNightsMetric(snrLimit=5)
        nnights = nNightsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nnights, len(np.unique(self.ssoObs['night'])))
        nNightsMetric = metrics.NNightsMetric(snrLimit=None)
        nnights = nNightsMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(nnights, len(np.unique(self.ssoObs['night']))-2)

    def testArcMetric(self):
        arcMetric = metrics.ObsArcMetric(snrLimit=5)
        arc = arcMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(arc, self.ssoObs['expMJD'][-1] - self.ssoObs['expMJD'][0])
        arcMetric = metrics.ObsArcMetric(snrLimit=None)
        arc = arcMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(arc, self.ssoObs['expMJD'][-1] - self.ssoObs['expMJD'][5])

    def tearDown(self):
        del self.ssoObs
        del self.orb
        del self.Hval

class TestDiscoveryMetrics(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(61331)
        # Set up some ssoObs data to test the metrics on.
        # Note that ssoObs is a numpy recarray.
        # The expected set of columns in ssoObs is:
        cols = ['expMJD', 'night', 'fieldRA', 'fieldDec', 'rotSkyPos', 'filter',
                'visitExpTime', 'FWHMgeom', 'fiveSigmaDepth', 'solarElong',
                'delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt', 'phase', 'solarelon',
                'velocity', 'magFilter', 'dmagColor', 'dmagTrail', 'dmagDetect']
        # And stackers will often add
        addCols = ['appMag', 'magLimit', 'snr', 'vis']

        # Test metrics using ssoObs for a particular object.
        times = np.array([0.1, 0.2, 0.9,
                          1.1, 1.3,
                          5.1,
                          7.1, 7.2, 7.5,
                          10.1, 10.2,
                          13.1, 13.5],
                          dtype='float')
        ssoObs = np.recarray([len(times)], dtype=([('time', '<f8'), ('ra', '<f8'), ('dec', '<f8'),
                                                   ('ecLon', '<f8'), ('ecLat', '<f8'), ('solarElong', '<f8'),
                                                   ('appMag', '<f8'), ('expMJD', '<f8'), ('night', '<f8'),
                                                   ('magLimit', '<f8'), ('velocity', '<f8'),
                                                   ('SNR', '<f8'), ('vis', '<f8'), ('magFilter', '<f8'),
                                                   ('fiveSigmaDepth', '<f8'), ('FWHMgeom', '<f8'),
                                                   ('visitExpTime', '<f8'), ('dmagDetect', '<f8')]))

        ssoObs['time'] = times
        ssoObs['expMJD'] = times
        ssoObs['night'] = np.floor(times)
        ssoObs['ra'] = np.arange(len(times))
        ssoObs['dec'] = np.arange(len(times)) + 5
        ssoObs['ecLon'] = ssoObs['ra'] + 10
        ssoObs['ecLat'] = ssoObs['dec'] + 20
        ssoObs['solarElong'] = ssoObs['ra'] + 30
        ssoObs['appMag'] = np.zeros(len(times), dtype='float') + 24.0
        ssoObs['magFilter'] = np.zeros(len(times), dtype='float') + 24.0
        ssoObs['fiveSigmaDepth'] = np.zeros(len(times), dtype='float') + 25.0
        ssoObs['dmagDetect'] = np.zeros(len(times), dtype='float')
        ssoObs['magLimit'] = np.zeros(len(times), dtype='float') + 25.0
        ssoObs['SNR'] = np.zeros(len(times), dtype='float') + 5.0
        ssoObs['vis'] = np.zeros(len(times), dtype='float') + 1
        ssoObs['vis'][0:5] = 0
        ssoObs['velocity'] = rng.rand(len(times))
        ssoObs['FWHMgeom'] = np.ones(len(times), 'float')
        ssoObs['visitExpTime'] = np.ones(len(times), 'float') * 24.0
        self.ssoObs = ssoObs
        self.orb = np.recarray([len(times)], dtype=([('H', '<f8')]))
        self.orb['H'] = np.zeros(len(times), dtype='float') + 8
        self.Hval = 8

    def testDiscoveryMetric(self):
        discMetric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0.0, tMax=0.3,
                                nNightsPerWindow=3, tWindow=9, snrLimit=5)
        metricValue = discMetric.run(self.ssoObs, self.orb, self.Hval)
        child = metrics.Discovery_N_ObsMetric(discMetric, i=0)
        nobs = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nobs, 8)
        child = metrics.Discovery_N_ObsMetric(discMetric, i=1)
        nobs = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nobs, 7)
        child = metrics.Discovery_N_ChancesMetric(discMetric)
        nchances = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(nchances, 2)
        child = metrics.Discovery_TimeMetric(discMetric, i=0)
        time = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(time, self.ssoObs['expMJD'][0])
        child = metrics.Discovery_TimeMetric(discMetric, i=1)
        time = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(time, self.ssoObs['expMJD'][3])
        child = metrics.Discovery_RADecMetric(discMetric, i=0)
        ra, dec = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(ra, 0)
        self.assertEqual(dec, 5)
        child = metrics.Discovery_EcLonLatMetric(discMetric, i=0)
        lon, lat, solarElong = child.run(self.ssoObs, self.orb, self.Hval, metricValue)
        self.assertEqual(lon, 10)
        self.assertEqual(lat, 25)

        discMetric2 = metrics.DiscoveryChancesMetric(nObsPerNight=2, tNight=0.3,
                                             nNightsPerWindow=3, tWindow=9, snrLimit=5)
        metricValue2 = discMetric2.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue2, 2)

        discMetric3 = metrics.MagicDiscoveryMetric(nObs=5, tWindow=2, snrLimit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 1)
        discMetric3 = metrics.MagicDiscoveryMetric(nObs=3, tWindow=1, snrLimit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 2)
        discMetric3 = metrics.MagicDiscoveryMetric(nObs=4, tWindow=4, snrLimit=5)
        magic = discMetric3.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(magic, 6)


    def testHighVelocityMetric(self):
        rng = np.random.RandomState(8123)
        velMetric = metrics.HighVelocityMetric(psfFactor=1.0, snrLimit=5)
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs['velocity'][0:2] = 1.5
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 2)
        velMetric = metrics.HighVelocityMetric(psfFactor=2.0, snrLimit=5)
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs['velocity'][0:2] = rng.rand(1)

    def testHighVelocityNightsMetric(self):
        velMetric = metrics.HighVelocityNightsMetric(psfFactor=1.0, nObsPerNight=1, snrLimit=5)
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 0)
        self.ssoObs['velocity'][0:2] = 1.5
        metricValue = velMetric.run(self.ssoObs, self.orb, self.Hval)
        self.assertEqual(metricValue, 1)


if __name__ == "__main__":
    unittest.main()
