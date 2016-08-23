import matplotlib
matplotlib.use("Agg")
import numpy as np
import healpy as hp
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.utils.tests


class TestSummaryMetrics(unittest.TestCase):

    def testTableFractionMetric(self):
        """Test the table summary metric """
        metricdata1 = np.arange(0, 1.5, .02)
        metricdata = np.array(zip(metricdata1), dtype=[('testdata', 'float')])
        for nbins in [10, 20, 5]:
            metric = metrics.TableFractionMetric('testdata', nbins=nbins)
            table = metric.run(metricdata)
            self.assertEqual(len(table), nbins+3)
            self.assertEqual(table['value'][0], np.size(np.where(metricdata1 == 0)[0]))
            self.assertEqual(table['value'][-1], np.size(np.where(metricdata1 > 1)[0]))
            self.assertEqual(table['value'][-2], np.size(np.where(metricdata1 == 1)[0]))
            self.assertEqual(table['value'].sum(), metricdata1.size)

    def testIdentityMetric(self):
        """Test identity metric."""
        dv = np.arange(0, 10, .5)
        dv = np.array(zip(dv), dtype=[('testdata', 'float')])
        testmetric = metrics.IdentityMetric('testdata')
        np.testing.assert_equal(testmetric.run(dv), dv['testdata'])

    def testfONv(self):
        """
        Test the fONv metric.
        """
        nside = 128
        metric = metrics.fONv(col='ack', nside=nside, Nvisit=825, Asky=18000.)
        npix = hp.nside2npix(nside)
        names = ['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names, types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        slicePoint = {'sid': 0}
        result1 = metric.run(data, slicePoint)
        deginsph = 129600./np.pi
        np.testing.assert_almost_equal(result1*18000., deginsph)
        data['blah'][:data.size/2] = 0
        result2 = metric.run(data, slicePoint)
        np.testing.assert_almost_equal(result2*18000., deginsph/2.)

    def testfOArea(self):
        """Test fOArea metric."""
        nside = 128
        metric = metrics.fOArea(col='ack', nside=nside, Nvisit=825, Asky=18000.)
        npix = hp.nside2npix(nside)
        names = ['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names, types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        slicePoint = {'sid': 0}
        result1 = metric.run(data, slicePoint)
        np.testing.assert_almost_equal(result1*825, 826)
        data['blah'][:data.size/2] = 0
        result2 = metric.run(data, slicePoint)

    def testNormalizeMetric(self):
        """Test normalize metric."""
        data = np.ones(10, dtype=zip(['testcol'], ['float']))
        metric = metrics.NormalizeMetric(col='testcol', normVal=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float)/5.5)

    def testZeropointMetric(self):
        """Test zeropoint metric."""
        data = np.ones(10, dtype=zip(['testcol'], ['float']))
        metric = metrics.ZeropointMetric(col='testcol', zp=5.5)
        result = metric.run(data)
        np.testing.assert_equal(result, np.ones(10, float)+5.5)

    def testTotalPowerMetric(self):
        nside = 128
        data = np.ones(12*nside**2, dtype=zip(['testcol'], ['float']))
        metric = metrics.TotalPowerMetric(col='testcol')
        result = metric.run(data)
        np.testing.assert_equal(result, 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
