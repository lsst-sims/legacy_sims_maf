import numpy as np
import healpy as hp
import unittest
import lsst.sims.maf.metrics as metrics


class TestSummaryMetrics(unittest.TestCase):

    def testTableFractionMetric(self):
        """Test the table summary metric """
        metricdata = np.arange(0, 1, .02)
        metricdata = np.array(zip(metricdata), dtype=[('testdata', 'float')])
        nbins = 10       
        metric = metrics.TableFractionMetric('testdata', nbins=nbins)
        table = metric.run(metricdata)
        self.assertEqual(len(table), 13)
        

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
        nside=128
        metric = metrics.fONv(col='ack', nside=nside, Nvisit=825, Asky=18000.)
        npix = hp.nside2npix(nside)
        names=['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names,types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        slicePoint = {'sid':0}
        result1 = metric.run(data, slicePoint)
        deginsph = 129600./np.pi
        np.testing.assert_almost_equal(result1*18000., deginsph)
        data['blah'][:data.size/2]=0
        result2 =  metric.run(data, slicePoint)
        np.testing.assert_almost_equal(result2*18000., deginsph/2.)

    def testfOArea(self):
        """Test fOArea metric """
        nside=128
        metric = metrics.fOArea(col='ack',nside=nside, Nvisit=825, Asky=18000.)
        npix = hp.nside2npix(nside)
        names=['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names,types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        slicePoint = {'sid':0}
        result1 = metric.run(data, slicePoint)        
        np.testing.assert_almost_equal(result1*825, 826)
        data['blah'][:data.size/2]=0
        result2 = metric.run(data, slicePoint)
        
        
if __name__ == '__main__':
    unittest.main()

        
