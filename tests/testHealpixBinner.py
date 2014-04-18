import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.ma as ma
import matplotlib.pyplot as plt
import unittest
import healpy as hp
from lsst.sims.maf.binners.healpixBinner import HealpixBinner
from lsst.sims.maf.binners.uniBinner import UniBinner


def makeDataValues(size=100, minval=0., maxval=1., ramin=0, ramax=2*np.pi,
                   decmin=-np.pi, decmax=np.pi, random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, in 1 dimensions (optionally sorted), together with RA/Dec values for each data value."""
    data = []
    # Generate data values min - max.
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(maxval) - float(minval)) / (datavalues.max() - datavalues.min()) 
    datavalues += minval
    if random:
        randorder = np.random.rand(size)        
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(zip(datavalues), dtype=[('testdata', 'float')])
    data.append(datavalues)
    # Generate RA/Dec values equally spaces on sphere between ramin/max, decmin/max.
    ra = np.arange(0, size, dtype='float')
    ra *= (float(ramax) - float(ramin)) / (ra.max() - ra.min())
    if random:
        randorder = np.random.rand(size)        
        randind = np.argsort(randorder)
        ra = ra[randind]
    ra = np.array(zip(ra), dtype=[('ra', 'float')])
    data.append(ra)
    v = np.arange(0, size, dtype='float')
    v *= ((np.cos(decmax+np.pi) + 1.)/2.0 - (np.cos(decmin+np.pi)+1.)/2.0) / (v.max() - v.min())
    v += (np.cos(decmin+np.pi)+1.)/2.0
    dec = np.arccos(2*v-1) - np.pi
    if random:
        randorder = np.random.rand(size)        
        randind = np.argsort(randorder)
        dec = dec[randind]
    dec = np.array(zip(dec), dtype=[('dec', 'float')])
    data.append(dec)
    data = rfn.merge_arrays(data, flatten=True, usemask=False)
    return data

def calcDist_vincenty(RA1, Dec1, RA2, Dec2):
    """Calculates distance on a sphere using the Vincenty formula. 
    Give this function RA/Dec values in radians. Returns angular distance(s), in radians.
    Note that since this is all numpy, you could input arrays of RA/Decs."""
    D1 = (np.cos(Dec2)*np.sin(RA2-RA1))**2 + \
        (np.cos(Dec1)*np.sin(Dec2) - \
        np.sin(Dec1)*np.cos(Dec2)*np.cos(RA2-RA1))**2
    D1 = np.sqrt(D1)
    D2 = (np.sin(Dec1)*np.sin(Dec2) + \
        np.cos(Dec1)*np.cos(Dec2)*np.cos(RA2-RA1))
    D = np.arctan2(D1,D2)
    return D

class TestHealpixBinnerSetup(unittest.TestCase):    
    def testBinnertype(self):
        """Test instantiation of binner sets binner type as expected."""
        testbinner = HealpixBinner(nside=16, verbose=False)
        self.assertEqual(testbinner.binnerName, testbinner.__class__.__name__)
        self.assertEqual(testbinner.binnerName, 'HealpixBinner')

    def testNsidesNbins(self):
        """Test that number of sides passed to binner produces expected number of bins."""
        nsides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        npixx = [12, 48, 192, 768, 3072, 12288, 49152, 196608, 786432, 3145728]
        for nside, npix in zip(nsides, npixx):
            testbinner = HealpixBinner(nside=nside, verbose=False)
            self.assertEqual(testbinner.nbins, npix)

    def testNsidesError(self):
        """Test that if passed an incorrect value for nsides that get expected exception."""
        self.assertRaises(ValueError, HealpixBinner, nside=3)

class TestHealpixBinnerEqual(unittest.TestCase):
    def setUp(self):
        self.nside = 16
        self.testbinner = HealpixBinner(nside=self.nside, verbose=False)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testBinnerEquivalence(self):
        """Test that binners are marked equal when appropriate, and unequal when appropriate."""
        # Note that they are judged equal based on nsides (not on data in ra/dec spatial tree).
        testbinner2 = HealpixBinner(nside=self.nside, verbose=False)
        self.assertEqual(self.testbinner, testbinner2)
        testbinner2 = HealpixBinner(nside=self.nside/2.0, verbose=False)
        self.assertNotEqual(self.testbinner, testbinner2)
        
class TestHealpixBinnerIteration(unittest.TestCase):
    def setUp(self):
        self.nside = 16
        self.testbinner = HealpixBinner(nside=self.nside, verbose=False)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testIteration(self):
        """Test iteration goes through expected range and ra/dec are in expected range (radians)."""
        npix = hp.nside2npix(self.nside)
        for i, b in enumerate(self.testbinner):
            self.assertEqual(i, b[0])
            ra = b[1]
            dec = b[2]
            self.assertGreaterEqual(ra, 0)
            self.assertLessEqual(ra, 2*np.pi)
            self.assertGreaterEqual(dec, -np.pi)
            self.assertLessEqual(dec, np.pi)
        # npix would count starting at 1, while i counts starting at 0 ..
        #  so add one to check end point
        self.assertEqual(i+1, npix)

    def testGetItem(self):
        """Test getting indexed value."""
        for i, b in enumerate(self.testbinner):
            np.testing.assert_equal(self.testbinner[i], b)

class TestHealpixBinnerSlicing(unittest.TestCase):
    # Note that this is really testing baseSpatialBinner, as slicing is done there for healpix grid
    def setUp(self):
        self.nside = 16
        self.testbinner = HealpixBinner(nside=self.nside, verbose=False,
                                        spatialkey1='ra', spatialkey2='dec')
        nvalues = 10000
        self.dv = makeDataValues(size=nvalues, minval=0., maxval=1.,
                                ramin=0, ramax=2*np.pi,
                                decmin=-np.pi, decmax=0,
                                random=True)
        self.radius = 1.8


    def tearDown(self):
        del self.testbinner
        self.testbinner = None
    
    def testSlicing(self):
        """Test slicing returns (all) data points which are within 'radius' of bin point."""
        # Test that slicing fails before setupBinner
        self.assertRaises(NotImplementedError, self.testbinner.sliceSimData, 0)
        self.testbinner.setupBinner(self.dv, radius=self.radius)
        for b in self.testbinner:
            binra = b[1]
            bindec = b[2]
            distances = calcDist_vincenty(binra, bindec, self.dv['ra'], self.dv['dec'])
            didxs = np.where(distances<=np.radians(self.radius))
            binidxs = self.testbinner.sliceSimData(b)
            self.assertEqual(len(binidxs), len(didxs[0]))
            if len(binidxs) > 0:
                didxs = np.sort(didxs[0])
                binidxs = np.sort(binidxs)
                np.testing.assert_equal(self.dv['testdata'][didxs], self.dv['testdata'][binidxs])

class TestHealpixBinnerPlotting(unittest.TestCase):
    def setUp(self):
        self.nside = 16
        self.testbinner = HealpixBinner(nside=self.nside, verbose=False,
                                        spatialkey1='ra', spatialkey2='dec')
        nvalues = 10000
        self.dv = makeDataValues(size=nvalues, minval=0., maxval=1.,
                                ramin=0, ramax=2*np.pi,
                                decmin=-np.pi, decmax=0,
                                random=True)
        self.radius = 1.8
        self.testbinner.setupBinner(self.dv, radius=self.radius)
        self.metricdata = ma.MaskedArray(data = np.zeros(len(self.testbinner), dtype='float'),
                                         mask = np.zeros(len(self.testbinner), 'bool'),
                                         fill_value = self.testbinner.badval)
        for i, b in enumerate(self.testbinner):
            idxs = self.testbinner.sliceSimData(b)
            if len(idxs) > 0:
                self.metricdata.data[i] = np.mean(self.dv['testdata'][idxs])
            else:
                self.metricdata.mask[i] = True
        self.metricdata2 = ma.MaskedArray(data = np.random.rand(len(self.testbinner)),
                                          mask = np.zeros(len(self.testbinner), 'bool'),
                                          fill_value = self.testbinner.badval)


    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testSkyMap(self):
        """Test plotting the sky map (mean of random data)"""
        self.testbinner.plotSkyMap(self.metricdata, units=None, title='Mean of random test data',
                        clims=None, ylog=False, cbarFormat='%.2g')
        self.testbinner.plotSkyMap(self.metricdata2, units=None, title='Random Test Data',
                        clims=None, ylog=False, cbarFormat='%.2g')
    
    def testPowerSpectrum(self):
        """Test plotting the power spectrum (mean of random data)."""
        self.testbinner.plotPowerSpectrum(self.metricdata, title='Mean of random test data',
                                          fignum=None, maxl=500.,
                                          legendLabel=None, addLegend=False, removeDipole=True,
                                          verbose=False)
        self.testbinner.plotPowerSpectrum(self.metricdata2, title='Random test data',
                                          fignum=None, maxl=500.,
                                          legendLabel=None, addLegend=False, removeDipole=True,
                                          verbose=False)
        
    def testHistogram(self):
        """Test plotting the histogram (mean of random data)."""
        self.testbinner.plotHistogram(self.metricdata, title='Mean of random test data', xlabel=None,
                                      ylabel='Area (1000s of square degrees)',
                                      fignum=None, legendLabel=None, addLegend=False,
                                      legendloc='upper left',
                                      bins=100, cumulative=False, histRange=None,
                                      ylog=False, flipXaxis=False, scale=None)
        plt.figure()
        plt.hist(self.metricdata.compressed(), bins=100)
        plt.title('Histogram straight from metric data')
        self.testbinner.plotHistogram(self.metricdata2, title='Random test data', xlabel=None,
                                      ylabel='Area (1000s of square degrees)',
                                      fignum=None, legendLabel=None, addLegend=False,
                                      legendloc='upper left',
                                      bins=100, cumulative=False, histRange=None,
                                      ylog=False, flipXaxis=False, scale=None)

                
                        
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealpixBinnerSetup)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealpixBinnerEqual)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealpixBinnerIteration)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealpixBinnerSlicing)
    unittest.TextTestRunner(verbosity=2).run(suite)

    #suite = unittest.TestLoader().loadTestsFromTestCase(TestHealpixBinnerPlotting)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #plt.show()
