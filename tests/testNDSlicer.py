import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import unittest
from lsst.sims.maf.slicers.nDSlicer import NDSlicer
from lsst.sims.maf.slicers.uniSlicer import UniSlicer

def makeDataValues(size=100, min=0., max=1., nd=3, random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, in nd dimensions, but (optional) random order."""
    data = []
    for d in range(nd):
        datavalues = np.arange(0, size, dtype='float')
        datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min()) 
        datavalues += min
        if random:
            randorder = np.random.rand(size)        
            randind = np.argsort(randorder)
            datavalues = datavalues[randind]
        datavalues = np.array(zip(datavalues), dtype=[('testdata'+ '%d' %(d), 'float')])
        data.append(datavalues)
    data = rfn.merge_arrays(data, flatten=True, usemask=False)
    return data


class TestNDSlicerSetup(unittest.TestCase):    
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)
        
    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testSlicertype(self):
        """Test instantiation of slicer sets slicer type as expected."""        
        self.assertEqual(self.testslicer.slicerName, self.testslicer.__class__.__name__)
        self.assertEqual(self.testslicer.slicerName, 'NDSlicer')

    def testSetupSlicerBins(self):
        """Test setting up slicer using defined bins."""
        # Used right bins?
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        binlist = []
        for d in range(self.nd):
            binlist.append(bins)
        self.testslicer.setupSlicer(self.dv, binsList=binlist)
        for d in range(self.nd):
            np.testing.assert_equal(self.testslicer.bins[d], bins)
        self.assertEqual(self.testslicer.nbins, (len(bins)-1)**self.nd)
        
    def testSetupSlicerNbins(self):
        """Test setting up slicer using nbins."""
        for nvalues in (100, 1000, 10000):
            for nbins in (5, 10, 25, 75):
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=False)
                # Right number of bins? 
                # expect one more 'bin' to accomodate last right edge, but nbins accounts for this
                self.testslicer.setupSlicer(dv, nbinsList=nbins)
                self.assertEqual(self.testslicer.nbins, nbins**self.nd)
                # Bins of the right size?
                for i in range(self.nd):
                    bindiff = np.diff(self.testslicer.bins[i])
                    expectedbindiff = (self.dvmax - self.dvmin) / float(nbins)
                    np.testing.assert_allclose(bindiff, expectedbindiff)
                # Can we use a list of nbins too and get the right number of bins?
                nbinsList = []
                expectednbins = 1
                for d in range(self.nd):
                    nbinsList.append(nbins + d)
                    expectednbins *= (nbins + d)
                self.testslicer.setupSlicer(dv, nbinsList=nbinsList)
                self.assertEqual(self.testslicer.nbins, expectednbins)

    def testSetupSlicerEquivalent(self):
        """Test setting up slicer using defined bins and nbins is equal where expected."""
        dvmin = 0
        dvmax = 1
        for nbins in (20, 50, 100, 105):
            bins = makeDataValues(nbins+1, self.dvmin, self.dvmax, self.nd, random=False)
            binsList = []
            for i in bins.dtype.names:
                binsList.append(bins[i])
            for nvalues in (100, 1000, 10000):
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
                self.testslicer.setupSlicer(dv, nbinsList=nbins)
                for i in range(self.nd):
                    np.testing.assert_allclose(self.testslicer.bins[i], binsList[i])

                    
class TestNDSlicerEqual(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)
        self.testslicer.setupSlicer(self.dv, nbinsList=100)
        
    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testEquivalence(self):
        """Test equals method."""
        # Note that two ND slicers will be considered equal if they are both the same kind of
        # slicer AND have the same bins in all dimensions.
        # Set up another slicer to match (same bins, although not the same data).
        dv2 = makeDataValues(100, self.dvmin, self.dvmax, self.nd, random=True)
        dvlist = dv2.dtype.names
        testslicer2 = NDSlicer(sliceDataColList=dvlist)
        testslicer2.setupSlicer(dv2, binsList = self.testslicer.bins)
        self.assertEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different bins)
        dv2 = makeDataValues(1000, self.dvmin+1, self.dvmax+1, self.nd, random=True)
        testslicer2 = NDSlicer(sliceDataColList=dvlist)
        testslicer2.setupSlicer(dv2, nbinsList = 100)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up another slicer that should not match (different dimensions)
        dv2 = makeDataValues(1000, self.dvmin, self.dvmax, self.nd-1, random=True)
        testslicer2 = NDSlicer(dv2.dtype.names)
        testslicer2.setupSlicer(dv2, nbinsList=100)
        self.assertNotEqual(self.testslicer, testslicer2)
        # Set up a different kind of slicer that should not match.
        testslicer2 = UniSlicer()
        dv2 = makeDataValues(100, 0, 1, random=True)
        testslicer2.setupSlicer(dv2)
        self.assertNotEqual(self.testslicer, testslicer2)

                    

class TestNDSlicerIteration(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)
        nvalues = 1000
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        binsList = []
        self.iterlist = []
        for i in range(self.nd):
            binsList.append(bins)
            # (remember iteration doesn't use the very last bin in 'bins')
            self.iterlist.append(bins[:-1])  
        dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.testslicer.setupSlicer(dv, binsList=binsList)
        
    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testIteration(self):
        """Test iteration."""
        for b, ib in zip(self.testslicer, itertools.product(*self.iterlist)):
            self.assertEqual(b, ib)

    def testGetItem(self):
        """Test getting indexed binpoint."""
        for i, b in enumerate(self.testslicer):
            self.assertEqual(self.testslicer[i], b)
        self.assertEqual(self.testslicer[0], (0.0, 0.0, 0.0))

class TestNDSlicerSlicing(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None
    
    def testSlicing(self):
        """Test slicing."""
        # Test get error if try to slice before setup.
        self.assertRaises(NotImplementedError, self.testslicer.sliceSimData, 0)
        nbins = 10
        binsize = (self.dvmax - self.dvmin) / (float(nbins))
        for nvalues in (1000, 10000, 100000):
            dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
            self.testslicer.setupSlicer(dv, nbinsList=nbins)
            sum = 0
            for i, b in enumerate(self.testslicer):
                idxs = self.testslicer.sliceSimData(b)
                dataslice = dv[idxs]
                sum += len(idxs)
                if len(dataslice)>0:
                    for i, dvname in zip(range(self.nd), self.dvlist):
                        self.assertGreaterEqual((dataslice[dvname].min() - b[i]), 0)
                    if i < self.testslicer.nbins-1:
                        self.assertLessEqual((dataslice[dvname].max() - b[i]), binsize)
                    else:
                        self.assertAlmostEqual((dataslice[dvname].max() - b[i]), binsize)
                    self.assertTrue(len(dataslice), nvalues/float(nbins))
            # and check that every data value was assigned somewhere.
            self.assertEqual(sum, nvalues)
                    
class TestNDSlicerHistogram(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 1
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testslicer = NDSlicer(self.dvlist)

    def tearDown(self):
        del self.testslicer
        self.testslicer = None

    def testHistogram(self):
        """Test that histogram values match those generated by numpy hist, when using 1d."""
        for nbins in [10, 20, 30, 75, 100, 33]:
            for nvalues in [1000, 10000, 250000]:
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
                self.testslicer.setupSlicer(dv, nbinsList=nbins)
                metricval = np.zeros(len(self.testslicer), 'float')
                for i, b in enumerate(self.testslicer):
                    idxs = self.testslicer.sliceSimData(b)
                    metricval[i] = len(idxs)
                numpycounts, numpybins = np.histogram(dv['testdata0'], bins=nbins)
                np.testing.assert_equal(numpybins, self.testslicer.bins[0], 'Numpy bins do not match testslicer bins')
                np.testing.assert_equal(numpycounts, metricval, 'Numpy histogram counts do not match testslicer counts')


class TestNDSlicerPlotting(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1

    def tearDown(self):
        del self.testslicer
        self.testslicer = None
        
    def testPlotting(self):
        """Test plotting."""
        nvalues = 100
        nd = 3
        dv = makeDataValues(nvalues, self.dvmin, self.dvmax, nd, random=True)
        dvlist = dv.dtype.names
        condition = (dv[dvlist[0]] < .5)
        dv[dvlist[0]][condition] = 0.25
        dv[dvlist[2]][condition] = 0.5
        bins = np.arange(0, 1+0.01, 0.01)        
        binsList = []
        for d in range(nd):
            binsList.append(bins)
        print ''
        self.testslicer = NDSlicer(dvlist)
        self.testslicer.setupSlicer(dv, binsList=binsList)
        metricval = np.zeros(len(self.testslicer), 'float')
        for i, b in enumerate(self.testslicer):
            idxs = self.testslicer.sliceSimData(b)
            metricval[i] = len(idxs)
        self.testslicer.plotBinnedData1D(metricval, axis=0, xlabel='xrange', ylabel='count')
        self.testslicer.plotBinnedData1D(metricval, axis=0, xlabel='xrange', ylabel='count',
                                         filled=True, ylog=True)
        self.testslicer.plotBinnedData1D(metricval, axis=0, xlabel='xrange', ylabel='count',
                                         filled=True, title='axis with hump')
        self.testslicer.plotBinnedData1D(metricval, axis=1, xlabel='xrange', ylabel='count',
                                         filled=True, title='axis with flat distro')
        self.testslicer.plotBinnedData1D(metricval, axis=2, xlabel='xrange', ylabel='count',
                                        filled=True, title='axis with hump based on axis 0')
        self.testslicer.plotBinnedData2D(metricval, xaxis=0, yaxis=1)
        self.testslicer.plotBinnedData2D(metricval, xaxis=0, yaxis=2)
        self.testslicer.plotBinnedData2D(metricval, xaxis=1, yaxis=2)
        plt.show()
        

if __name__ == "__main__":
    unittest.main()

