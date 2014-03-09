import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import itertools
import unittest
from lsst.sims.operations.maf.binners.nDBinner import NDBinner

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
    

class TestNDBinnerSetup(unittest.TestCase):    
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testbinner = NDBinner(self.dvlist)
        
    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testBinnertype(self):
        """Test instantiation of binner sets binner type as expected."""        
        self.assertEqual(self.testbinner.binnertype, 'ND')

    def testSetupBinnerBins(self):
        """Test setting up binner using defined bins."""
        # Used right bins?
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        binlist = []
        for d in range(self.nd):
            binlist.append(bins)
        self.testbinner.setupBinner(self.dv, binsList=binlist)
        for d in range(self.nd):
            np.testing.assert_equal(self.testbinner.bins[d], bins)
        self.assertEqual(self.testbinner.nbins, (len(bins)-1)**self.nd)
        
    def testSetupBinnerNbins(self):
        """Test setting up binner using nbins."""
        for nvalues in (100, 1000, 10000):
            for nbins in (5, 10, 25, 75):
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=False)
                # Right number of bins? 
                # expect one more 'bin' to accomodate last right edge), but nbins accounts for this
                self.testbinner.setupBinner(dv, nbinsList=nbins)
                self.assertEqual(self.testbinner.nbins, nbins**self.nd)
                # Bins of the right size?
                for i in range(self.nd):
                    bindiff = np.diff(self.testbinner.bins[i])
                    expectedbindiff = (self.dvmax - self.dvmin) / float(nbins)
                    np.testing.assert_allclose(bindiff, expectedbindiff)
            

    def testSetupBinnerEquivalent(self):
        """Test setting up binner using defined bins and nbins is equal where expected."""
        dvmin = 0
        dvmax = 1
        for nbins in (20, 50, 100, 105):
            bins = makeDataValues(nbins+1, self.dvmin, self.dvmax, self.nd, random=False)
            binsList = []
            for i in bins.dtype.names:
                binsList.append(bins[i])
            for nvalues in (100, 1000, 10000):
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
                self.testbinner.setupBinner(dv, nbinsList=nbins)
                for i in range(self.nd):
                    np.testing.assert_allclose(self.testbinner.bins[i], binsList[i])


class TestNDBinnerIteration(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testbinner = NDBinner(self.dvlist)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testIteration(self):
        nvalues = 1000
        bins = np.arange(self.dvmin, self.dvmax, 0.1)
        binsList = []
        iterlist = []
        for i in range(self.nd):
            binsList.append(bins)
            # (remember iteration doesn't use the very last bin in 'bins')
            iterlist.append(bins[:-1])  
        dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.testbinner.setupBinner(dv, binsList=binsList)
        for b, ib in zip(self.testbinner, itertools.product(*iterlist)):
            self.assertEqual(b, ib)


class TestNDBinnerSlicing(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 3
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testbinner = NDBinner(self.dvlist)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None
    
    def testSlicing(self):
        print ''
        nbins = 10
        binsize = (self.dvmax - self.dvmin) / (float(nbins))
        for nvalues in (1000, 10000, 100000):
            dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
            self.testbinner.setupBinner(dv, nbinsList=nbins)
            sum = 0
            for i, b in enumerate(self.testbinner):
                idxs = self.testbinner.sliceSimData(b)
                dataslice = dv[idxs]
                sum += len(idxs)
                if len(dataslice)>0:
                    for i, dvname in zip(range(self.nd), self.dvlist):
                        self.assertGreaterEqual((dataslice[dvname].min() - b[i]), 0)
                    if i < self.testbinner.nbins-1:
                        self.assertLessEqual((dataslice[dvname].max() - b[i]), binsize)
                    else:
                        self.assertAlmostEqual((dataslice[dvname].max() - b[i]), binsize)
                    self.assertTrue(len(dataslice), nvalues/float(nbins))
            # and check that every data value was assigned somewhere.
            self.assertEqual(sum, nvalues)
                    
class TestNDBinnerFunction(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 1
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testbinner = NDBinner(self.dvlist)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testHistogram(self):
        """Test that histogram values match those generated by numpy hist, when using 1d."""
        for nbins in [10, 20, 30, 75, 100, 33]:
            for nvalues in [1000, 10000, 250000]:
                dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
                self.testbinner.setupBinner(dv, nbinsList=nbins)
                metricval = np.zeros(len(self.testbinner), 'float')
                for i, b in enumerate(self.testbinner):
                    idxs = self.testbinner.sliceSimData(b)
                    metricval[i] = len(idxs)
                numpycounts, numpybins = np.histogram(dv['testdata0'], bins=nbins)
                np.testing.assert_equal(numpybins, self.testbinner.bins[0], 'Numpy bins do not match testbinner bins')
                np.testing.assert_equal(numpycounts, metricval, 'Numpy histogram counts do not match testbinner counts')


class TestNDBinnerPlotting(unittest.TestCase):
    def setUp(self):
        self.dvmin = 0
        self.dvmax = 1
        nvalues = 1000
        self.nd = 2
        self.dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        self.dvlist = self.dv.dtype.names
        self.testbinner = NDBinner(self.dvlist)

    def tearDown(self):
        del self.testbinner
        self.testbinner = None
        
    def testPlotting(self):
        """Test plotting."""
        nbins = 100
        nvalues = 10000
        dv = makeDataValues(nvalues, self.dvmin, self.dvmax, self.nd, random=True)
        testbinner.setupBinner(dv, nbinsLits=nbins)
        metricval = np.zeros(len(testbinner), 'float')
        for i, b in enumerate(testbinner):
            idxs = testbinner.sliceSimData(b)
            metricval[i] = len(idxs)
        testbinner.plotBinnedData(metricval, xlabel='xrange', ylabel='count')
        plt.show()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNDBinnerSetup)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNDBinnerIteration)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNDBinnerSlicing)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNDBinnerFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestNDBinnerPlotting)
    #unittest.TextTestRunner(verbosity=2).run(suite)
