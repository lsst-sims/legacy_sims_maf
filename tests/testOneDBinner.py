import numpy as np
import matplotlib.pyplot as plt
import unittest
from lsst.sims.operations.maf.binners.oneDBinner import OneDBinner
from lsst.sims.operations.maf.binners.uniBinner import UniBinner

def makeDataValues(size=100, min=0., max=1., random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""    
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min()) 
    datavalues += min
    if random:
        randorder = np.random.rand(size)        
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(zip(datavalues), dtype=[('testdata', 'float')])
    return datavalues
    

class TestOneDBinnerSetup(unittest.TestCase):    
    def setUp(self):
        self.testbinner = OneDBinner(sliceDataColName='testdata')
        
    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testBinnertype(self):
        """Test instantiation of binner sets binner type as expected."""        
        self.assertEqual(self.testbinner.binnerName, self.testbinner.__class__.__name__)
        self.assertEqual(self.testbinner.binnerName, 'OneDBinner')        

    def testSetupBinnerBins(self):
        """Test setting up binner using defined bins."""
        dvmin = 0
        dvmax = 1        
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        # Used right bins?
        self.testbinner.setupBinner(dv, bins=bins)
        np.testing.assert_equal(self.testbinner.bins, bins)
        self.assertEqual(self.testbinner.nbins, len(bins)-1)
        
    def testSetupBinnerNbins(self):
        """Test setting up binner using nbins."""
        for nvalues in (100, 1000, 10000):
            for nbins in (5, 10, 25, 75):
                dvmin = 0
                dvmax = 1
                dv = makeDataValues(nvalues, dvmin, dvmax, random=False)
                # Right number of bins? 
                # expect one more 'bin' to accomodate last right edge), but nbins accounts for this
                self.testbinner.setupBinner(dv, nbins=nbins)
                self.assertEqual(self.testbinner.nbins, nbins)
                # Bins of the right size?
                bindiff = np.diff(self.testbinner.bins)
                expectedbindiff = (dvmax - dvmin) / float(nbins)
                np.testing.assert_allclose(bindiff, expectedbindiff)
            

    def testSetupBinnerEquivalent(self):
        """Test setting up binner using defined bins and nbins is equal where expected."""
        dvmin = 0
        dvmax = 1
        for nbins in (20, 50, 100, 105):
            bins = makeDataValues(nbins+1, dvmin, dvmax, random=False)
            bins = bins['testdata']
            for nvalues in (100, 1000, 10000):
                dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
                self.testbinner.setupBinner(dv, nbins=nbins)
                np.testing.assert_allclose(self.testbinner.bins, bins)


class TestOneDBinnerIteration(unittest.TestCase):
    def setUp(self):
        self.testbinner = OneDBinner(sliceDataColName='testdata')

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testIteration(self):
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.01)        
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        self.testbinner.setupBinner(dv, bins=bins)
        for b, ib in zip(self.testbinner, bins[:-1]):
            self.assertEqual(b, ib)

class TestOneDBinnerEqual(unittest.TestCase):
    def setUp(self):
        self.testbinner = OneDBinner(sliceDataColName='testdata')

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testEquivalence(self):
        """Test equals method."""
        # Note that two OneD binners will be considered equal if they are both the same kind of
        # binner AND have the same bins.
        # Set up self..
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.01)        
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        self.testbinner.setupBinner(dv, bins=bins)

        # Set up another binner to match (same bins, although not the same data).
        testbinner2 = OneDBinner(sliceDataColName='testdata')
        dv2 = makeDataValues(nvalues+100, dvmin, dvmax, random=True)
        testbinner2.setupBinner(dv2, bins=bins)
        self.assertEqual(self.testbinner, testbinner2)
        # Set up another binner that should not match (different bins)
        testbinner2 = OneDBinner(sliceDataColName='testdata')
        dv2 = makeDataValues(nvalues, dvmin+1, dvmax+1, random=True)
        testbinner2.setupBinner(dv2, nbins=len(bins))
        self.assertNotEqual(self.testbinner, testbinner2)
        # Set up a different kind of binner that should not match.
        testbinner2 = UniBinner()
        dv2 = makeDataValues(100, 0, 1, random=True)
        testbinner2.setupBinner(dv2)
        self.assertNotEqual(self.testbinner, testbinner2)

            
class TestOneDBinnerSlicing(unittest.TestCase):            
    def setUp(self):
        self.testbinner = OneDBinner(sliceDataColName='testdata')

    def tearDown(self):
        del self.testbinner
        self.testbinner = None
    
    def testSlicing(self):
        dvmin = 0
        dvmax = 1
        nbins = 100
        binsize = (dvmax - dvmin) / (float(nbins))
        for nvalues in (1000, 10000, 100000):
            dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
            self.testbinner.setupBinner(dv, nbins=nbins)
            sum = 0
            for i, b in enumerate(self.testbinner):
                idxs = self.testbinner.sliceSimData(b)
                dataslice = dv['testdata'][idxs]
                sum += len(idxs)
                if len(dataslice)>0:
                    self.assertGreaterEqual((dataslice.min() - b), 0)
                    if i < self.testbinner.nbins-1:
                        self.assertLessEqual((dataslice.max() - b), binsize)
                    else:
                        self.assertAlmostEqual((dataslice.max() - b), binsize)
                    self.assertTrue(len(dataslice), nvalues/float(nbins))
                else:
                    self.assertTrue(len(dataslice) > 0, 'Data in test case expected to always be > 0 len after slicing.')
            self.assertTrue(sum, nvalues)

class TestOneDBinnerHistogram(unittest.TestCase):
    def setUp(self):
        self.testbinner = OneDBinner(sliceDataColName='testdata')

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testHistogram(self):
        """Test that histogram values match those generated by numpy hist."""
        dvmin = 0 
        dvmax = 1
        for nbins in [10, 20, 30, 75, 100, 33]:
            for nvalues in [1000, 10000, 250000]:
                dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
                self.testbinner.setupBinner(dv, nbins=nbins)
                metricval = np.zeros(len(self.testbinner), 'float')
                for i, b in enumerate(self.testbinner):
                    idxs = self.testbinner.sliceSimData(b)
                    metricval[i] = len(idxs)
                numpycounts, numpybins = np.histogram(dv['testdata'], bins=nbins)
                np.testing.assert_equal(numpybins, self.testbinner.bins, 'Numpy bins do not match testbinner bins')
                np.testing.assert_equal(numpycounts, metricval, 'Numpy histogram counts do not match testbinner counts')

    @unittest.skip("Run interactively")
    def testPlotting(self):
        """Test plotting."""
        testbinner = OneDBinner(sliceDataColName='testdata')
        dvmin = 0 
        dvmax = 1
        nbins = 100
        nvalues = 10000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        testbinner.setupBinner(dv, nbins=nbins)
        metricval = np.zeros(len(testbinner), 'float')
        for i, b in enumerate(testbinner):
            idxs = testbinner.sliceSimData(b)
            metricval[i] = len(idxs)
        testbinner.plotBinnedData(metricval, xlabel='xrange', ylabel='count')
        plt.show()


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOneDBinnerSetup)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOneDBinnerIteration)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOneDBinnerEqual)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOneDBinnerSlicing)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOneDBinnerHistogram)
    unittest.TextTestRunner(verbosity=2).run(suite)
