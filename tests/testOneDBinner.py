import numpy as np
import matplotlib.pyplot as plt
import unittest
from lsst.sims.operations.maf.binners.oneDBinner import OneDBinner


class TestOneDBinner(unittest.TestCase):
    
    def dataValues(self, size=100, min=0., max=1., random=True):
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
    
    def testBinnertype(self):
        """Test instantiation of binner sets binner type as expected."""
        testbinner = OneDBinner()
        self.assertEqual(testbinner.binnertype, 'ONED')

    def testSetupBinnerNbins(self):
        """Test setting up binner using nbins."""
        testbinner = OneDBinner()
        for nvalues in (100, 1000, 10000):
            for nbins in (5, 10, 25, 75):
                dvmin = 0
                dvmax = 1
                dv = self.dataValues(nvalues, dvmin, dvmax, random=False)
                # Right number of bins?
                testbinner.setupBinner(dv, 'testdata', nbins=nbins)
                self.assertEqual(testbinner.nbins, nbins)
                # Bins of the right size?
                bindiff = np.diff(testbinner.bins)
                expectedbindiff = (dvmax - dvmin) / float(nbins)
                np.testing.assert_allclose(bindiff, expectedbindiff)
            
    def testSetupBinnerBins(self):
        """Test setting up binner using defined bins."""
        testbinner = OneDBinner()
        dvmin = 0
        dvmax = 1        
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)
        dv = self.dataValues(nvalues, dvmin, dvmax, random=True)
        # Right bins?
        testbinner.setupBinner(dv, 'testdata', bins=bins)
        np.testing.assert_equal(testbinner.bins, bins)

    def testSetupBinnerEquivalent(self):
        """Test setting up binner using defined bins and nbins is equal where expected."""
        testbinner = OneDBinner()
        dvmin = 0
        dvmax = 1
        nbins = 100
        bins = self.dataValues(nbins+1, dvmin, dvmax, random=False)
        bins = bins['testdata'][:-1]
        for nvalues in (100, 1000, 10000):
            dv = self.dataValues(nvalues, dvmin, dvmax, random=True)
            testbinner.setupBinner(dv, 'testdata', nbins=nbins)
            np.testing.assert_allclose(testbinner.bins, bins)

    def testIteration(self):
        testbinner = OneDBinner()
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        bins = np.arange(dvmin, dvmax, 0.1)        
        dv = self.dataValues(nvalues, dvmin, dvmax, random=True)
        testbinner.setupBinner(dv, 'testdata', bins=bins)
        for b, ib in zip(testbinner, bins):
            self.assertEqual(b, ib)
            
    def testSlicing(self):
        testbinner = OneDBinner()
        dvmin = 0
        dvmax = 1
        nbins = 100
        binsize = (dvmax - dvmin) / (float(nbins))
        for nvalues in (1000, 10000, 100000):
            dv = self.dataValues(nvalues, dvmin, dvmax, random=True)
            testbinner.setupBinner(dv, 'testdata', nbins=101)
            for b in testbinner:
                idxs = testbinner.sliceSimData(b)
                dataslice = dv['testdata'][idxs]
                self.assertTrue((dataslice.min() - b) >= 0)
                self.assertTrue((dataslice.max() - b) <= binsize)
                self.assertTrue(len(dataslice), nvalues/float(nbins))

    def testPlotting(self):
        """Test plotting."""
        testbinner = OneDBinner()
        dvmin = 0 
        dvmax = 1
        nbins = 100
        nvalues = 10000
        dv = self.dataValues(nvalues, dvmin, dvmax, random=True)
        testbinner.setupBinner(dv, 'testdata', nbins=101)
        metricval = np.zeros(len(testbinner), 'float')
        for i, b in enumerate(testbinner):
            idxs = testbinner.sliceSimData(b)
            metricval[i] = len(idxs)
        testbinner.plotBinnedData(metricval, xlabel='xrange', ylabel='count')
        plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)  
