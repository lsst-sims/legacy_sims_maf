import numpy as np
import matplotlib.pyplot as plt
import unittest
from lsst.sims.operations.maf.binners.uniBinner import UniBinner
from lsst.sims.operations.maf.binners.oneDBinner import OneDBinner

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
    

class TestUniBinnerSetup(unittest.TestCase):    
    def setUp(self):
        self.testbinner = UniBinner()
        
    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testBinnertype(self):
        """Test instantiation of binner sets binner type as expected."""        
        self.assertEqual(self.testbinner.binnertype, 'UNI')

    def testBinnerNbins(self):
        self.assertEqual(self.testbinner.nbins, 1)
        
    def testSetupBinnerIndices(self):
        """Test binner returns correct indices (all) after setup. Note this also tests slicing."""
        dvmin = 0
        dvmax = 1        
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        self.testbinner.setupBinner(dv)
        self.assertEqual(len(self.testbinner.indices), len(dv['testdata']))
        np.testing.assert_equal(dv[self.testbinner.indices], dv)


class TestUniBinnerIteration(unittest.TestCase):
    def setUp(self):
        self.testbinner = UniBinner()

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testIteration(self):
        """Test iteration -- which is a one-step identity op for a unibinner."""
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        self.testbinner.setupBinner(dv)
        for i, b in enumerate(self.testbinner):
            pass
        self.assertEqual(i, 0)

class TestUniBinnerEqual(unittest.TestCase):
    def setUp(self):
        self.testbinner = UniBinner()
        dvmin = 0
        dvmax = 1
        nvalues = 1000
        dv = makeDataValues(nvalues, dvmin, dvmax, random=True)
        self.testbinner.setupBinner(dv)    

    def tearDown(self):
        del self.testbinner
        self.testbinner = None

    def testEquivalence(self):
        """Test equals method."""
        # Note that two uni binners will be considered equal if they are both the same kind of
        # binner (unibinner). They will not necessarily slice data equally though (the indices are
        #  not necessarily the same!).
        # These should be the same, even though data is not the same.
        testbinner2 = UniBinner()
        dv2 = makeDataValues(100, 0, 1, random=True)
        testbinner2.setupBinner(dv2)
        self.assertEqual(self.testbinner, testbinner2)
        # these will not be the same, as different binner type.
        testbinner2 = OneDBinner(sliceDataColName='testdata')
        testbinner2.setupBinner(dv2, nbins=10)
        self.assertNotEqual(self.testbinner, testbinner2)
            
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUniBinnerSetup)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUniBinnerIteration)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUniBinnerEqual)
    unittest.TextTestRunner(verbosity=2).run(suite)

    # slicing tested as part of setup here, and 'function' is identity function
    #  so equivalent to slicing. 
