import matplotlib
matplotlib.use("Agg")
import unittest
import lsst.sims.maf.utils.opsimUtils as opsimUtils
import lsst.utils.tests


class TestOpsimUtils(unittest.TestCase):

    def testScaleBenchmarks(self):
        """Test scaling the design and stretch benchmarks for the length of the run."""
        # First test that method returns expected dictionaries.
        for i in ('design', 'stretch'):
            benchmark = opsimUtils.scaleBenchmarks(10.0, i)
            self.assertTrue(isinstance(benchmark, dict))
            expectedkeys = ('Area', 'nvisitsTotal', 'nvisits', 'seeing', 'skybrightness',
                            'singleVisitDepth')
            expectedfilters = ('u', 'g', 'r', 'i', 'z', 'y')
            for k in expectedkeys:
                self.assertTrue(k in benchmark)
            expecteddictkeys = ('nvisits', 'seeing', 'skybrightness', 'singleVisitDepth')
            for k in expecteddictkeys:
                for f in expectedfilters:
                    self.assertTrue(f in benchmark[k])

    def testCalcCoaddedDepth(self):
        """Test the expected coadded depth calculation."""
        benchmark = opsimUtils.scaleBenchmarks(10, 'design')
        coadd = opsimUtils.calcCoaddedDepth(benchmark['nvisits'], benchmark['singleVisitDepth'])
        for f in coadd:
            self.assertTrue(coadd[f] < 1000)
        singlevisits = {'u': 1, 'g': 1, 'r': 1, 'i': 1, 'z': 1, 'y': 1}
        coadd = opsimUtils.calcCoaddedDepth(singlevisits, benchmark['singleVisitDepth'])
        for f in coadd:
            self.assertAlmostEqual(coadd[f], benchmark['singleVisitDepth'][f])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
