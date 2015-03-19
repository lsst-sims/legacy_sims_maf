import matplotlib
matplotlib.use("Agg")
import unittest
import lsst.sims.maf.utils.opsimUtils as opsimUtils

class TestOpsimUtils(unittest.TestCase):

    def testScale(self):
        """Test scaling the design and stretch benchmarks for the length of the run."""
        # First test that method returns expected dictionaries.
        design = opsimUtils.scaleBenchmarks(10.0, 'design')
        self.assertTrue(isinstance(design, dict))
        expectedkeys = ('Area', 'nvisitsTotal', 'nvisits', 'seeing', 'skybrightness',
                        'singleVisitDepth')
        expectedfilters = ('u', 'g', 'r', 'i', 'z', 'y')
        for k in expectedkeys:
            self.assertTrue(k in design)
        expecteddictkeys = ('nvisits', 'seeing', 'skybrightness', 'singleVisitDepth')
        for k in expecteddictkeys:
            for f in expectedfilters:
                self.assertTrue(f in design[k])

if __name__ == "__main__":
    unittest.main()
