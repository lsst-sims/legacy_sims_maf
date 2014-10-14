import matplotlib
matplotlib.use("Agg")
import unittest
import lsst.sims.maf.utils.opsimUtils as opsimUtils

class TestOpsimUtils(unittest.TestCase):

    def testScale(self):
        """Test scaling the design and stretch benchmarks for the length of the run."""
        # First test that method returns expected dictionaries.
        design, stretch = opsimUtils.scaleStretchDesign(10.0)
        self.assertTrue(isinstance(design, dict))
        self.assertTrue(isinstance(stretch, dict))
        expectedkeys = ('nvisits', 'seeing', 'skybrightness', 'singleVisitDepth', 'coaddedDepth')
        expectedfilters = ('u', 'g', 'r', 'i', 'z', 'y')
        for k in expectedkeys:
            self.assertTrue(k in design.keys())
            self.assertTrue(k in stretch.keys())
            for f in expectedfilters:
                self.assertTrue(f in design[k].keys())
                self.assertTrue(f in stretch[k].keys())
            
        for runLength in ([1, 5, 10]):
            design, stretch = opsimUtils.scaleStretchDesign(runLength)
            for f in expectedfilters:
                print runLength, design['nvisits']
            ## Need update from opsim team to check what these should be compared to.

if __name__ == "__main__":
    unittest.main()
