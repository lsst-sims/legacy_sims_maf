import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.driver import runBundle
import unittest, os
import shutil
import glob

class TestRunBundle(unittest.TestCase):

    def testInput(self):
        """
        Check that a value error gets thrown
        """
        badDict = {}
        self.assertRaises(ValueError, runBundle, mafBundle=badDict)

    def testRun(self):
        """
        Check that expected output gets made
        """
        filepath = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/')
        dbAddress ='sqlite:///' + filepath + 'opsimblitz1_1133_sqlite.db'
        sqlWhere = 'filter = "r" and night < 200'
        slicer = slicers.HealpixSlicer(nside=64)
        metricList = []
        metricList.append(metrics.MeanMetric(col='airmass',
                                             summaryStatList=[metrics.MeanMetric(), metrics.RmsMetric()]))
        outDir = 'BundleExampleOut'
        mafBundle = {'slicer':slicer, 'metricList':metricList, 'dbAddress':dbAddress,
                     'sqlWhere':sqlWhere, 'outDir':outDir}

        metricResults = runBundle(mafBundle, verbose=False)
        outpdfs = glob.glob(outDir+'/*.pdf')
        outcfg = glob.glob(outDir+'/config*')
        outpng = glob.glob(outDir+'/*.png')
        outdb = glob.glob(outDir+'/*.db')

        assert(len(outpdfs) == 3 )
        assert(len(outcfg) == 2 )
        assert(len(outpng) == 3)
        assert(outdb[0] == outDir+'/resultsDb_sqlite.db')

        statsRun = ['Mean', 'Rms']
        for key in metricResults.summaryStats.keys():
            for key2 in metricResults.summaryStats[key].keys():
                assert(key2 in statsRun)

    def teardown(self):
        if os.path.isdir('BundleExampleOut'):
            shutil.rmtree('BundleExampleOut')



if __name__ == "__main__":
    unittest.main()
