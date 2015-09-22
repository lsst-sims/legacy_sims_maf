import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundle

# Test the 2DSlicers and the vector metrics

class Test2D(unittest.TestCase):

    def setUp(self):
        names = ['night', 'fieldID', 'fieldRA', 'fieldDec', 'fiveSigmaDepth']
        types = [int,int,float,float,float]
        # Picking RA and Dec values that will hit nside=16 healpixels
        self.simData = np.zeros(99, dtype=zip(names,types))
        self.simData['night'][0:50] = 1
        self.simData['fieldID'][0:50] = 1
        self.simData['fieldRA'][0:50] = np.radians(10.)
        self.simData['fieldDec'][0:50] = 0
        self.simData['fiveSigmaDepth'][0:50] = 25.

        self.simData['night'][50:] = 2
        self.simData['fieldID'][50:] = 2
        self.simData['fieldRA'][50:] = np.radians(190.)
        self.simData['fieldDec'][50:] = np.radians(-20.)
        self.simData['fiveSigmaDepth'][50:] = 24.

        self.fieldData = np.zeros(2, dtype=zip(['fieldID', 'fieldRA', 'fieldDec'],[int,float,float]))
        self.fieldData['fieldID'] = [1,2]
        self.fieldData['fieldRA'] = np.radians([10.,190.])
        self.fieldData['fieldDec'] = np.radians([0.,-20.])


    def testOpsim2dSlicer(self):
        metric = metrics.AccumulateCountMetric()
        slicer = slicers.Opsim2dSlicer(bins=[0.5,1.5,2.5])
        sql = ''
        mb = metricBundle.MetricBundle(metric,slicer,sql)
        mbg = metricBundle.MetricBundleGroup({0:mb}, None)
        mbg.setCurrent('')
        mbg.fieldData = self.fieldData
        mbg.runCurrent('', simData=self.simData)
        expected = np.array( [[-666.,   50.,   50.],
                              [-666., -666.,   49.]])
        assert(np.array_equal(mb.metricValues.data, expected))

    def testHealpix2dSlicer(self):
        metric = metrics.AccumulateCountMetric()
        slicer = slicers.Healpix2dSlicer(nside=16, bins=[0.5,1.5,2.5])
        sql = ''
        mb = metricBundle.MetricBundle(metric,slicer,sql)
        mbg = metricBundle.MetricBundleGroup({0:mb}, None)
        mbg.setCurrent('')
        mbg.runCurrent('', simData=self.simData)

        good = np.where(mb.metricValues.mask[:,-1] == False)[0]
        expected =  np.array( [[-666.,   50.,   50.],
                              [-666., -666.,   49.]])
        assert(np.array_equal(mb.metricValues.data[good,:], expected))


    def testHistogramMetric(self):
        metric = metrics.HistogramMetric()
        slicer = slicers.Healpix2dSlicer(nside=16, bins=[0.5,1.5,2.5])
        sql = ''
        mb = metricBundle.MetricBundle(metric,slicer,sql)
        mbg = metricBundle.MetricBundleGroup({0:mb}, None)
        mbg.setCurrent('')
        mbg.runCurrent('', simData=self.simData)

        good = np.where(mb.metricValues.mask[:,-1] == False)[0]
        expected =  np.array( [[50.,   0., 0.],
                              [0.,   49., 0.]])

        assert(np.array_equal(mb.metricValues.data[good,:], expected))

        # Check that I can run a different statistic
        metric = metrics.HistogramMetric(col='fiveSigmaDepth',
                                         statistic='sum')
        mb = metricBundle.MetricBundle(metric,slicer,sql)
        mbg = metricBundle.MetricBundleGroup({0:mb}, None)
        mbg.setCurrent('')
        mbg.runCurrent('', simData=self.simData)
        expected = np.array( [[25.*50.,   0., 0.],
                              [0.,   24.*49., 0.]])
        assert(np.array_equal(mb.metricValues.data[good,:], expected))

    def testAccumulateMetric(self):
        metric=metrics.AccumulateMetric(col='fiveSigmaDepth')
        slicer = slicers.Healpix2dSlicer(nside=16, bins=[0.5,1.5,2.5])
        sql = ''
        mb = metricBundle.MetricBundle(metric,slicer,sql)
        mbg = metricBundle.MetricBundleGroup({0:mb}, None)
        mbg.setCurrent('')
        mbg.runCurrent('', simData=self.simData)
        good = np.where(mb.metricValues.mask[:,-1] == False)[0]
        expected =  np.array( [[-666., 50.*25, 50.*25.],
                              [-666.,  -666.,  49.*24 ]])
        assert(np.array_equal(mb.metricValues.data[good,:], expected))


if __name__ == "__main__":
    unittest.main()
