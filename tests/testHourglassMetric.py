import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics


class TestHourglassmetric(unittest.TestCase):

    @unittest.skip("5 April 2016 -- this test causes a malloc error")
    def testHourglassMetric(self):
        """Test the hourglass metric """
        names = ['expMJD', 'night', 'filter']
        types = [float, float, str]
        npts = 50
        data = np.zeros(npts, dtype=zip(names, types))
        day0 = 59000
        data['expMJD'] = np.arange(0, 10, .2)[:npts] + day0
        data['night'] = np.floor(data['expMJD']-day0)
        data['filter'] = 'r'
        data['filter'][-1] = 'g'
        slicePoint = [0]
        metric = metrics.HourglassMetric()
        result = metric.run(data, slicePoint)
        pernight = result['pernight']
        perfilter = result['perfilter']

        assert(np.size(pernight) == np.size(np.unique(data['night'])))
        # All the gaps are larger than 2 min.
        assert(np.size(perfilter) == 2*data.size)
        # Check that the format is right at least
        assert(len(pernight.dtype.names) == 9)


if __name__ == '__main__':

    unittest.main()
