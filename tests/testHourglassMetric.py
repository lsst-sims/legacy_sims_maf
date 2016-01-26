import matplotlib
matplotlib.use("Agg")
import numpy as np
import healpy as hp
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers


class TestHourglassmetric(unittest.TestCase):

    def testHourglassMetric(self):
        """Test the hourglass metric """
        names = [ 'expMJD', 'night','filter']
        types = [float,float,str]
        data = np.zeros(10, dtype = zip(names,types))
        data['night'] = np.round(np.arange(0, 2, .1))[:10]
        data['expMJD'] = np.sort(np.random.rand(10)) + data['night']
        data['filter'] = 'r'
        slicePoint = [0]
        metric = metrics.HourglassMetric()
        result = metric.run(data, slicePoint)
        pernight = result['pernight']
        perfilter = result['perfilter']
        # Check that the format is right at least
        assert(np.size(perfilter) == 2*data.size)
        assert(len(pernight.dtype.names) == 9)


if __name__ == '__main__':

    unittest.main()
