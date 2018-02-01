from builtins import zip
import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.utils.tests


class TestStringCount(unittest.TestCase):

    def testsc(self):
        metric = metrics.StringCountMetric()
        data = np.array(['a', 'a', 'b', 'c', '', '', ''])
        dt = np.dtype([('filter', '|1U')])
        data.dtype = dt
        result = metric.run(data)
        # Check that the metricValue is correct
        expected_results = {'a': 2, 'b': 1, 'c': 1, 'blank': 3}
        for key in expected_results:
            assert(result[key] == expected_results[key])

        # Check that the reduce functions got made and return expected result
        for key in expected_results:
            assert(metric.reduceFuncs[key](result) == expected_results[key])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
