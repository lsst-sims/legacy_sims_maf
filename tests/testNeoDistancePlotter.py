from builtins import zip
import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.plots as plots
import lsst.utils.tests


class TestNeoDistancePlotter(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(61723009)
        names = ['eclipLat', 'eclipLon', 'MaxGeoDist',
                 'NEOHelioX', 'NEOHelioY', 'filter']
        types = [float]*5
        types.append('|S1')
        npts = 100
        self.metricValues = np.zeros(npts, list(zip(names, types)))
        self.metricValues['MaxGeoDist'] = rng.rand(npts)*2.
        self.metricValues['eclipLat'] = rng.rand(npts)
        self.metricValues['NEOHelioX'] = rng.rand(npts)*3-1.5
        self.metricValues['NEOHelioY'] = rng.rand(npts)*3-1.5+1
        self.metricValues['filter'] = 'g'

    def testPlotter(self):
        """
        Just test that it can make a figure without throwing an error.
        """
        plotter = plots.NeoDistancePlotter()
        # Need to wrap in a list because it will usually go through the
        # UniSlicer, and will thus be an array inside a 1-element masked array
        fig = plotter([self.metricValues], None, {})


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
