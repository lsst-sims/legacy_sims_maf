import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import warnings
import os

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.maps as maps


def makeDataValues(size=100, min=0., max=1., random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random:
        randorder = np.random.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    ids = np.arange(size)
    datavalues = np.array(zip(datavalues, datavalues, ids),
                          dtype=[('fieldRA', 'float'),
                                 ('fieldDec', 'float'), ('fieldID', 'int')])
    return datavalues

def makeFieldData():
    names=['fieldID', 'fieldRA','fieldDec']
    types = [int, float, float]
    fieldData = np.zeros(100, dtype=zip(names,types))
    fieldData['fieldID'] = np.arange(100)
    fieldData['fieldRA'] = np.random.rand(100)
    fieldData['fieldDec'] = np.random.rand(100)
    return fieldData

class TestMaps(unittest.TestCase):

    def testDustMap(self):

        mapPath = os.environ['SIMS_DUSTMAPS_DIR']

        if os.path.isfile(os.path.join(mapPath, 'DustMaps/dust_nside_128.npz')):

            data = makeDataValues()
            dustmap = maps.DustMap()

            slicer1 = slicers.HealpixSlicer()
            slicer1.setupSlicer(data)
            result1 = dustmap.run(slicer1.slicePoints)
            assert('ebv' in result1.keys())

            fieldData = makeFieldData()

            slicer2 = slicers.OpsimFieldSlicer()
            slicer2.setupSlicer(data, fieldData)
            result2 = dustmap.run(slicer2.slicePoints)
            assert('ebv' in result2.keys())

            # Check interpolation works
            dustmap = maps.DustMap(interp=True)
            result3 = dustmap.run(slicer2.slicePoints)

            # Check warning gets raised
            dustmap = maps.DustMap(nside=4)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result4 = dustmap.run(slicer1.slicePoints)
                self.assertTrue("nside" in str(w[-1].message))
        else:
            warnings.warn('Did not find dustmaps, not running testMaps.py')

    def testStarMap(self):
        mapPath = os.environ['SIMS_DUSTMAPS_DIR']

        if os.path.isfile(os.path.join(mapPath, 'StarMaps/starDensity_r_nside_64.npz')):
            data = makeDataValues()
            # check that it works if nside does not match map nside of 64
            nsides = [32,64,128]
            for nside in nsides:
                starmap = maps.StellarDensityMap()
                slicer1 = slicers.HealpixSlicer(nside=nside)
                slicer1.setupSlicer(data)
                result1 = starmap.run(slicer1.slicePoints)
                assert('starMapBins' in result1.keys())
                assert('starLumFunc' in result1.keys())
                assert(np.max(result1['starLumFunc'] > 0))

            fieldData = makeFieldData()

            slicer2 = slicers.OpsimFieldSlicer()
            slicer2.setupSlicer(data, fieldData)
            result2 = starmap.run(slicer2.slicePoints)
            assert('starMapBins' in result2.keys())
            assert('starLumFunc' in result2.keys())
            assert(np.max(result2['starLumFunc'] > 0))

        else:
            warnings.warn('Did not find stellar density map, skipping test.')

if __name__ == '__main__':

    unittest.main()
