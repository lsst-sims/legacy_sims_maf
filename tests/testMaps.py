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

            names=['fieldID', 'fieldRA','fieldDec']
            types = [int, float, float]
            fieldData = np.zeros(100, dtype=zip(names,types))
            fieldData['fieldID'] = np.arange(100)
            fieldData['fieldRA'] = np.random.rand(100)
            fieldData['fieldDec'] = np.random.rand(100)

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
            print 'Did not find dustmaps, not running testMaps.py'
        

if __name__ == '__main__':

    unittest.main()
