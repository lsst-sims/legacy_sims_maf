import matplotlib
matplotlib.use("Agg")
import numpy as np
import numpy.ma as ma
import unittest
import lsst.sims.maf.slicers as slicers
import healpy as hp
import os
import lsst.utils.tests


class TestSlicers(unittest.TestCase):

    def setUp(self):
        self.filenames = []
        self.baseslicer = slicers.BaseSlicer()

    def test_healpixSlicer_obj(self):
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside)).astype('object')
        metricValues = ma.MaskedArray(data=metricValues,
                                      mask=np.where(metricValues < .1, True, False),
                                      fill_value=slicer.badval)
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        metadata = 'testdata'
        slicer.writeData(filename, metricValues, metadata=metadata)
        metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
        np.testing.assert_almost_equal(metricValuesBack, metricValues)
        assert(slicer == slicerBack)
        assert(metadata == header['metadata'])
        attr2check = ['nside', 'nslice', 'columnsNeeded', 'lonCol', 'latCol']
        for att in attr2check:
            assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_healpixSlicer_floats(self):
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues, metadata='testdata')
        metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
        np.testing.assert_almost_equal(metricValuesBack, metricValues)
        assert(slicer == slicerBack)
        attr2check = ['nside', 'nslice', 'columnsNeeded', 'lonCol', 'latCol']
        for att in attr2check:
            assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_healpixSlicer_masked(self):
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricValues = ma.MaskedArray(data=metricValues,
                                      mask=np.where(metricValues < .1, True, False),
                                      fill_value=slicer.badval)
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues, metadata='testdata')
        metricValuesBack, slicerBack, header = self.baseslicer.readData(filename)
        np.testing.assert_almost_equal(metricValuesBack, metricValues)
        assert(slicer == slicerBack)
        attr2check = ['nside', 'nslice', 'columnsNeeded', 'lonCol', 'latCol']
        for att in attr2check:
            assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_oneDSlicer(self):
        slicer = slicers.OneDSlicer(sliceColName='testdata')
        dataValues = np.zeros(10000, dtype=[('testdata', 'float')])
        dataValues['testdata'] = np.random.rand(10000)
        slicer.setupSlicer(dataValues)
        filename = 'oned_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, dataValues[:100])
        dataBack, slicerBack, header = self.baseslicer.readData(filename)
        assert(slicer == slicerBack)
        # np.testing.assert_almost_equal(dataBack,dataValues[:100])
        attr2check = ['nslice', 'columnsNeeded']
        for att in attr2check:
            if type(getattr(slicer, att)).__module__ == 'numpy':
                np.testing.assert_almost_equal(getattr(slicer, att), getattr(slicerBack, att))
            else:
                assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_opsimFieldSlicer(self):
        slicer = slicers.OpsimFieldSlicer()
        names = ['ra', 'dec', 'fieldId']
        dt = ['float', 'float', 'int']
        metricValues = np.random.rand(100)
        fieldData = np.zeros(100, dtype=zip(names, dt))
        fieldData['ra'] = np.random.rand(100)
        fieldData['dec'] = np.random.rand(100)
        fieldData['fieldId'] = np.arange(100)
        names = ['data1', 'data2', 'fieldId']
        simData = np.zeros(100, dtype=zip(names, dt))
        simData['data1'] = np.random.rand(100)
        simData['fieldId'] = np.arange(100)
        slicer.setupSlicer(simData, fieldData)
        filename = 'opsimslicer_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues)
        metricBack, slicerBack, header = self.baseslicer.readData(filename)
        assert(slicer == slicerBack)
        np.testing.assert_almost_equal(metricBack, metricValues)
        attr2check = ['nslice', 'columnsNeeded', 'lonCol', 'latCol', 'simDataFieldIDColName']
        for att in attr2check:
            if type(getattr(slicer, att)).__name__ == 'dict':
                for key in getattr(slicer, att).keys():
                    np.testing.assert_almost_equal(getattr(slicer, att)[key], getattr(slicerBack, att)[key])
            else:
                assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_unislicer(self):
        slicer = slicers.UniSlicer()
        data = np.zeros(1, dtype=[('testdata', 'float')])
        data[:] = np.random.rand(1)
        slicer.setupSlicer(data)
        filename = 'unislicer_test.npz'
        self.filenames.append(filename)
        metricValue = np.array([25.])
        slicer.writeData(filename, metricValue)
        dataBack, slicerBack, header = self.baseslicer.readData(filename)
        assert(slicer == slicerBack)
        np.testing.assert_almost_equal(dataBack, metricValue)
        attr2check = ['nslice', 'columnsNeeded']
        for att in attr2check:
            assert(getattr(slicer, att) == getattr(slicerBack, att))

    def test_complex(self):
        # Test case where there is a complex metric
        nside = 8
        slicer = slicers.HealpixSlicer(nside=nside)
        data = np.zeros(slicer.nslice, dtype='object')
        for i, ack in enumerate(data):
            n_el = np.random.rand(1)*4  # up to 4 elements
            data[i] = np.arange(n_el)
        filename = 'heal_complex.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, data)
        dataBack, slicerBack, header = self.baseslicer.readData(filename)
        assert(slicer == slicerBack)
        # This is a crazy slow loop!
        for i, ack in enumerate(data):
            np.testing.assert_almost_equal(dataBack[i], data[i])

    def test_nDSlicer(self):
        colnames = ['test1', 'test2', 'test3']
        data = []
        for c in colnames:
            data.append(np.random.rand(1000))
        dv = np.core.records.fromarrays(data, names=colnames)
        slicer = slicers.NDSlicer(colnames, binsList=10)
        slicer.setupSlicer(dv)
        filename = 'nDSlicer_test.npz'
        self.filenames.append(filename)
        metricdata = np.zeros(slicer.nslice, dtype='float')
        for i, s in enumerate(slicer):
            metricdata[i] = i
        slicer.writeData(filename, metricdata)
        dataBack, slicerBack, header = self.baseslicer.readData(filename)
        assert(slicer == slicerBack)
        np.testing.assert_almost_equal(dataBack, metricdata)

    def tearDown(self):
        for filename in self.filenames:
            os.remove(filename)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
