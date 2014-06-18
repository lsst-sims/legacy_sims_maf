import numpy as np
import numpy.ma as ma
import unittest
#from lsst.sims.maf.metrics import SimpleMetrics as sm
import lsst.sims.maf.slicers as slicers
import healpy as hp
import os


class TestSlicers(unittest.TestCase):
    def setUp(self):
        self.filenames=[]

    def test_healpixSlicer_obj(self):
        nside = 128
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside)).astype('object')
        metricValues = ma.MaskedArray(data=metricValues, mask = np.where(metricValues < .1, True, False), fill_value=slicer.badval)
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        metadata = 'poop'
        slicer.writeData(filename, metricValues, metadata=metadata)
        metricValuesBack,slicerBack,header = slicer.readData(filename)
        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(slicer == slicerBack) 
        assert(metadata == header['metadata'])
        attr2check = ['nside', 'nbins', 'columnsNeeded', 'bins', 'spatialkey1', 'spatialkey2']
        for att in attr2check:
            assert(getattr(slicer,att) == getattr(slicerBack,att))
        
    def test_healpixSlicer_floats(self):
        nside = 128
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues, metadata='poop')
        metricValuesBack,slicerBack,header = slicer.readData(filename)
        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(slicer == slicerBack) #I don't think this is the right way to compare
        attr2check = ['nside', 'nbins', 'columnsNeeded', 'bins', 'spatialkey1', 'spatialkey2']
        for att in attr2check:
            assert(getattr(slicer,att) == getattr(slicerBack,att))
       
        
    def test_healpixSlicer_masked(self):
        nside = 128
        slicer = slicers.HealpixSlicer(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricValues = ma.MaskedArray(data=metricValues, mask = np.where(metricValues < .1, True, False), fill_value=slicer.badval)
        metricName = 'Noise'
        filename = 'healpix_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues, metadata='poop')
        metricValuesBack,slicerBack,header = slicer.readData(filename)

        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(slicer == slicerBack) #I don't think this is the right way to compare
        attr2check = ['nside', 'nbins', 'columnsNeeded', 'bins', 'spatialkey1', 'spatialkey2']
        for att in attr2check:
            assert(getattr(slicer,att) == getattr(slicerBack,att))


    def test_oneDSlicer(self):
        slicer=slicers.OneDSlicer(sliceDim='poop')
        dataValues = np.zeros(10000, dtype=[('poop','float')])
        dataValues['poop'] = np.random.rand(10000)
        slicer.setupSlicer(dataValues)
        filename = 'oned_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, dataValues[:100])

        dataBack,slicerBack,header = slicer.readData(filename)
        assert(slicer == slicerBack)
        assert(np.all(slicer.bins == slicerBack.bins))
        #np.testing.assert_almost_equal(dataBack,dataValues[:100])
        attr2check = ['nbins', 'columnsNeeded', 'bins']
        for att in attr2check:
            if type(getattr(slicer,att)).__module__ == 'numpy':
                np.testing.assert_almost_equal(getattr(slicer,att), getattr(slicerBack,att))
            else:
                assert(getattr(slicer,att) == getattr(slicerBack,att))

    def test_opsimFieldSlicer(self):
        slicer=slicers.OpsimFieldSlicer(np.arange(100))
        names=['fieldRA','fieldDec','fieldID',]
        dt = ['float','float','int']
        metricValues = np.random.rand(100)
        fieldData = np.zeros(100, dtype=zip(names,dt))
        fieldData['fieldRA'] = np.random.rand(100)
        fieldData['fieldDec'] = np.random.rand(100)
        fieldData['fieldID'] = np.arange(100)
        names=['data1','data2','fieldID',]
        simData = np.zeros(100, dtype=zip(names,dt))
        simData['data1'] = np.random.rand(100)
        simData['fieldID'] = np.arange(100)
        slicer.setupSlicer(simData,fieldData)
        filename = 'opsimslicer_test.npz'
        self.filenames.append(filename)
        slicer.writeData(filename, metricValues)
        metricBack, slicerBack,header = slicer.readData(filename)
        assert(slicer == slicerBack)
        np.testing.assert_almost_equal(metricBack,metricValues)
        attr2check = ['nbins', 'columnsNeeded', 'bins', 'spatialkey1', 'spatialkey2','simDataFieldIDColName']
        for att in attr2check:
            if type(getattr(slicer,att)).__name__ == 'dict':
                for key in getattr(slicer,att).keys():
                    np.testing.assert_almost_equal(getattr(slicer,att)[key], getattr(slicerBack,att)[key])
            else:
                assert(getattr(slicer,att) == getattr(slicerBack,att))

    def test_unislicer(self):
        slicer = slicers.UniSlicer()
        data = np.zeros(1, dtype=[('poop','float')])
        data[:] = np.random.rand(1)
        slicer.setupSlicer(data)
        filename='unislicer_test.npz'
        self.filenames.append(filename)
        metricValue=np.array([25.])
        slicer.writeData(filename, metricValue)
        dataBack, slicerBack,header = slicer.readData(filename)
        assert(slicer == slicerBack)
        np.testing.assert_almost_equal(dataBack,metricValue)
        attr2check = ['nbins', 'columnsNeeded', 'bins']
        for att in attr2check:
            assert(getattr(slicer,att) == getattr(slicerBack,att))


    def test_complex(self):
        """Test case where there is a complex metric """
        nside = 32
        slicer = slicers.HealpixSlicer(nside=nside)
        data = np.zeros(slicer.nbins, dtype='object')
        for i,ack in enumerate(data):
            n_el = np.random.rand(1)*4 # up to 4 elements
            data[i] = np.arange(n_el)
        filename = 'heal_complex.npz'
        self.filenames.append(filename)
        slicer.writeData(filename,data)
        dataBack,slicerBack,header = slicer.readData(filename)
        assert(slicer == slicerBack)
        # This is a crazy slow loop!  
        for i, ack in enumerate(data):
            np.testing.assert_almost_equal(dataBack[i],data[i])
        
#    def test_nDSlicer(self):
#        colnames = ['ack1','ack2','poop']
#        types = ['float','float','int']
#        data = np.zeros(1000, dtype=zip(colnames,types))
#        slicer = slicers.NDSlicer()
#        slicer.setupSlicer([data['ack1'], data['ack2'], data['poop']])
#        filename = 'nDBInner_test.npz'
#        slicer.writeData(filename,data)
#        dataBack,slicerBack,header = slicer.readData(filename)
#        assert(slicer == slicerBack)
#        np.testing.assert_almost_equal(dataBack,data)
       

    def tearDown(self):
        for filename in self.filenames:
            os.remove(filename)
    
if __name__ == '__main__':
    unittest.main()
 
