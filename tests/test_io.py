import numpy as np
import unittest
from lsst.sims.operations.maf.metrics import SimpleMetrics as sm
import lsst.sims.operations.maf.binners as binners
import healpy as hp
import numpy.ma as ma


class TestBinners(unittest.TestCase):
    def setUp(self):
        pass

    def test_healpixBinner_obj(self):
        nside = 128
        binner = binners.HealpixBinner(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside)).astype('object')
        metricValues = ma.MaskedArray(data=metricValues, mask = np.where(metricValues < .1, True, False), fill_value=binner.badval)
        metricName = 'Noise'
        filename = 'healpix_test.fits'
        binner.writeMetricData(filename, metricValues, metadata='poop')
        metricValuesBack,binnerBack,header = binner.readMetricData(filename)
        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(binner == binnerBack) #I don't think this is the right way to compare

        
    def test_healpixBinner_floats(self):
        nside = 128
        binner = binners.HealpixBinner(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricName = 'Noise'
        filename = 'healpix_test.fits'
        binner.writeMetricData(filename, metricValues, metadata='poop')
        metricValuesBack,binnerBack,header = binner.readMetricData(filename)

        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(binner == binnerBack) #I don't think this is the right way to compare

    def test_healpixBinner_masked(self):
        nside = 128
        binner = binners.HealpixBinner(nside=nside)
        metricValues = np.random.rand(hp.nside2npix(nside))
        metricValues = ma.MaskedArray(data=metricValues, mask = np.where(metricValues < .1, True, False), fill_value=binner.badval)
        metricName = 'Noise'
        filename = 'healpix_test.fits'
        binner.writeMetricData(filename, metricValues, metadata='poop')
        metricValuesBack,binnerBack,header = binner.readMetricData(filename)

        np.testing.assert_almost_equal(metricValuesBack,metricValues)
        assert(binner == binnerBack) #I don't think this is the right way to compare


    def test_oneDBinner(self):
        binner=binners.OneDBinner()
        dataValues = np.zeros(10000, dtype=[('poop','float')])
        dataValues['poop'] = np.random.rand(10000)
        binner.setupBinner(dataValues, 'poop')
        filename = 'oned_test.fits'
        binner.writeMetricData(filename, dataValues[:100])

        dataBack,binnerBack,header = binner.readMetricData(filename)
        assert(binner == binnerBack)
        assert(np.all(binner.bins == binnerBack.bins))
        #np.testing.assert_almost_equal(dataBack,dataValues[:100])

    def test_opsimFieldBinner(self):
        binner=binners.OpsimFieldBinner(np.arange(100))
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
        binner.setupBinner(simData,'fieldID',fieldData,
                           'fieldID', 'fieldRA','fieldDec')
        filename = 'opsimbinner_test.fits'
        binner.writeMetricData(filename, metricValues)
        metricBack, binnerBack,header = binner.readMetricData(filename)
        assert(binner == binnerBack)
        np.testing.assert_almost_equal(metricBack,metricValues)


    def test_unibinner(self):
        binner = binners.UniBinner()
        data = np.zeros(1, dtype=[('poop','float')])
        data[:] = np.random.rand(1)
        binner.setupBinner(data)
        filename='unibinner_test.fits'
        metricValue=np.array([25.])
        binner.writeMetricData(filename, metricValue)
        dataBack, binnerBack,header = binner.readMetricData(filename)
        assert(binner == binnerBack)
        np.testing.assert_almost_equal(dataBack,metricValue)


#    def test_nDBinner(self):
#        colnames = ['ack1','ack2','poop']
#        types = ['float','float','int']
#        data = np.zeros(1000, dtype=zip(colnames,types))
#        binner = binners.NDBinner()
#        binner.setupBinner([data['ack1'], data['ack2'], data['poop']])
#        filename = 'nDBInner_test.fits'
#        binner.writeMetricData(filename,data)
#        dataBack,binnerBack,header = binner.readMetricData(filename)
#        assert(binner == binnerBack)
#        np.testing.assert_almost_equal(dataBack,data)
       
        
if __name__ == '__main__':
    unittest.main()
 
