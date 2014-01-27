import numpy as np
import unittest
from lsst.sims.operations.maf.metrics import SimpleMetrics as sm
import lsst.sims.operations.maf.binners as binners
import healpy as hp
import matplotlib.pyplot as plt


class TestBinners(unittest.TestCase):

    def setUp(self):
        pass
        
    def test_healpixBinner(self):
        nside = 128
        binner = binners.HealpixBinner(nside=nside)
        metricValue = np.random.rand(hp.nside2npix(nside))
        metricName = 'Noise'
        binner.plotSkyMap(metricValue, metricName)
        binner.plotHistogram(metricValue, metricName)
        binner.plotPowerSpectrum(metricValue)
        plt.show() #wonder what the correct way to test plotting is?
        

    def test_oneDBinner(self):
        nbins = 100
        dataCol = np.zeros(1000, dtype=[('Noise','float')] )
        dataCol['Noise'] += np.random.rand(1000)
        binner = binners.OneDBinner()
        binner.setupBinner(dataCol['Noise'],'Noise', nbins=nbins)
        binner.plotBinnedData(dataCol['Noise'][:binner.nbins], 'Noise')
        plt.show()

    def test_uniBinner(self):
        binner = binners.UniBinner()
        dataCol = np.zeros(1000, dtype=[('Noise','float')] )
        dataCol['Noise'] += np.random.rand(1000)
        binner.setupBinner(dataCol)
        dataslice = binner.sliceSimData(0)
        self.assertEqual(len(dataslice[0]), len(dataCol['Noise']))
        
if __name__ == '__main__':
    unittest.main()
    
