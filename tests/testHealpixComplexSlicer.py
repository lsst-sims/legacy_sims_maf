import matplotlib
matplotlib.use("Agg")
import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
import healpy as hp
import lsst.sims.maf.slicers as slicers
import os



class testHealpixComplexSlicer(unittest.TestCase):
    def setUp(self):
        self.metricValue = np.ma.empty(10,dtype=object)
        for i in np.arange(10):
            self.metricValue[i] = np.random.rand(10)
         
    def testPlotHistogram(self):
        slicer = slicers.HealpixComplexSlicer()

        
        # Check that the various plotting methods run
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1, histStyle=False)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1,
                                          metricReduce='MeanMetric' )
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1,
                                          metricReduce='MeanMetric', singleHP=2 )

        # Check save/restore works.
        plotDictIn = {'binMin':0, 'binMax':1, 'binsize':0.1}
        slicer.writeData('temp.npz', self.metricValue, plotDict=plotDictIn)
        metBack, slicerBack,header,plotDictBack = slicer.readData('temp.npz')
        assert(plotDictBack == plotDictIn)
        assert(slicer == slicerBack)
        
        

    def tearDown(self):
        os.remove('temp.npz')
        
if __name__ == "__main__":
    unittest.main()
