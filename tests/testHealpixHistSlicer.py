import numpy as np
import numpy.lib.recfunctions as rfn
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
import healpy as hp
import lsst.sims.maf.slicers as slicers




class testHealpixHistSlicer(unittest.TestCase):
    def setUp(self):
        self.metricValue = np.ma.empty(10,dtype=object)
        for i in np.arange(10):
            self.metricValue[i] = np.random.rand(10)
         
    def testPlotHistogram(self):
        slicer = slicers.HealpixHistSlicer()
       
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1, histStyle=False)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1)
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1,
                                          metricReduce='MeanMetric' )
        num = slicer.plotConsolidatedHist(self.metricValue, binMin=0,binMax=1., binsize=0.1,
                                          metricReduce='MeanMetric', singleHP=2 )
                          

        
if __name__ == "__main__":
    unittest.main()
