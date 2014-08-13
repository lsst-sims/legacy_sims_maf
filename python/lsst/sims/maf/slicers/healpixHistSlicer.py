# This is a slicer that is similar to the healpix slicer,
#but will simply histogram all the metric values, even if they are multivalued

import numpy as np
from .healpixSlicer import HealpixSlicer
import matplotlib.pyplot as plt
import os

class HealpixHistSlicer(HealpixSlicer):

    def __init__(self, **kwargs):
        super(HealpixHistSlicer, self).__init__(**kwargs)
        # Only use the new plotHistogram
        self.plotFuncs = {'plotHistogram':self.plotHistogram}
        self.plotObject = True



        

    def plotHistogram(self, metricValue, fignum=None, binMin=0,
                      binMax=5000., binsize=120.,linestyle='-',
                      title=None,
                      **kwargs):
        """ Take results of a metric that histograms things up"""
        fig = plt.figure(fignum)
        
        #import pdb ; pdb.set_trace()
        
        finalHist = np.sum(metricValue.compressed(), axis=0)
        bins = np.arange(binMin, binMax+binsize,binsize)
        x = np.ravel(zip(bins[:-1], bins[:-1]+binsize))
        y = np.ravel(zip(finalHist, finalHist))
        plt.plot(x,y, linestyle=linestyle)
        plt.title(title)
        plt.xlabel('Observation Gap Time (seconds)')
        plt.ylabel('Count')
        return fig.number
        
