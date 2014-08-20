# This is a slicer that is similar to the healpix slicer,
#but will simply histogram all the metric values, even if they are multivalued

import numpy as np
from .healpixSlicer import HealpixSlicer
import lsst.sims.maf.metrics as metrics
import matplotlib.pyplot as plt
import os

class HealpixHistSlicer(HealpixSlicer):

    def __init__(self, **kwargs):
        super(HealpixHistSlicer, self).__init__(**kwargs)
        # Only use the new plotHistogram
        self.plotFuncs = {'plotHistogram':self.plotHistogram}
        self.plotObject = True

    def plotHistogram(self, metricValue, numpyReduce='sum', metricReduce=None, histStyle=True,
                      binMin=0.5, binMax=60.5, binsize=0.5,linestyle='-',
                      title=None, xlabel=None, units=None, ylabel=None,
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      cumulative=False, xMin=None, xMax=None, yMin=None, yMax=None,
                      logScale='auto', flipXaxis=False,
                      yaxisformat='%.3f', color='b',
                      **kwargs):
        """ Take results of a metric that histograms things up"""
        fig = plt.figure(fignum)
        if not xlabel:
            xlabel = units


        if numpyReduce is not None and metricReduce is not None:
            raise Exception('Both numpyReduce and metric Reduce are not None, can only reduce one way')

        if numpyReduce is  None and metricReduce is None:
            raise Exception('Both numpyReduce and metric Reduce are None, need one way to reduce to be set')

        if numpyReduce is not None:
            # just use a numpy function with axis=0 to 
            finalHist = getattr(np,numpyReduce)(metricValue.compressed(), axis=0)

        if metricReduce is not None:
            # can I just change the dtype? no, can't do that
            mV = np.array(metricValue.compressed().tolist())
            finalHist = np.zeros(mV.shape[1], dtype=float)
            for i in np.arange(finalHist.size):
                finalHist[i] = getattr(metrics,metricReduce)(mV[i,:])

        bins = np.arange(binMin, binMax+binsize,binsize)
        if histStyle:
            x = np.ravel(zip(bins[:-1], bins[:-1]+binsize))
            y = np.ravel(zip(finalHist, finalHist))
        else:
            x = bins[:-1]
            y = finalHist
            
        plt.plot(x,y, linestyle=linestyle, label=label, color=color)

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if flipXaxis:
            # Might be useful for magnitude scales.
            x0, x1 = plt.xlim()
            plt.xlim(x1, x0)
        if title is not None:
            plt.title(title)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc)
            
        return fig.number
        
