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

    def plotHistogram(self, metricValue, binMin=0.5,
                      binMax=60.5, binsize=0.5,linestyle='-',
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
        
        # Need to think of best way to allow various ways to collapse.  Leave as "sum" for now,
        # but will probably want to have mean and median as options eventually.
        finalHist = np.sum(metricValue.compressed(), axis=0)
        bins = np.arange(binMin, binMax+binsize,binsize)
        x = np.ravel(zip(bins[:-1], bins[:-1]+binsize))
        y = np.ravel(zip(finalHist, finalHist))
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
        
