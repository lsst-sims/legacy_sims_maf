# This is a slicer that is similar to the healpix slicer,
#but will simply histogram all the metric values, even if they are multivalued

import numpy as np
from .healpixSlicer import HealpixSlicer
import lsst.sims.maf.metrics as metrics
import matplotlib.pyplot as plt
import warnings

__all__ = ['HealpixComplexSlicer']

class HealpixComplexSlicer(HealpixSlicer):

    def __init__(self, **kwargs):
        """
        Slicer designed to be used with complex metrics where the slicer
        will collapse the metric results, e.g., summing histograms that were computed
        at each healpixel.
        """
        super(HealpixComplexSlicer, self).__init__(**kwargs)
        # Only use the new plotHistogram
        self.plotFuncs = {'plotConsolidatedHist':self.plotConsolidatedHist}
        self.plotObject = True

    def plotConsolidatedHist(self, metricValue, metricReduce='SumMetric', histStyle=True,
                      binMin=0.5, binMax=60.5, binsize=0.5,linestyle='-', singleHP=None,
                      title=None, xlabel=None, units=None, ylabel=None,
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      cumulative=False, xMin=None, xMax=None, yMin=None, yMax=None,
                      logScale='auto',
                      yaxisformat='%.3f', color='b',
                      **kwargs):
        """ This plotting method takes plots/histograms from each healpixel and consolidates them into a
        single histogram that is plotted.  Note that the metric should set the binMin, binMax, binsize kwargs
        to ensure the slicer histogram bins match the metric bins.

        metricReduce: metric name that will be used to combine the histograms bin-by-bin from each healpixel.
                      We currently do not support using metrics that require kwargs.
        histStyle:  Set to True for the data to be plotted as a histogram.
                    If False, a simple line-plot is made.
        binMin/binMax/binSize:  parameters for setting up the bins.
                                Ideally, the metric will set these with the plotDict keyword to ensure they
                                are correct.
        singleHP:  int, plot only the single healpixel id rather than combining the plots.

        """
        fig = plt.figure(fignum)
        if not xlabel:
            xlabel = units

        # If we are only plotting a single healpixel
        if singleHP is not None:
            # only plot a single histogram
            if metricValue[singleHP].mask == True:
                warnings.warn("Pixel %i is masked, nothing to plot for plotConsolidatedHist")
                return
            finalHist = metricValue[singleHP]
        # Combining all the healpixels
        else:
            if metricReduce is not None:
                # Get the data type
                dt = metricValue.compressed()[0].dtype
                # Change an array of arrays (dtype=object) to a 2-d array of correct dtype
                mV = np.array(metricValue.compressed().tolist(), dtype=[('metricValue',dt)])
                # Make an array to hold the combined result
                finalHist = np.zeros(mV.shape[1], dtype=float)
                metric = metrics.BaseMetric.getClass(metricReduce)(col='metricValue')
                # Loop over each bin and use the selected metric to combine the results
                for i in np.arange(finalHist.size):
                    finalHist[i] = metric.run(mV[:,i])

        # Recreate the bins.  Note this needs to be the same as in the metrics.
        bins = np.arange(binMin, binMax+binsize,binsize)
        if histStyle:
            x = np.ravel(zip(bins[:-1], bins[:-1]+binsize))
            y = np.ravel(zip(finalHist, finalHist))
        else:
            # Could use this to plot things like FFT
            x = bins[:-1]
            y = finalHist

        plt.plot(x,y, linestyle=linestyle, label=label, color=color)

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc)

        return fig.number
