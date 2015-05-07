import numpy as np
import warnings
import healpy as hp
from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lsst.sims.maf.metrics import fOArea, fONv, SumMetric

from .plotHandler import BasePlotter

__all__ = ['FOPlot']

class FOPlot(BasePlotter):
    def __init__(self):
        self.plotType = 'FO'
        self.defaultPlotDict = {'title':None, 'xlabel':'Number of visits',
                                'ylabel':'Area (1000s of square degrees)',
                                'scale':None, 'Asky':18000., 'Nvisits':825,
                                'xMin':None, 'xMax':None, 'yMin':None, 'yMax':None,
                                'linewidth':2, 'reflinewidth':2}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Note that Asky and Nvisit need to be set for both the slicer and the summary statistic
          for the plot and returned summary stat values to be consistent!
        """
        if not hasattr(slicer, 'nside'):
            raise ValueError('FOPlot to be used with healpix or healpix derived slicers.')
        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        if plotDict['scale'] is None:
            plotDict['scale'] = (hp.nside2pixarea(slicer.nside, degrees=True)  / 1000.0)

        # Expect metricValue to be something like number of visits
        cumulativeArea = np.arange(1,metricValue.compressed().size+1)[::-1]*plotDict['scale']
        plt.plot(np.sort(metricValue.compressed()), cumulativeArea,'k-', linewidth=plotDict['linewidth'], zorder = 0)
        # This is breaking the rules and calculating the summary stats in two places.
        # Could just calculate summary stats and pass in labels.
        rarr = np.array(zip(metricValue.compressed()),
                dtype=[('fO', metricValue.dtype)])
        fOArea_value = fOArea(col='fO', Asky=plotDict['Asky'], norm=False,
                              nside=slicer.nside).run(rarr)
        fONv_value = fONv(col='fO', Nvisit=plotDict['Nvisit'], norm=False,
                          nside=slicer.nside).run(rarr)
        fOArea_value_n = fOArea(col='fO', Asky=plotDict['Asky'], norm=True,
                                nside=slicer.nside).run(rarr)
        fONv_value_n = fONv(col='fo',Nvisit=plotDict['Nvisit'], norm=True,
                            nside=slicer.nside).run(rarr)

        plt.axvline(x=plotDict['Nvisit'], linewidth=plotDict['reflinewidth'], color='b')
        plt.axhline(y=plotDict['Asky']/1000., linewidth=plotDict['reflinewidth'],color='r')

        plt.axhline(y=fONv_value/1000., linewidth=plotDict['reflinewidth'], color='b',
                    alpha=.5, label=r'f$_0$ Nvisits=%.3g' %fONv_value_n)
        plt.axvline(x=fOArea_value , linewidth=plotDict['reflinewidth'], color='r',
                    alpha=.5, label='f$_0$ Area=%.3g'%fOArea_value_n)
        plt.legend(loc='lower left', fontsize='small', numpoints=1)

        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        plt.title(plotDict['title'])

        xMin = plotDict['xMin']
        xMax = plotDict['xMax']
        yMin = plotDict['yMin']
        yMax = plotDict['yMax']
        if (xMin is not None) & (xMax is not None):
            plt.xlim([xMin,xMax])
        if (yMin is not None) & (yMax is not None):
            plt.ylim([yMin,yMax])
        return fig.number


def consolidateHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'SummaryHistogram'
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'ylabel':None, 'label':None,
                                'cumulative':False, 'xMin':None, 'xMax':None, 'yMin':None, 'yMax':None,
                                'color':'b', 'linestyle':'-', 'histStyle':True,
                                'metricReduce':'SumMetric', 'bins':None}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        This plotting method takes the set of values @ each metricValue (i.e. a histogram at each slicer point)
        and consolidates them into a single histogram that is plotted, effectively marginalizing over the sky.
        Note that the plotDict['bins'] here should match the bins used with the metric and that
        plotDict['metricReduce'] will specify the metric used to combine the histograms bin-by-bin from each slicepoint.
        """
        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        # Combine the metric values across all slicePoints.
        if not isinstance(plotDict['metricReduce'], metrics.BaseMetric):
            raise ValueError('Expected plotDict[metricReduce] to be a MAF metric object.')
        # Get the data type
        dt = metricValue.compressed()[0].dtype
        # Change an array of arrays (dtype=object) to a 2-d array of correct dtype
        mV = np.array(metricValue.compressed().tolist(), dtype=[('metricValue',dt)])
        # Make an array to hold the combined result
        finalHist = np.zeros(mV.shape[1], dtype=float)
        metric = plotDict['metricReduce']
        metric.colname = 'metricValue'
        # Loop over each bin and use the selected metric to combine the results
        for i in np.arange(finalHist.size):
            finalHist[i] = metric.run(mV[:,i])
        bins = plotDict['bins']
        if plotDict['histStyle']:
            x = np.ravel(zip(bins[:-1], bins[:-1]+binsize))
            y = np.ravel(zip(finalHist, finalHist))
        else:
            # Could use this to plot things like FFT
            x = bins[:-1]
            y = finalHist
        # Make the plot.
        plt.plot(x,y, linestyle=plotDict['linestyle'], label=plotDict['label'], color=plotDict['color'])
        # Add labels.
        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        plt.title(plotDict['title'])
        return fig.number
