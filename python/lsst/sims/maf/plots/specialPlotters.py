import numpy as np
import warnings
import healpy as hp
from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from lsst.sims.maf.metrics.summaryMetrics import fOArea, fONv

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
        plotDict.update(defaultPlotDict)
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
        fOArea_value = fOArea(col='fO', Asky=Asky, norm=False,
                              nside=slicer.nside).run(rarr)
        fONv_value = fONv(col='fO', Nvisit=Nvisit, norm=False,
                          nside=slicer.nside).run(rarr)
        fOArea_value_n = fOArea(col='fO', Asky=Asky, norm=True,
                                nside=slicer.nside).run(rarr)
        fONv_value_n = fONv(col='fo',Nvisit=Nvisit, norm=True,
                            nside=slicer.nside).run(rarr)

        plt.axvline(x=Nvisit, linewidth=plotDict['reflinewidth'], color='b')
        plt.axhline(y=Asky/1000., linewidth=plotDict['reflinewidth'],color='r')

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
