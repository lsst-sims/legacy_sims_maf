from .plotHandler import PlotHandler
import matplotlib.pylab as plt

__all__=['PlotBundle']

class PlotBundle(object):
    """
    Object designed to help organize multiple MetricBundles that will be plotted
    together using the PlotHandler.
    """

    def __init__(self, bundleList=None, plotDicts=None, plotFunc=None):
        """
        Init object and set things if desired.
        bundleList: A list of bundleDict objects
        plotDicts: A list of dictionaries with plotting kwargs
        plotFunc: A single MAF plotting function
        """
        if bundleList is None:
            self.bundleList = []
        else:
            self.bundleList = bundleList

        if plotDicts is None:
            if len(self.bundleList) > 0:
                self.plotDicts = [{}]
            else:
                self.plotDicts = []
        else:
            self.plotDicts = plotDicts

        self.plotFunc = plotFunc

    def addBundle(self, bundle, plotDict=None, plotFunc=None):
        """
        Add bundle to the object.
        Optionally add a plotDict and/or replace the plotFunc
        """
        self.bundleList.append(bundle)
        if plotDict is not None:
            self.plotDicts.append(plotDict)
        else:
            self.plotDicts.append({})
        if plotFunc is not None:
            self.plotFunc = plotFunc

    def plot(self, outDir='Out', resultsDb=None, closeFigs=True):
        ph = PlotHandler(outDir=args.outDir, resultsDb=resultsDb)
        ph.setMetricBundles(self.bundleList)
        ph.setPlotDict(plotDict=self.plotDicts[0], plotFunc=self.plotFunc)
        ph.plot(self.plotFunc, plotDicts=self.plotDicts)
        if closeFigs:
            plt.close('all')
