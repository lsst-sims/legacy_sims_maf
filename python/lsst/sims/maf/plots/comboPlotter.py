from builtins import object
from lsst.sims.maf.plots import PlotHandler
from lsst.sims.maf.runComparison import RunComparison
from lsst.sims.maf.metricBundles import metricBundle

__all__ = ['comboPlotter']

class comboPlotter(object):

    def __init__(self, runList, metricName, metricMetadata, slicerName, baseDir = None):
        if baseDir is None:
            self.baseDir = '.'
        else:
            self.baseDir = baseDir
        self.runList = runList
        self.metricName = metricName
        self.metricMetadata = metricMetadata
        self.slicerName = slicerName
        self._makeFileDict()

    def _makeFileDict(self):

        self.compObj = RunComparison(baseDir=self.baseDir,
                                     runlist=self.runList,
                                     verbose=False)

        self.fileDict = self.compObj.getFileNames(self.metricName,
                                                  self.metricMetadata,
                                                  self.slicerName)

    def makePlotBundle(self, nrows = None, ncols = None, plotDicts = None):
        bundleList = []
        plotDict = {}
        for i, run in enumerate(self.runList):
            if ((nrows is not None) and (ncols is not None)):
                plotDict['subplot'] = int(str(nrows) + str(ncols) + str(i + 1))
            else:
                plotDict['subplot'] = '111'
            run_bundle = metricBundle.createEmptyMetricBundle()
            run_bundle.read(self.fileDict[run])
            run_bundle.setPlotDict(plotDict)
            bundleList.append(run_bundle)

        self.nrows = nrows
        self.ncols = ncols
        self.bundleList = bundleList

    def subplotDicts(self, figsize):
        dictList = [{} for r in self.runList]
        for d,r in zip(dictList,self.runList):
            d['figsize'] = figsize
            d['title'] = r + "\n"+ (self.metricName + ' ' + self.metricMetadata)
            d['label'] = ''
        return dictList

    def updateDictsList(self, dictList = None, keys = None, values = None):
        if dictList is None:
            dictList = [{} for r in self.runList]
        else:
            dictList = dictList
        for d,r in zip(dictList,self.runList):
            for k,v in zip(keys,values):
                d[k] = v

        return dictList






    def multiPlot(self, plotFunc = None, plotDicts = None):
        ph = PlotHandler(savefig=False)
        ph.setMetricBundles(self.bundleList)
        ph.setPlotDicts(plotDicts, reset = True)

        if plotFunc is None:
            for func in self.bundleList[0].plotFuncs:
                fignum = ph.plot(plotFunc=func)
        else:
            fignum = ph.plot(plotFunc=plotFunc)

        self.ph = ph

        return fignum
