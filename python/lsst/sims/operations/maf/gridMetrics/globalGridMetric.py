# Base class for global grids + metrics.
# Major difference between this gridmetric and the base gridmetric is in the plotting methods,
#  as global grids plot the simdata column rather than the metric data directly for a histogram.
#
# Also, globalGridMetric has a different possibility for calculating a summary statistic. Here,
#  the values don't have to be consolidated over the entire sky, but may (if it was a complex value
#  per grid point) have to be reduced to a single number. 

import os
import numpy as np
import matplotlib.pyplot as plt

from .baseGridMetric import BaseGridMetric

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class GlobalGridMetric(BaseGridMetric):

    def __init__(self, figformat='png'):
        """Instantiate global gridMetric object and set up (empty) dictionaries."""
        super(GlobalGridMetric, self).__init__(figformat=figformat)
        self.metricHistValues = {}
        self.metricHistBins = {}
        return

        
    def setGrid(self, grid):
        super(GlobalGridMetric, self).setGrid(grid)
        # Check this grid is a spatial type.
        if self.grid.gridtype != 'GLOBAL':
            raise Exception('Gridtype for grid should be GLOBAL, not %s' %(self.grid.gridtype))
        return

    def runGrid(self, metricList, simData, 
                simDataName='opsim', metadata='', sliceCol=None, histbins=100):
        """Run metric generation over grid.

        metricList = list of metric objects
        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.)
        sliceCol = column for slicing grid, if needed (default None)
        histbins = histogram bins (default = 100, but could pass number of bins or array). """
        super(GlobalGridMetric, self).runGrid(metricList, simData, simDataName=simDataName,
                                              metadata=metadata, sliceCol=sliceCol)
        # Set sliceCol.
        if sliceCol==None:
            sliceCol = simData.dtype.names[0]
        # Set up storage for histograms.
        for m in self.metrics:
            if isinstance(histbins, int):
                histlen = histbins
            else:
                histlen = len(histbins)
            self.metricHistValues[m.name] = np.zeros((len(self.grid), histlen+1), dtype ='int')
            self.metricHistBins[m.name] = np.zeros((len(self.grid), histlen+1), dtype= 'float')
        # Run through all gridpoints and generate histograms 
        for i, g in enumerate(self.grid):
            idxs = self.grid.sliceSimData(g, simData[sliceCol])
            slicedata = simData[idxs]
            if len(idxs)==0:
                # No data at this gridpoint.
                for m in self.metrics:
                    self.metricHistValues[m.name][i] = self.grid.badval
                    self.metricHistBins[m.name][i] = self.grid.badval
            else:
                for m in self.metrics:
                    self.metricHistValues[m.name][i][:-1], self.metricHistBins[m.name][i] = \
                      np.histogram(slicedata[m.colname], bins=histbins)
        return

    def writeMetric(self, metricName, comment='', outfileRoot=None, outDir=None, 
                    dt='float', gridfile=None):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
        outDir = directory to write output data (default '.').
        dt = data type.
        gridfile = the filename for the pickled grid"""
        outfile = self._buildOutfileName(metricName, outDir=outDir, outfileRoot=outfileRoot)
        self.grid.writeMetricData(outfile, self.metricValues[metricName], 
                                  self.metricHistValues[metricName],self.metricHistBins[metricName],
                                  metricName = metricName,
                                  simDataName = self.simDataName[metricName],
                                  metadata = self.metadata[metricName],
                                  comment = comment, dt=dt, gridfile=gridfile,
                                  badval = self.grid.badval)
        return


    
    def plotMetric(self, metricName, 
                   savefig=True, outDir=None, outfileRoot=None):
        """Plot histogram for 'metricName' (global gridMetric)."""
        # Check that metricName refers to plottable (exisiting) histogram data.
        try:
            self.metricHistValues[metricName]
            self.metricHistBins[metricName]
        except KeyError:
            raise ValueError('Metric %s does not have histogram data in gridMetric.' %(metricName))
        # Build plot title and label.
        plotTitle = self.simDataName[metricName] + ' ' + self.metadata[metricName]
        datacolumn = '_'.join(self._dupeMetricName(metricName).split('_')[1:])
        plotTitle += ' ' + datacolumn
        plotLabel = datacolumn
        # Plot the histogram.
        fignum = None
        for i, g in enumerate(self.grid):
            fignum = self.grid.plotBinnedData(self.metricHistBins[metricName][i], 
                                              self.metricHistValues[metricName][i],
                                              datacolumn, filled=True, 
                                              title=plotTitle, fignum=fignum)
        if savefig:
            outfile = self._buildOutfileName(metricName, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)
        return

    def plotComparisons(self, metricNameList, plotTitle=None, legendloc='upper left',
                        savefig=True, outDir=None, outfileRoot=None):
        """Create comparison plots of all metricValues in metricNameList.

        Will create one histogram with all values from metricNameList."""
        # Check is plottable data.
        for m in metricNameList:
            try:
                self.metricHistValues[m]
                self.metricHistBins[m]
            except KeyError:
                metricNameList.remove(m)
        # If there is only one metric remaining, just plot.
        if len(metricNameList) < 2:
            print 'Only one metric left in metricNameList - %s - so defaulting to plotMetric.' \
              %(metricNameList)
            self.plotMetric(metricNameList[0], savefig=savefig, 
                            outDir=outDir, outfileRoot=outfileRoot)
            return    
        # Find unique data to plot. 
        datacolumns = []
        simDataNames = []
        metadatas = []
        metricNames = []
        for m in metricNameList:
            datacol = '_'.join(self._dupeMetricName(m).split('_')[1:])
            if datacol in datacolumns:
                idx = datacolumns.index(datacol)
                if ((self.simDataName[m] != simDataNames[idx]) or 
                    (self.metadata[m] != metadatas[idx])):
                    datacolumns.append(datacol)
                    simDataNames.append(self.simDataName[m])
                    metadatas.append(self.metadata[m])
                    metricNames.append(m)
            else:
                datacolumns.append(datacol)
                simDataNames.append(self.simDataName[m])
                metadatas.append(self.metadata[m])
                metricNames.append(m)
        # Create a plot title from the unique parts of the simData/metadata/metric names.
        if plotTitle == None:
            plotTitle = ''
            if len(set(simDataNames)) == 1:
                plotTitle += ' ' + simDataNames[0]
            if len(set(metadatas)) == 1:
                plotTitle += ' ' + metadatas[0]
            if len(set(datacolumns)) == 1:
                plotTitle += ' ' + datacolumns[0]
            if plotTitle == '':
                # If there were more than one of everything above, join metricNames with commas. 
                plotTitle = ', '.join(list(set(datacolumns)))
        # Create a plot x-axis label (metricLabel)
        plotLabel = ', '.join(list(set(datacolumns)))
        # Plot the histogram.
        histfignum = None
        addLegend = False
        for sim, meta, datacol, metric in zip(simDataNames, metadatas, datacolumns, metricNames): 
            if metric == metricNames[-1:][0]:
                addLegend = True
            legendLabel = sim + ' ' + meta + ' ' + datacol
            print meta, addLegend, metric, metricNames[-1:][0]

            for i, g in enumerate(self.grid):
                histfignum = self.grid.plotBinnedData(self.metricHistBins[metric][i], 
                                                      self.metricHistValues[metric][i],
                                                      plotLabel, title=plotTitle, fignum=histfignum,
                                                      alpha=0.3,
                                                      legendLabel=legendLabel, addLegend=addLegend, legendloc=legendloc)
        if savefig:
            outfile = self._buildOutfileName(plotTitle, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)        
        return
        
        
    def computeSummaryStatistics(self, metricName, summaryMetric=None):
        """Compute summary statistic for metricName, using function summaryMetric.

        Since global grids already are summarized over the sky, this will typically only
        be a 'reduce' function if metricName was a complex metric. Otherwise, the summary
        statistic is the metricValue."""
        if summaryMetric == None:
            summaryNumber = self.metricValues[metricName]
        else:
            good = np.where(self.metricValues[metricName] != self.grid.badval)
            summaryNumber = summaryMetric(self.metricValues[metricName][good])
        return summaryNumber
