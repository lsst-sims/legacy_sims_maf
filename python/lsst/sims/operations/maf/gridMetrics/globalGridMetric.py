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

    def setGrid(self, grid):
        super(SpatialGridMetric, self).setGrid(grid)
        # Check this grid is a spatial type.
        if self.grid.gridtype != 'GLOBAL':
            raise Exception('Gridtype for grid should be GLOBAL, not %s' %(self.grid.gridtype))
        return
    
    # Have to get simdata in here .. but how? (note that it's more than just one simdata - one
    #  column per metric, but could come from different runs)
    
    def plotMetric(self, metricName, 
                   savefig=True, outDir=None, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        # Check that metricName refers to plottable ('float') data.
        if not isinstance(self.metricValues[metricName][0], float):
            raise ValueError('Metric data in %s is not float-type.' %(metricName))
        # Build plot title and label.
        plotTitle = self.simDataName[metricName] + ' ' + self.metadata[metricName]
        plotTitle += ' ' + metricName
        plotLabel = metricName
        # Plot the histogram.
        histfignum = self.grid.plotHistogram(self.metricValues[metricName], 
                                             plotLabel, title=plotTitle)
        if savefig:
            outfile = self._buildOutfileName(metricName, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)
        return

    def plotComparisons(self, metricNameList, 
                        savefig=True, outDir=None, outfileRoot=None):
        """Create comparison plots of all metricValues in metricNameList.

        Will create one histogram with all values from metricNameList, similarly for 
        power spectra if applicable. Will create skymap difference plots if only two metrics."""
        # Check is plottable data.
        for m in metricNameList:
            if not isinstance(self.metricValues[m], float):
                metricNameList.remove(m)
        # Build plot title and label.
        plotTitle = self.simDataName[metricName] + ' ' + self.metadata[metricName]
        plotTitle += ' ' + metricName
        plotLabel = metricName
        # Plot the histogram.
        histfignum = self.grid.plotHistogram(self.metricValues[metricName], 
                                             plotLabel, title=plotTitle)
        if savefig:
            outfile = self._buildOutfileName(metricName, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)        
            
        
        
    def computeSummaryStatistics(self, metricName, summaryMetric=None):
        """Compute summary statistic for metricName, using function summaryMetric.

        Since global grids already are summarized over the sky, this will typically only
        be a 'reduce' function if metricName was a complex metric. Otherwise, the summary
        statistic is the metricValue."""
        if summaryMetric == None:
            summaryNumber = self.metricValues[metricName]
        else:
            summaryNumber = summaryMetric(self.metricValues[metricName])
        return summaryNumber
