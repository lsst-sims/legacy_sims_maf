# Base class for spatial grid & metrics. 
# Major difference between this gridmetric and the base gridmetric is in the plotting methods,
#  as spatial metrics plot the metricValues (rather than globalGrids which plot the simData[column]).
# 
# Also, spatialGridMetric includes calculating a summary statistic over the entire sky.


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class SpatialGridMetric(BaseGridMetric):

    # spatial grid metrics already have all the data necessary for plotting. (plotting metricValues).
    
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
        # Plot the sky map.
        skyfignum = self.grid.plotSkyMap(self.metricValues[metricName],
                                         plotLabel, title=plotTitle)
        if savefig:
            outfile = self._buildOutfileName(metricName, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='sky')
            plt.savefig(outfile, figformat=self.figformat)
        # And then plot the power spectrum if using an appropriate grid.
        if hasattr(self.grid, 'plotPowerSpectrum'):
            psfignum = self.grid.plotPowerSpectrum(self.metricValues[metricName], 
                                                   title=plotTitle, 
                                                   label=plotLabel)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
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
            
        
        
    def computeSummaryStatistics(self, metricName, summaryMetric):
        """Compute summary statistic for metricName, using function summaryMetric. 

        For spatial grids, this summaryMetric (i.e. 'mean', 'min',..) is applied to
        reduce the values in metricName to a single number over the whole sky."""
        if not isinstance(self.metricValues[metricName][0], float):
            raise Exception('Values in metricName should be float - apply reduce function first.')
        summaryNumber = summaryMetric(self.metricValues[metricName])
        return summaryNumber
