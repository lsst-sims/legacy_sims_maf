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
        mname = metricName.rstrip('_1234567890')
        plotTitle = self.simDataName[metricName] + ' ' + self.metadata[metricName]
        plotTitle += ' ' + mname
        plotLabel = mname
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
                                                   legendLabel=plotLabel)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)
        return

    def plotComparisons(self, metricNameList, histBins=None, histRange=None,
                        savefig=False, outDir=None, outfileRoot=None):
        """Create comparison plots of all metricValues in metricNameList.

        Will create one histogram with all values from metricNameList, similarly for 
        power spectra if applicable. Will create skymap difference plots if only two metrics."""
        # Check is plottable data.
        for m in metricNameList:
            if not isinstance(self.metricValues[m], float):
                metricNameList.remove(m)
        # If have only one metric remaining - 
        if len(metricNameList) < 2:
            self.plotMetric(metricNameList[0], savefig=savefig, 
                            outDir=outDir, outfileRoot=outfileRoot)
        # Else build plot titles. 
        simDataNames = ()
        metadatas = ()
        metricNames = ()
        for m in metricNameList:
            simDataNames.add(self.simDataName[m])
            metadatas.add(self.metadata[m])
            metricNames.add(m.rstrip('_0123456789'))
        # Create a plot title from the unique parts of the simData/metadata/metric names.
        #  (strip trailing _? values from metric names, as they were probably added from read funct).
        plotTitle = ''
        if len(simDataNames) == 1:
            plotTitle += list(simDataNames)[0]
        if len(metadatas) == 1:
            plotTitle += list(metadatas)[0]
        if len(metricNames) == 1:
            plotTitle += list(metricNames)[0]
        if plotTitle == '':
            # If there were more than one of everything above, join metricNames with commas. 
            plotTitle = ', '.join(metricNames)
        # Create a plot x-axis label (metricLabel)
        plotLabel = ', '.join(metricNames)                
        # Plot the histogram.
        histfignum = None
        addLegend = False
        for m in metricNameList:
            if m == metricNameList[-1:]:
                addLegend = True
            legendLabel = self.simDataName[m] + self.metadata[m] + m.rstrip('_0123456789')
            histfignum = self.grid.plotHistogram(self.metricValues[m], metricLabel=plotLabel,
                                                 fignum = histfignum, addLegend=addLegend,
                                                 bins = histBins, histRange = histRange,
                                                 legendLabel=legendLabel, title=plotTitle)
        if savefig:
            outfile = self._buildOutfileName(plotTitle, 
                                             outDir=outDir, outfileRoot=outfileRoot, 
                                             plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)        
        # Plot the power spectrum, if applicable.
        if hasattr(self.grid, 'plotPowerSpectrum'):
            psfignum = None
            addLegend = False
            for m in metricNameList:
                if m == metricNameList[-1:]:
                    addLegend = True
                legendLabel = self.simDataName[m] + self.metadata[m] + m.rstrip('_0123456789')
                psfignum = self.grid.plotPowerSpectrum(self.metricValues[m], addLegend=addLegend,
                                                       legendLabel=legendLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the sky map, if only two metricNames.
        if len(metricNameList) == 2:
            # Mask areas where either metric has bad data values, take difference elsewhere.
            mval0 = self.metricValues[0]
            mval1 = self.metricValues[1]
            diff = np.where((mval0 == self.grid.badval) or (mval1 == self.grid.badval), 
                            self.grid.badval, mval0 - mval1)
            skyfignum = self.grid.plotSkyMap(diff, plotLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='sky')
                plt.savefig(outfile, figformat=self.figformat)
        return
    
        
    def computeSummaryStatistics(self, metricName, summaryMetric):
        """Compute summary statistic for metricName, using function summaryMetric. 

        For spatial grids, this summaryMetric (i.e. 'mean', 'min',..) is applied to
        reduce the values in metricName to a single number over the whole sky."""
        if not isinstance(self.metricValues[metricName][0], float):
            raise Exception('Values in metricName should be float - apply reduce function first.')
        summaryNumber = summaryMetric(self.metricValues[metricName])
        return summaryNumber
