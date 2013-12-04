# Base class for spatial grid & metrics. 
# Major difference between this gridmetric and the base gridmetric is in the plotting methods,
#  as spatial metrics plot the metricValues (rather than globalGrids which plot the simData[column]).
# 
# Also, spatialGridMetric includes calculating a summary statistic over the entire sky.


import os
import numpy as np
import matplotlib.pyplot as plt

from .baseGridMetric import BaseGridMetric

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class SpatialGridMetric(BaseGridMetric):

    def setGrid(self, grid):
        super(SpatialGridMetric, self).setGrid(grid)
        # Check this grid is a spatial type.
        if self.grid.gridtype != 'SPATIAL':
            raise Exception('Gridtype for grid should be SPATIAL, not %s' %(self.grid.gridtype))
        return
    
    # spatial grid metrics already have all the data necessary for plotting (in metricValues).
    
    def plotMetric(self, metricName, 
                   savefig=True, outDir=None, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        # Check that metricName refers to plottable ('float') data.
        if not isinstance(self.metricValues[metricName][0], float):
            raise ValueError('Metric data in %s is not float-type.' %(metricName))
        # Build plot title and label.
        mname = self._dupeMetricName(metricName)
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

    def plotComparisons(self, metricNameList, histBins=100, histRange=None,
                        plotTitle=None,
                        savefig=False, outDir=None, outfileRoot=None):
        """Create comparison plots of all metricValues in metricNameList.

        Will create one histogram with all values from metricNameList, similarly for 
        power spectra if applicable.
        Will create skymap difference plots if only two metrics: skymap is intersection of 2 areas."""
        # Check is plottable data.
        for m in metricNameList:
            if self.metricValues[m].dtype == 'object':
                metricNameList.remove(m)
        # If have only one metric remaining - 
        if len(metricNameList) < 2:
            print 'Only one metric left in metricNameList - %s - so defaulting to plotMetric.' \
              %(metricNameList)
            self.plotMetric(metricNameList[0], savefig=savefig, 
                            outDir=outDir, outfileRoot=outfileRoot)
            return
        # Else build plot titles. 
        simDataNames = set()
        metadatas = set()
        metricNames = set()
        for m in metricNameList:
            simDataNames.add(self.simDataName[m])
            metadatas.add(self.metadata[m])
            metricNames.add(self._dupeMetricName(m))
        # Create a plot title from the unique parts of the simData/metadata/metric names.
        #  (strip trailing _? values from metric names, as they were probably added from read funct).
        if plotTitle == None:
            plotTitle = ''
            if len(simDataNames) == 1:
                plotTitle += ' ' + list(simDataNames)[0]
            if len(metadatas) == 1:
                plotTitle += ' ' + list(metadatas)[0]
            if len(metricNames) == 1:
                plotTitle += ' ' + list(metricNames)[0]
            if plotTitle == '':
                # If there were more than one of everything above, join metricNames with commas. 
                plotTitle = ', '.join(metricNames)
        # Create a plot x-axis label (metricLabel)
        plotLabel = ', '.join(metricNames)                
        # Plot the histogram.
        histfignum = None
        addLegend = False
        for i,m in enumerate(metricNameList):
            if i == (len(metricNameList)-1):
                addLegend = True
            legendLabel = self.simDataName[m] + ' ' + self.metadata[m] \
              + ' ' + self._dupeMetricName(m)
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
            for i,m in enumerate(metricNameList):
                if i == (len(metricNameList)-1):
                    addLegend = True
                legendLabel = self.simDataName[m] + ' '+  self.metadata[m] \
                  + ' ' + self._dupeMetricName(m)
                psfignum = self.grid.plotPowerSpectrum(self.metricValues[m], addLegend=addLegend,
                                                       fignum = psfignum,
                                                       legendLabel=legendLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the sky map, if only two metricNames.
        if len(metricNameList) == 2:
            # Mask areas where either metric has bad data values, take difference elsewhere.
            mval0 = self.metricValues[metricNameList[0]]
            mval1 = self.metricValues[metricNameList[1]]
            diff = np.where(mval0 == self.grid.badval, self.grid.badval, mval0 - mval1)
            diff = np.where(mval1 == self.grid.badval, self.grid.badval, diff)
            plotLabel = metricNameList[0] + ' - ' + metricNameList[1]
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
