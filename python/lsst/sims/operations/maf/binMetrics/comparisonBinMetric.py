# comparisonBinMetric is intended to generate comparison plots. 
# This basically holds baseBinMetrics, but multiple baseBinMetrics which will be used to 
#  create comparison plots.

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import lsst.sims.operations.maf.binners as binners
from .baseBinMetric import BaseBinMetric


import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

class ComparisonBinMetric(object):
    """ComparisonBinMetric"""
    def __init__(self, figformat='png', verbose=True):
        self.figformat = figformat
        self.verbose = verbose
        # The bbmDict stores the baseBinMetrics, keyed by a number. This same number is used to
        #  keep the information of the filename that the metric data came from, 
        #  the metric names, and the metric values.
        self.bbmList = []
        self.bbmDict = {}
        self.bbmFiles = {}
        self.bbmMetricNames = {}
        
    def readMetrics(self, filenames):
        """Read metric data values and binners from filenames."""
        if not hasattr(filenames, '__iter__'):
            filenames = [filenames, ]
        dictlen = len(self.bbmDict)
        for i, f in enumerate(filenames):
            bbm = BaseBinMetric(figformat=self.figformat)
            # Read the (single) metric data value and binner into a baseBinMetric.
            bbm.readMetricValues(f)
            dictnum = i + dictlen
            self.bbmList.append(dictnum)
            self.bbmDict[dictnum] = bbm
            self.bbmFiles[dictnum] = f
            self.bbmMetricNames[dictnum] = bbm.metricNames
            if self.verbose:
                print 'Read metrics %s from %s with binner %s to dictnum %d' %(bbm.metricNames, 
                                                                               f,
                                                                               bbm.binner.binnertype, dictnum)

    def setMetrics(self, bbm):
        """Add a basebinmetric object directly."""
        dictnum = len(self.bbmDict)
        self.bbmList.append(dictnum)
        self.bbmFiles[dictnum] = None    
        self.bbmMetricNames[dictnum] = bbm.metricNames
        if self.verbose:
            print 'Added metrics %s from basebinmetric with binner %s' %(bbm.metricNames,
                                                                         bbm.binner.binnertype)

    def uniqueMetrics(self):
        """Examine metric names and return the set of unique metric names"""
        uniqueMetrics = set()
        bbm = self.bbmDict[0]
        for bbmIdx in self.bbmList:
            for metricname in self.bbmMetricNames[bbmIdx]:
                uniqueMetrics.add(bbm._dupeMetricName(metricname))
        return uniqueMetrics

    def uniqueMetadata(self):
        """Examine basebinMetrics and return the set of unique metadata values"""
        uniqueMetadata = set()
        bbm = self.bbmDict[0]
        for bbmIdx in self.bbmList:
            for metricname in self.bbmDict[bbmIdx].metadata:
                uniqueMetadata.add(self.bbmDict[bbmIdx].metadata[metricname])
        return uniqueMetadata

    def uniqueSimDataNames(self):
        """Examine baseBinMetrics and return the set of unique simDataNames"""
        uniqueSimDataNames = set()
        bbm = self.bbmDict[0]
        for bbmIdx in self.bbmList:
            for metricname in self.bbmDict[bbmIdx].metadata:
                uniqueSimDataNames.add(self.bbmDict[bbmIdx].simDataName[metricname])
        return uniqueSimDataNames

    def identifyDictNums(self, simDataName=None, metricNames=None, metadata=None):
        """Identify dictionary keys of baseBinMetrics, potentially restricted by
        simDataName/metricName/metadata (which can be lists)."""
        if simDataName != None:
            if not hasattr(simDataName, '__iter__'):
                simDataName = [simDataName, ]
        else:
            simDataName = []
        if metricNames != None:
            if not hasattr(metricNames, '__iter__'):
                metricNames = [metricNames, ]
        else:
            metricNames = []
        if metadata != None:
            if not hasattr(metadata, '__iter__'):
                metadata = [metadata, ]
        else:
            metadata = []
        dictlist = self.bbmList
        for d in dictlist:
            for s in simDataName:
                if s not in self.bbmDict[d].simDataName.values():
                    dictlist.remove(d)
            for m in metadata:
                if m not in self.bbmDict[d].metadata.values():
                    dictlist.remove(d)
            for mname in metricNames:
                if mname not in self.bbmMetricNames[d]:
                    dictlist.remove(d)
                    #for d in dictlist:
                    #           print self.bbmFiles[d], self.bbmMetricNames[d]
        return dictlist
                        
    # Maybe go through and find sets of metrics to compare? 
    # Otherwise, how do we know which ones to compare, in the next step? 
    # Also, have to do some work to make sure that binners for things to compare are compatible
    
    def plotComparisons(self, metricNameList, 
                        histBins=100, histRange=None, maxl=500.,
                        plotTitle=None, legendloc='upper left',
                        savefig=False, outDir=None, outfileRoot=None):
        """Create comparison plots of all metricValues in metricNameList.

        Will create one histogram or binned data plot with all values from metricNameList,
        similarly for power spectra if applicable.
        Will create skymap difference plots if only two metrics: skymap is intersection of 2 areas."""
        # Check if 'metricName' is plottable data.
        for m in metricNameList:
            # Remove if an 'object' type. 
            if self.metricValues[m].dtype == 'object':
                metricNameList.remove(m)
            # Remove if a numpy rec array or anything else longer than float.
            if len(self.metricValues[m].dtype) > 0: 
                metricNameList.remove(m)
        # If there is only one metric remaining, just plot.
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
        # Plot the binned data if applicable. 
        if hasattr(self.binner, 'plotBinnedData'):
            histfignum = None
            addLegend = False
            for i, m in enumerate(metricNameList):
                if i == (len(metricNameList)-1):
                    addLegend = True
                legendLabel = self.simDataName[m] + ' ' + self.metadata[m] \
                    + ' ' + self._dupeMetricName(m)
                histfignum = self.binner.plotBinnedData(self.metricValues[m].filled(0), 
                                                        plotLabel, title=plotTitle, 
                                                        fignum=histfignum,
                                                        alpha=0.3,
                                                        legendLabel=legendLabel, addLegend=addLegend, 
                                                        legendloc=legendloc)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)        
        # Plot the histogram if applicable.
        if hasattr(self.binner, 'plotHistogram'):
            histfignum = None
            addLegend = False
            for i,m in enumerate(metricNameList):
                if i == (len(metricNameList)-1):
                    addLegend = True
                legendLabel = self.simDataName[m] + ' ' + self.metadata[m] \
                  + ' ' + self._dupeMetricName(m)
                histfignum = self.binner.plotHistogram(self.metricValues[m].compressed(),
                                                     metricLabel=plotLabel,
                                                     fignum = histfignum, addLegend=addLegend, 
                                                     legendloc=legendloc,
                                                     bins = histBins, histRange = histRange,
                                                     legendLabel=legendLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)        
        # Plot the power spectrum, if applicable.
        if hasattr(self.binner, 'plotPowerSpectrum'):
            psfignum = None
            addLegend = False
            for i,m in enumerate(metricNameList):
                if i == (len(metricNameList)-1):
                    addLegend = True
                legendLabel = self.simDataName[m] + ' '+  self.metadata[m] \
                  + ' ' + self._dupeMetricName(m)
                psfignum = self.binner.plotPowerSpectrum(self.metricValues[m].filled(),
                                                         addLegend=addLegend,
                                                         fignum = psfignum, maxl = maxl, 
                                                         legendLabel=legendLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the sky map, if only two metricNames.
        if len(metricNameList) == 2:
            # Mask areas where either metric has bad data values, take difference elsewhere.
            mask = self.metricValues[metricNameList[0]].mask
            mask = np.where(self.metricValues[metricNameList[1]].mask == True, True, mask)
            diff = ma.MaskedArray(data = (self.metricValues[metricNameList[0]] - 
                                          self.metricValues[metricNameList[1]]), 
                                          mask=mask,
                                          filled_value = self.binner.badval)            
            # Make color bar label.
            if self._dupeMetricName(metricNameList[0]) == self._dupeMetricName(metricNameList[1]):
                plotLabel = 'Delta ' + self._dupeMetricName(metricNameList[0])
                plotLabel += ' (' + self.metadata[metricNameList[0]] + ' - ' + self.metadata[metricNameList[1]] + ')'
            else:
                plotLabel = metricNameList[0] + ' - ' + metricNameList[1]
            skyfignum = self.binner.plotSkyMap(diff, plotLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(plotTitle, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='sky')
                plt.savefig(outfile, figformat=self.figformat)
