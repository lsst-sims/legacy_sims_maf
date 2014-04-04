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
        #  The comparison bin metric stores data in dictionaries keyed by (the same) number:
        #     -- the baseBinMetrics (which then hold metric data and the binner),
        #     -- the filename a particular baseBinMetric came from (if read from file)
        self.binmetrics = {}
        self.bbmNames = {}
        
    def readMetricData(self, filename):
        """Read metric data values and binners from a (single) filename

        Reads data one file at a time so that it can return the dict # to the user."""
        dictnum = len(self.bbmDict) 
        bbm = BaseBinMetric(figformat=self.figformat)
        # Read the metric data value and binner into a baseBinMetric.
        bbm.readMetricValues(filename)
        self.binmetrics[dictnum] = bbm
        self.bbmNames[dictnum] = filename
        if self.verbose:
            print '%d (%s) Read metrics %s from %s with binner %s' %(dictnum, self.bbmNames[dictnum],
                                                                    bbm.metricNames,  filename, 
                                                                    bbm.binner.binnertype)
        return dictnum

    def setMetricData(self, bbm, nametag=None):
        """Add a basebinmetric object directly.

        Run metrics in baseBinMetric and pass the whole thing directly here, if it's still in memory.
        Returns dict # to the user."""
        dictnum = len(self.bbmDict)
        self.binmetrics[dictnum] = bbm
        self.bbmNames[dictnum] = nametag    
        if self.verbose:
            print '%d (%s) Added metrics %s from basebinmetric with binner %s' %(dictnum,
                                                                                 self.bbmNames[dictnum],
                                                                                 bbm.metricNames,
                                                                                 bbm.binner.binnertype)
        return dictnum

    def uniqueMetrics(self, dictNums=None):
        """Examine metric names and return the set of unique metric names (optionally, for only dictNums)."""
        uniqueMetrics = set()
        if dictNums is None:
            dictNums = self.binmetrics.keys()
        for d in dictNums:
            for mname in self.binmetrics[d].metricValues:
                uniqueMetrics.add(self.binmetrics[d]._dupeMetricName(mname))
        return uniqueMetrics

    def uniqueMetadata(self, dictNums=None):
        """Examine basebinMetrics and return the set of unique metadata values (optionally, for only dictNums)"""
        uniqueMetadata = set()
        if dictNums is None:
            dictNums = self.binmetrics.keys()
        for d in dictNums:
            for mname in self.binmetrics[d].metadata:
                uniqueMetadata.add(self.binmetrics[d].metadata[mname])
        return uniqueMetadata

    def uniqueSimDataNames(self, dictNums=None):
        """Examine baseBinMetrics and return the set of unique simDataNames (optionally, for only dictNums)"""
        uniqueSimDataNames = set()
        if dictNums is None:
            dictNums = self.binmetrics.keys()
        for d in dictNums:
            for mname in self.binmetrics[d].simDataName:
                uniqueSimDataNames.add(self.binmetrics[d].simDataName[mname])
        return uniqueSimDataNames

    def tagsToDictNum(self, tagname):
        """Return corresponding dictNum for a given filename or 'nametag' of a baseBinMetric."""
        if tagname not in self.bbmNames.values():
            print 'Tag %s not found in set of filenames or nametags (%s)' %(tagname, self.bbmNames)
            return None
        else:
            # There's gotta be a better way...
            for k, v in self.bbmNames.items():
                if v == tagname:
                    return k

    def dictNumToTags(self, dictNum):
        """Return corresponding filename or 'nametag' for a given dictNum."""
        return self.bbmNames.get(dictnum)


    def metricNamesInDictNum(self, dictNum):
        """Return metric names associated with a particular baseBinMetric identified by 'dictNum'."""
        return self.binmetrics[dictNum].metricValues.keys()
                            
    def findDictNums(self, simDataName=None, metricNames=None, metadata=None):
        """Identify dictionary keys of baseBinMetrics, potentially restricted by
        simDataName/metricName/metadata (which can be lists). WORK IN PROGRESS."""
        if simDataName is not None:
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
        for d in self.bbmList:
            print d, self.bbmDict[d].simDataName, s
            for s in simDataName:
                if s not in self.bbmDict[d].simDataName.values():
                    print 'removing %d for s' %(d, s)
                    dictlist.remove(d)
            for m in metadata:
                if m not in self.bbmDict[d].metadata.values():
                    print 'removing %d for m' %(d, m)
                    dictlist.remove(d)
            for mname in metricNames:
                if mname not in self.bbmMetricNames[d]:
                    print 'removing %d, for %s' %(d, mname)
                    dictlist.remove(d)
                    #for d in dictlist:
                    #           print self.bbmFiles[d], self.bbmMetricNames[d]
        return dictlist
                        
    # Maybe go through and find sets of metrics to compare? 
    # Otherwise, how do we know which ones to compare, in the next step? 
    # Also, have to do some work to make sure that binners for things to compare are compatible
    
    def plotComparisons(self, dictNums, metricNames, 
                        histBins=100, histRange=None, maxl=500.,
                        plotTitle=None, legendloc='upper left',
                        savefig=False, outDir=None, outfileRoot=None):
        """Create comparison plots.

        dictNums (a list) identifies which binMetrics to use to create the comparison plots,
        while metricNames identifies which metric data within each binMetric to use.

        Will create one histogram or binned data plot with all values from metricNameList,
        similarly for power spectra if applicable.
        Will create skymap difference plots if only two metrics: skymap is intersection of 2 areas."""
        if len(dictNums) != len(metricNames):
            raise Exception('dictNums must be same length as metricNames list')                                
        # Check if 'metricName' is plottable data.
        for i, d, m in enumerate(zip(dictNums, metricNames)):
            # Remove if an 'object' type. 
            if self.binmetrics[d].metricValues[m].dtype == 'object':
                del dictNums[i]
                del metricNames[i]
            # Remove if a numpy rec array or anything else longer than float.
            if len(self.metricValues[m].dtype) > 0:
                del dictNums[i]
                del metricNames[i]
        # Check that binners are compatible (for plotting)

        
        # If there is only one metric remaining, just plot.
        if len(metricNames) < 2:
            print 'Only one metric left in metricNameList - %s - so defaulting to plotMetric.' \
              %(metricNames)
            self.binmetrics[d].plotMetric(metricNames[0], savefig=savefig,
                                          outDir=outDir, outfileRoot=outfileRoot)
            return
        # Else build plot titles. 
        simDataNames = self.uniqueSimDataNames(dictNums)
        metadatas = self.uniqueMetaData(dictNums)
        metricNames = self.uniqueMetrics(dictNums)
        # Create a plot title from the unique parts of the simData/metadata/metric names.
        if plotTitle is None:
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
