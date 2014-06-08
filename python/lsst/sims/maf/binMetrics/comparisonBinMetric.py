# comparisonBinMetric is intended to generate comparison plots. 
# This basically holds baseBinMetrics, but multiple baseBinMetrics which will be used to 
#  create comparison plots.

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings

import lsst.sims.maf.binners as binners
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
        dictnum = len(self.binmetrics) 
        bbm = BaseBinMetric(figformat=self.figformat)
        # Read the metric data value and binner into a baseBinMetric.
        bbm.readMetricValues(filename)
        self.binmetrics[dictnum] = bbm
        self.bbmNames[dictnum] = filename
        if self.verbose:
            print '%d (name %s) Read metrics %s from %s with binner %s' %(dictnum, self.bbmNames[dictnum],
                                                                        bbm.metricNames,  filename, 
                                                                        bbm.binner.binnerName)
        return dictnum

    def setMetricData(self, bbm, nametag=None):
        """Add a basebinmetric object directly.

        Run metrics in baseBinMetric and pass the whole thing directly here, if it's still in memory.
        Returns dict # to the user."""
        dictnum = len(self.binmetrics)
        self.binmetrics[dictnum] = bbm
        self.bbmNames[dictnum] = nametag    
        if self.verbose:
            print '%d (%s) Added metrics %s from basebinmetric with binner %s' %(dictnum,
                                                                                 self.bbmNames[dictnum],
                                                                                 bbm.metricNames,
                                                                                 bbm.binner.binnerName)
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
                            
    def findDictNums(self, simDataName=None, metricNames=None, metadata=None, binnerName=None):
        """Identify dictionary keys of baseBinMetrics, potentially restricted by
        simDataName/metricName/metadata (which can be lists) and binnerName (single binnerName)."""
        def makeList(name):
            if name is not None:
                if not hasattr(name, '__iter__'):
                    name = [name,]
            else:
                name = []
            return name
        simDataName = makeList(simDataName)
        metricNames = makeList(metricNames)
        metadata = makeList(metadata)
        dictlist = list(self.binmetrics.keys())
        for i,d in enumerate(dictlist):
            for s in simDataName:
                if s not in self.binmetrics[d].simDataName.values():                
                    dictlist[i] = None
            for m in metadata:
                if m not in self.binmetrics[d].metadata.values():
                    dictlist[i] = None
            for mname in metricNames:
                if mname not in self.binmetrics[d].metricValues:
                    dictlist[i] = None
            if binnerName is not None:
                if (self.binmetrics[d].binner.binnerName != binnerName):
                    dictlist[i] = None
        dictlist = [x for x in dictlist if (x is not None)]
        return dictlist
                        
    def _checkPlottable(self, dictNums, metricNames):
        """Given dictNums and metricNames lists, checks if there exists a metric 'metricName'
        in binMetric, checks if the values are plottable and
        returns updated lists of dictNums and metricNames containing only plottable values."""
        for i, (d, m) in enumerate(zip(dictNums, metricNames)):
            ##  print d, m, self.binmetrics[d].metricValues[m].dtype
            # Remove if metric m is not part of binmetric[d]
            if m not in self.binmetrics[d].metricValues:
                dictNums[i] = None
                metricNames[i] = None
            # Remove if an 'object' type. 
            elif self.binmetrics[d].metricValues[m].dtype == 'object':
                dictNums[i] = None
                metricNames[i] = None
            # Remove if a numpy rec array or anything else longer than float.
            elif len(self.binmetrics[d].metricValues[m].dtype) > 0:
                dictNums[i] = None
                metricNames[i] = None
        dictNums = [x for x in dictNums if (x is not None)]
        metricNames = [x for x in metricNames if (x is not None)]
        return dictNums, metricNames

    def _buildPlotTitle(self, dictNums, metricNames):
        """Build a plot title from the simDataName, metadata and metric names."""
        usimDataNames = self.uniqueSimDataNames(dictNums)
        umetadatas = self.uniqueMetadata(dictNums)
        umetricNames = list(set(metricNames))
        # Create a plot title from the unique parts of the simData/metadata/metric names.
        plotTitle = ''
        if len(usimDataNames) == 1:
            plotTitle += ' ' + list(usimDataNames)[0]
        if len(umetadatas) == 1:
            plotTitle += ' ' + list(umetadatas)[0]
        if len(umetricNames) == 1:
            plotTitle += ' ' + list(umetricNames)[0]
        if plotTitle == '':
            # If there were more than one of everything above, join metricNames with commas. 
            plotTitle = ', '.join(umetricNames)
        return plotTitle
    
    def plotHistograms(self, dictNums, metricNames, 
                        bins=100, histMin=None,histMax=None,
                        title=None, xlabel=None,color=None, labels=None,
                        legendloc='upper left', bnamelen=4, alpha=1.0,
                        savefig=False, outDir=None, outfileRoot=None, plotkwargs=None):
        """
        Create a plot containing the histogram visualization from all possible metrics in dictNum +
          metricNames.

        dictNums (a list) identifies which binMetrics to use to create the comparison plots,
        while metricNames (a list) identifies which metric data within each binMetric to use.
        plotkwargs is a list of dicts with plotting parameters that override the defaults.
        """
        ### todo (relevant to all plotting methods)
        #  consider returning plot info to caller via dictionary (may want set of info on metric names,
        #    sqlconstraint, metadata, output filename, etc) .. useful for filling in ResultsSummary.dat
        if len(dictNums) != len(metricNames):
            raise Exception('dictNums must be same length as metricNames list')
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        # Check if the binner has a histogram type visualization (or remove from list).
        for i, d in enumerate(dictNums):
            binner = self.binmetrics[d].binner
            if (not hasattr(binner, 'plotBinnedData')) and (not hasattr(binner, 'plotHistogram')):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) == 0:
            warnings.warn('Removed all dictNums and metricNames from list, due to binnerName, metricname absence or type of metric data.')
            return
        if title is None:
            title = self._buildPlotTitle(dictNums, metricNames)
        # Create a plot x-axis label (metricLabel)
        if xlabel is None:
            if self.binmetrics[d].binner.binnerName == 'OneDBinner':
                tmpMnames = [''.join(m.split()[1:]) for m in metricNames]
                xlabel = ', '.join(list(set(tmpMnames)))
            else:
                xlabel = ', '.join(list(set(metricNames)))
        # Plot the data.
        fignum = None
        addLegend = False
        for i, (d, m) in enumerate(zip(dictNums, metricNames)):
            # If we're at the end of the list, add the legend.
            if i == len(metricNames) - 1:
                addLegend = True
            # Build legend label for this dictNum/metricName.
            if labels is None:
               label = (self.binmetrics[d].simDataName[m] + ' ' + self.binmetrics[d].metadata[m] + ' ' 
                              + self.binmetrics[d]._dupeMetricName(m) +
                              ' ' + self.binmetrics[d].binner.binnerName[:bnamelen])
            # Plot data using 'plotBinnedData' if that method available (oneDBinner)
            if hasattr(self.binmetrics[d].binner, 'plotBinnedData'):
                plotParams = {'xlabel':xlabel, 'title':title,
                              'alpha':alpha, 'label':label, 'addLegend':addLegend,'legendloc':legendloc,
                              'color':color}
                if plotkwargs is not None:
                   for key in plotkwargs[i].keys():
                      plotParams[key] = plotkwargs[i][key] 
                fignum = self.binmetrics[d].binner.plotBinnedData(self.binmetrics[d].metricValues[m],
                                                                 fignum=fignum, **plotParams)
            # Plot data using 'plotHistogram' if that method available (any spatial binner)
            if hasattr(self.binmetrics[d].binner, 'plotHistogram'):
                plotParams = {'xlabel':xlabel, 'histMin':histMin, 'histMax':histMax, 
                              'bins':bins, 'title':title, 'label':label, 
                              'addLegend':addLegend, 'legendloc':legendloc, 'color':color}
                if plotkwargs is not None:
                   for key in plotkwargs[i].keys():
                      plotParams[key] = plotkwargs[i][key]
                fignum = self.binmetrics[d].binner.plotHistogram(self.binmetrics[d].metricValues[m],
                                                                 fignum=fignum, **plotParams)
        if savefig:
            outfile = self.binmetrics[d]._buildOutfileName(title,
                                                          outDir=outDir, outfileRoot=outfileRoot,
                                                          plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)
        else:
            outfile = None
        return fignum, title, outfile

    def plotPowerSpectra(self, dictNums, metricNames, maxl=500., removeDipole=True,
                         title=None, legendloc='upper left', bnamelen=4,
                         savefig=False, outDir=None, outfileRoot=None):
        """Create a plot containing the power spectrum visualization from all possible metrics in dictNum +
                       metricNames.

        dictNums (a list) identifies which binMetrics to use to create the comparison plots,
        while metricNames identifies which metric data within each binMetric to use."""
        if len(dictNums) != len(metricNames):
            raise Exception('dictNums must be same length as metricNames list')                                
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        # Check if the binner has a histogram type visualization (or remove from list).
        for i, d in enumerate(dictNums):
            binner = self.binmetrics[d].binner
            if (not hasattr(binner, 'plotPowerSpectrum')):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) == 0:
            warnings.warn('Removed all dictNums and metricNames from list, due to binnerName, metricname absence or type of metric data.')
            return
        if plotTitle is None:
            plotTitle = self._buildPlotTitle(dictNums, metricNames)
        # Plot the data.
        fignum = None
        addLegend = False
        for i, (d, m) in enumerate(zip(dictNums, metricNames)):
            # If we're at the end of the list, add the legend.
            if i == len(metricNames) - 1:
                addLegend = True
            # Build legend label for this dictNum/metricName.
            label = (self.binmetrics[d].simDataName[m] + ' ' + self.binmetrics[d].metadata[m] + ' ' 
                           + self.binmetrics[d]._dupeMetricName(m) +
                           ' ' + self.binmetrics[d].binner.binnerName[:bnamelen])    
            # Plot data.
            fignum = self.binmetrics[d].binner.plotPowerSpectrum(self.binmetrics[d].metricValues[m],
                                                                maxl=maxl, removeDipole=removeDipole,
                                                                title=title,
                                                                fignum=fignum,
                                                                label=label,
                                                                addLegend=addLegend)
        if savefig:
            outfile = self.binmetrics[d]._buildOutfileName(title,
                                                          outDir=outDir, outfileRoot=outfileRoot,
                                                          plotType='hist')
            plt.savefig(outfile, figformat=self.figformat)
        else:
            outfile = None
        return fignum, title, outfile
    

    def plotSkyMaps(self, dictNums, metricNames, units=None, title=None,
                    clims=None, cmap=None, cbarFormat='%.2g', 
                    savefig=False, outDir=None, outfileRoot=None):
        """Create a skymap plot of the difference between two dictNum/metricNames.

        dictNums (a list) identifies which binMetrics to use to create the comparison plots,
        while metricNames identifies which metric data within each binMetric to use."""
        if (len(dictNums) != 2) & (len(metricName) != 2):
            raise Exception('Pass only two values for dictNums/metricNames to create skymap difference.')
        if (self.binmetrics[dictNums[0]].binner) != (self.binmetrics[dictNums[1]].binner):
            raise Exception('Binners must be equal')
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        for i, d in enumerate(dictNums):
            binner = self.binmetrics[d].binner
            if not hasattr(binner, 'plotSkyMap'):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) != 2:
            warnings.warn('Removed one or more of the dictNums/metricNames due to metric absence, binner type or metric data type.')
        if plotTitle is None:
            plotTitle = self._buildPlotTitle(dictNums, metricNames)
        # Plot the data.
        fignum = None
        addLegend = False
        # Mask areas where either metric has bad data values, take difference elsewhere.
        mask = self.binmetrics[dictNums[0]].metricValues[metricNames[0]].mask
        mask = np.where(self.binmetrics[dictNums[1]].metricValues[metricNames[1]].mask == True, True, mask)
        diff = ma.MaskedArray(data = (self.binmetrics[dictNums[0]].metricValues[metricNames[0]] -
                                      self.binmetrics[dictNums[1]].metricValues[metricNames[1]]),
                                      mask=mask,
                                      filled_value = self.binmetrics[dictNum[0]].binner.badval)            
        # Make color bar label.
        if units is None:
            mname0 = self.binmetrics[dictNum[0]]._dupeMetricName(metricNames[0])
            mname1 = self.binmetrics[dictNum[1]]._dupeMetricName(metricNames[1])
            if (mname0 == mname1):
                units = (mname0 + ' (' + self.binmetrics[dictNum[0]].metadata[metricNames[0]]
                            + ' - ' + self.binmetrics[dictNum[1]].metadata[metricNames[1]])                
            else:
                units = mname0 + ' - ' + mname1
        # Plot data.
        fignum = self.binmetric[dictNums[0]].binner.plotSkyMap(diff, units=units, title=title,
                                                               clims=clims, cmap=cmap, cbarFormat=cbarFormat)
        if savefig:
            outfile = self.binmetric[dictNums[0]]._buildOutfileName(title, 
                                                                    outDir=outDir, outfileRoot=outfileRoot, 
                                                                    plotType='sky')
            plt.savefig(outfile, figformat=self.figformat)
        else:
            outfile = None
        return fignum, title, outfile
