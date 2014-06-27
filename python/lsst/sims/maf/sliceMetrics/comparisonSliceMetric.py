# comparisonSliceMetric is intended to generate comparison plots. 
# This basically holds baseSliceMetrics, but multiple baseSliceMetrics which will be used to 
#  create comparison plots.

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings

import lsst.sims.maf.slicers as slicers
from .baseSliceMetric import BaseSliceMetric


import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

class ComparisonSliceMetric(object):
    """ComparisonSliceMetric"""
    def __init__(self, figformat='pdf', dpi=None, verbose=True):
        self.figformat = figformat
        self.dpi = dpi
        self.verbose = verbose
        #  The comparison bin metric stores data in dictionaries keyed by (the same) number:
        #     -- the baseSliceMetrics (which then hold metric data and the slicer),
        #     -- the filename a particular baseSliceMetric came from (if read from file)
        self.slicemetrics = {}
        self.bbmNames = {}
        
    def readMetricData(self, filename):
        """Read metric data values and slicers from a (single) filename

        Reads data one file at a time so that it can return the dict # to the user."""
        dictnum = len(self.slicemetrics) 
        bbm = BaseSliceMetric(figformat=self.figformat)
        # Read the metric data value and slicer into a baseSliceMetric.
        bbm.readMetricValues(filename)
        self.slicemetrics[dictnum] = bbm
        self.bbmNames[dictnum] = filename
        if self.verbose:
            print '%d (name %s) Read metrics %s from %s with slicer %s' %(dictnum, self.bbmNames[dictnum],
                                                                        bbm.metricNames,  filename, 
                                                                        bbm.slicer.slicerName)
        return dictnum

    def setMetricData(self, bbm, nametag=None):
        """Add a baseslicemetric object directly.

        Run metrics in baseSliceMetric and pass the whole thing directly here, if it's still in memory.
        Returns dict # to the user."""
        dictnum = len(self.slicemetrics)
        self.slicemetrics[dictnum] = bbm
        self.bbmNames[dictnum] = nametag    
        if self.verbose:
            print '%d (%s) Added metrics %s from baseslicemetric with slicer %s' %(dictnum,
                                                                                 self.bbmNames[dictnum],
                                                                                 bbm.metricNames,
                                                                                 bbm.slicer.slicerName)
        return dictnum

    def uniqueMetrics(self, dictNums=None):
        """Examine metric names and return the set of unique metric names (optionally, for only dictNums)."""
        uniqueMetrics = set()
        if dictNums is None:
            dictNums = self.slicemetrics.keys()
        for d in dictNums:
            for mname in self.slicemetrics[d].metricValues:
                uniqueMetrics.add(self.slicemetrics[d]._dupeMetricName(mname))
        return uniqueMetrics

    def uniqueMetadata(self, dictNums=None):
        """Examine baseslicemetrics and return the set of unique metadata values (optionally, for only dictNums)"""
        uniqueMetadata = set()
        if dictNums is None:
            dictNums = self.slicemetrics.keys()
        for d in dictNums:
            for mname in self.slicemetrics[d].metadata:
                uniqueMetadata.add(self.slicemetrics[d].metadata[mname])
        return uniqueMetadata

    def uniqueSimDataNames(self, dictNums=None):
        """Examine baseSliceMetrics and return the set of unique simDataNames (optionally, for only dictNums)"""
        uniqueSimDataNames = set()
        if dictNums is None:
            dictNums = self.slicemetrics.keys()
        for d in dictNums:
            for mname in self.slicemetrics[d].simDataName:
                uniqueSimDataNames.add(self.slicemetrics[d].simDataName[mname])
        return uniqueSimDataNames

    def tagsToDictNum(self, tagname):
        """Return corresponding dictNum for a given filename or 'nametag' of a baseSliceMetric."""
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
        """Return metric names associated with a particular baseSliceMetric identified by 'dictNum'."""
        return self.slicemetrics[dictNum].metricValues.keys()
                            
    def findDictNums(self, simDataName=None, metricNames=None, metadata=None, slicerName=None):
        """Identify dictionary keys of baseSliceMetrics, potentially restricted by
        simDataName/metricName/metadata (which can be lists) and slicerName (single slicerName)."""
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
        dictlist = list(self.slicemetrics.keys())
        for i,d in enumerate(dictlist):
            for s in simDataName:
                if s not in self.slicemetrics[d].simDataName.values():                
                    dictlist[i] = None
            for m in metadata:
                if m not in self.slicemetrics[d].metadata.values():
                    dictlist[i] = None
            for mname in metricNames:
                if mname not in self.slicemetrics[d].metricValues:
                    dictlist[i] = None
            if slicerName is not None:
                if (self.slicemetrics[d].slicer.slicerName != slicerName):
                    dictlist[i] = None
        dictlist = [x for x in dictlist if (x is not None)]
        return dictlist
                        
    def _checkPlottable(self, dictNums, metricNames):
        """Given dictNums and metricNames lists, checks if there exists a metric 'metricName'
        in slicemetric, checks if the values are plottable and
        returns updated lists of dictNums and metricNames containing only plottable values."""
        for i, (d, m) in enumerate(zip(dictNums, metricNames)):
            ##  print d, m, self.slicemetrics[d].metricValues[m].dtype
            # Remove if metric m is not part of slicemetric[d]
            if m not in self.slicemetrics[d].metricValues:
                dictNums[i] = None
                metricNames[i] = None
            # Remove if an 'object' type. 
            elif self.slicemetrics[d].metricValues[m].dtype == 'object':
                dictNums[i] = None
                metricNames[i] = None
            # Remove if a numpy rec array or anything else longer than float.
            elif len(self.slicemetrics[d].metricValues[m].dtype) > 0:
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

        dictNums (a list) identifies which slicemetrics to use to create the comparison plots,
        while metricNames (a list) identifies which metric data within each slicemetric to use.
        plotkwargs is a list of dicts with plotting parameters that override the defaults.
        """
        ### todo (relevant to all plotting methods)
        #  consider returning plot info to caller via dictionary (may want set of info on metric names,
        #    sqlconstraint, metadata, output filename, etc) .. useful for filling in ResultsSummary.dat
        if len(dictNums) != len(metricNames):
            raise Exception('dictNums must be same length as metricNames list')
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        # Check if the slicer has a histogram type visualization (or remove from list).
        for i, d in enumerate(dictNums):
            slicer = self.slicemetrics[d].slicer
            if (not hasattr(slicer, 'plotBinnedData')) and (not hasattr(slicer, 'plotHistogram')):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) == 0:
            warnings.warn('Removed all dictNums and metricNames from list, due to slicerName, metricname absence or type of metric data.')
            return
        if title is None:
            title = self._buildPlotTitle(dictNums, metricNames)
        # Create a plot x-axis label (metricLabel)
        if xlabel is None:
            if self.slicemetrics[d].slicer.slicerName == 'OneDSlicer':
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
               label = (self.slicemetrics[d].simDataName[m] + ' ' + self.slicemetrics[d].metadata[m] + ' ' 
                              + self.slicemetrics[d]._dupeMetricName(m) +
                              ' ' + self.slicemetrics[d].slicer.slicerName[:bnamelen])
            # Plot data using 'plotBinnedData' if that method available (oneDSlicer)
            if hasattr(self.slicemetrics[d].slicer, 'plotBinnedData'):
                plotParams = {'xlabel':xlabel, 'title':title,
                              'alpha':alpha, 'label':label, 'addLegend':addLegend,'legendloc':legendloc,
                              'color':color}
                if plotkwargs is not None:
                   for key in plotkwargs[i].keys():
                      plotParams[key] = plotkwargs[i][key] 
                fignum = self.slicemetrics[d].slicer.plotBinnedData(self.slicemetrics[d].metricValues[m],
                                                                 fignum=fignum, **plotParams)
            # Plot data using 'plotHistogram' if that method available (any spatial slicer)
            if hasattr(self.slicemetrics[d].slicer, 'plotHistogram'):
                plotParams = {'xlabel':xlabel, 'histMin':histMin, 'histMax':histMax, 
                              'bins':bins, 'title':title, 'label':label, 
                              'addLegend':addLegend, 'legendloc':legendloc, 'color':color}
                if plotkwargs is not None:
                   for key in plotkwargs[i].keys():
                      plotParams[key] = plotkwargs[i][key]
                fignum = self.slicemetrics[d].slicer.plotHistogram(self.slicemetrics[d].metricValues[m],
                                                                 fignum=fignum, **plotParams)
        if savefig:
            outfile = self.slicemetrics[d]._buildOutfileName(title,
                                                          outDir=outDir, outfileRoot=outfileRoot,
                                                          plotType='hist')
            plt.savefig(outfile, figformat=self.figformat, dpi=self.dpi)
        else:
            outfile = None
        return fignum, title, outfile

    def plotPowerSpectra(self, dictNums, metricNames, maxl=500., removeDipole=True,
                         title=None, legendloc='upper left', bnamelen=4,
                         savefig=False, outDir=None, outfileRoot=None):
        """Create a plot containing the power spectrum visualization from all possible metrics in dictNum +
                       metricNames.

        dictNums (a list) identifies which slicemetrics to use to create the comparison plots,
        while metricNames identifies which metric data within each slicemetric to use."""
        if len(dictNums) != len(metricNames):
            raise Exception('dictNums must be same length as metricNames list')                                
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        # Check if the slicer has a histogram type visualization (or remove from list).
        for i, d in enumerate(dictNums):
            slicer = self.slicemetrics[d].slicer
            if (not hasattr(slicer, 'plotPowerSpectrum')):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) == 0:
            warnings.warn('Removed all dictNums and metricNames from list, due to slicerName, metricname absence or type of metric data.')
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
            label = (self.slicemetrics[d].simDataName[m] + ' ' + self.slicemetrics[d].metadata[m] + ' ' 
                           + self.slicemetrics[d]._dupeMetricName(m) +
                           ' ' + self.slicemetrics[d].slicer.slicerName[:bnamelen])    
            # Plot data.
            fignum = self.slicemetrics[d].slicer.plotPowerSpectrum(self.slicemetrics[d].metricValues[m],
                                                                maxl=maxl, removeDipole=removeDipole,
                                                                title=title,
                                                                fignum=fignum,
                                                                label=label,
                                                                addLegend=addLegend)
        if savefig:
            outfile = self.slicemetrics[d]._buildOutfileName(title,
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

        dictNums (a list) identifies which slicemetrics to use to create the comparison plots,
        while metricNames identifies which metric data within each slicemetric to use."""
        if (len(dictNums) != 2) & (len(metricName) != 2):
            raise Exception('Pass only two values for dictNums/metricNames to create skymap difference.')
        if (self.slicemetrics[dictNums[0]].slicer) != (self.slicemetrics[dictNums[1]].slicer):
            raise Exception('Slicers must be equal')
        dictNums, metricNames = self._checkPlottable(dictNums, metricNames)
        for i, d in enumerate(dictNums):
            slicer = self.slicemetrics[d].slicer
            if not hasattr(slicer, 'plotSkyMap'):
                del dictNums[i]
                del metricNames[i]
        if len(dictNums) != 2:
            warnings.warn('Removed one or more of the dictNums/metricNames due to metric absence, slicer type or metric data type.')
        if plotTitle is None:
            plotTitle = self._buildPlotTitle(dictNums, metricNames)
        # Plot the data.
        fignum = None
        addLegend = False
        # Mask areas where either metric has bad data values, take difference elsewhere.
        mask = self.slicemetrics[dictNums[0]].metricValues[metricNames[0]].mask
        mask = np.where(self.slicemetrics[dictNums[1]].metricValues[metricNames[1]].mask == True, True, mask)
        diff = ma.MaskedArray(data = (self.slicemetrics[dictNums[0]].metricValues[metricNames[0]] -
                                      self.slicemetrics[dictNums[1]].metricValues[metricNames[1]]),
                                      mask=mask,
                                      filled_value = self.slicemetrics[dictNum[0]].slicer.badval)            
        # Make color bar label.
        if units is None:
            mname0 = self.slicemetrics[dictNum[0]]._dupeMetricName(metricNames[0])
            mname1 = self.slicemetrics[dictNum[1]]._dupeMetricName(metricNames[1])
            if (mname0 == mname1):
                units = (mname0 + ' (' + self.slicemetrics[dictNum[0]].metadata[metricNames[0]]
                            + ' - ' + self.slicemetrics[dictNum[1]].metadata[metricNames[1]])                
            else:
                units = mname0 + ' - ' + mname1
        # Plot data.
        fignum = self.slicemetric[dictNums[0]].slicer.plotSkyMap(diff, units=units, title=title,
                                                               clims=clims, cmap=cmap, cbarFormat=cbarFormat)
        if savefig:
            outfile = self.slicemetric[dictNums[0]]._buildOutfileName(title, 
                                                                    outDir=outDir, outfileRoot=outfileRoot, 
                                                                    plotType='sky')
            plt.savefig(outfile, figformat=self.figformat)
        else:
            outfile = None
        return fignum, title, outfile
