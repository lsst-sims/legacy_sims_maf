# The binMetric class is used for running/generating metric output,
#  and can also be used for generating comparisons or summary statistics on 
#  already calculated metric outputs.
# In either case, there is only one binner per binMetric, 
#   although there may be many metrics.
# 
# An important aspect of the binMetric is handling the metadata about each metric.
#  This includes the opsim run name, the sql constraint on the query that pulled the
#  input data (e.g. 'r band', 'X<1.2', 'WFD prop'), and the binner type that the 
#  metric was run on. The metadata is important for
#  understanding what the metric means, and should be presented in plots & saved in the
#  output files. 
#
# Instantiate the binMetric object by providing a binner object. 
# Then, Metric data can enter the binMetric through either running metrics on simData,
#  or reading metric values from files. 
# To run metrics on simData, 
#  runBins - pass list of metrics, simData, and metadata for the metric run;
#      validates that simData has needed cols, then runs metrics over the binpoints in the binner. 
#      Stores the results in a dictionary keyed by the metric names.
#
# 'readMetric' will read metric data from files. In this case, the metadata 
#   may not be the same for all metrics (e.g. comparing two different opsim runs). 
# To get multiple metric data into the binMetric, in this case run 'readMetric' 
#   multiple times (once for each metric data file) -- the metrics will be added
#   to an internal list, along with their metadata. 
#   Note that all metrics must use the same binner. 
#
# A mixture of readMetric & runBins can also be used to populate the data in the binMetric.
#
# runBins applies to multiple metrics at once; most other methods apply to one metric 
#  at a time but a convenience method to run over all metrics is provided (i.e. reduceAll)
#
# Metric data values, as well as metadata for each metric, are stored in
#  dictionaries keyed by the metric names (a property of the metric). 

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import pyfits as pyf

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class BaseBinMetric(object):
    def __init__(self, figformat='png'):
        """Instantiate binMetric object and set up (empty) dictionaries."""
        # Set up dictionaries to hold metric values, reduced metric values,
        #   simDataName(s) and metadata(s). All dictionary keys should be
        #   metric name (reduced metric data is originalmetricName.reduceFuncName). 
        self.metricValues = {}
        self.simDataName = {}
        self.metadata = {}
        self.comment={}
        # Set figure format for output plot files.
        self.figformat = figformat

    def _buildOutfileName(self, metricName,
                          outDir=None, outfileRoot=None, plotType=None):
        """Builds an automatic output file name for metric data or plots."""
        # Set output directory and root 
        if outDir == None:
            outDir = '.'
        # Start building output file name.
        if outfileRoot == None:
            try:
                # If given appropriate metric name, use simDataName associated with that.
                outfileRoot = self.simDataName[metricName]
            except KeyError:
                # (we weren't given a metricName in the dictionary .. a plot title, for example).
                try:
                    # Then use the simDataName associated with the first metric.
                    outfileRoot = self.simDataName[self.metrics[0].name]
                except AttributeError: 
                    # (the binMetric doesn't have self.metrics set .. perhaps because have just
                    #  read back in the metric values.)
                    outfileRoot = 'comparison'
        # Start building output file name. Strip trailing numerals from metricName.
        oname = outfileRoot + '_' + self._dupeMetricName(metricName)
        # Add summary of the metadata if it exists.
        try:
            self.metadata[metricName]    
            if len(self.metadata[metricName]) > 0:        
                oname = oname + '_' + self.metadata[metricName]#[:5]
        except KeyError:
            pass
        # Add letter to distinguish binner types
        #   (which otherwise might have the same output name).
        oname = oname + '_' + self.binner.binnertype[:2]
        # Add plot name, if plot.
        if plotType:
            oname = oname + '_' + plotType + '.' + self.figformat
        # Build outfile (with path) and strip white spaces (replace with underscores). 
        outfile = os.path.join(outDir, oname.replace(' ', '_'))
        return outfile

    def _deDupeMetricName(self, metricName):
        """In case of multiple metrics having the same 'metricName', add additional characters to de-dupe."""
        mname = metricName
        i =0 
        while mname in self.metricValues.keys():
            mname = metricName + '__' + str(i)
            i += 1
        return mname

    def _dupeMetricName(self, metricName):
        """Remove additional characters added to de-dupe metric name."""
        mname = metricName.split('__')
        if len(mname) > 1:
            return ''.join(mname[:-1])
        else:
            return metricName

    def setBinner(self, binner):
        """Set binner for binMetric."""
        self.binner = binner
    
    def validateMetricData(self, metricList, simData):
        """Validate that simData has the required data values for the metrics in metricList."""
        simCols = metricList[0].classReigstry.uniqueCols()
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n',
                                metricList[0].classRegistry)

    def runBins(self, metricList, simData, 
                simDataName='opsim', metadata=''):
        """Run metric generation over binner.

        metricList = list of metric objects
        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.) """
        # Set metrics (convert to list if not iterable). 
        if hasattr(metricList, '__iter__'):
            self.metrics = metricList
        else:
            self.metrics = [metricList,]
        # Set metadata for each metric.
        for m in self.metrics:
            self.simDataName[m.name] = simDataName
            self.metadata[m.name] = metadata
        # Set up (masked) arrays to store metric data. 
        for m in self.metrics:
            self.metricValues[m.name] = ma.MaskedArray(data = np.empty(len(self.binner), m.metricDtype),
                                                       mask = np.zeros(len(self.binner), 'bool'),
                                                       fill_value=self.binner.badval)
        # Run through all binpoints and calculate metrics 
        #    (slicing the data once per binpoint for all metrics).
        for i, b in enumerate(self.binner):
            idxs = self.binner.sliceSimData(b)
            slicedata = simData[idxs]
            if len(slicedata)==0:
                # No data at this binpoint. Mask data values.
                for m in self.metrics:
                    self.metricValues[m.name].mask[i] = True
            else:
                for m in self.metrics:
                    self.metricValues[m.name].data[i] = m.run(slicedata)
        # Mask data where metrics could not be computed (according to metric bad value).
        for m in self.metrics:
            self.metricValues[m.name] = ma.masked_where(self.metricValues[m.name] == m.badval, 
                                                        self.metricValues[m.name], copy=False)
            # For some reason, the mask fill value is not preserved, so reset.
            self.metricValues[m.name].fill_value = self.binner.badval

    def reduceAll(self, metricList=None):
        """Run all reduce functions on all (complex) metrics.

        Optional: provide a list of metric classes on which to run reduce functions."""
        if metricList == None:
            metricList = self.metrics
        for m in metricList:
            # Check if there are reduce functions to apply.
            try:
                m.reduceFuncs
            except: 
                continue
            # Apply reduce functions
            self.reduceMetric(m.name, m.reduceFuncs.values())            
                
    def reduceMetric(self, metricName, reduceFunc):
        """Run 'reduceFunc' (method on metric object) on metric data 'metricName'. 

        reduceFunc can be a list of functions to be applied to the same metric data."""
        if not isinstance(reduceFunc, list):
            reduceFunc = [reduceFunc,]
        # Set up metric reduced value names.
        rNames = []
        for r in reduceFunc:
            rNames.append(metricName + '_' + r.__name__.lstrip('reduce'))
        # Set up reduced metric values masked arrays, copying metricName's mask.
        for rName in rNames:
            self.metricValues[rName] = ma.MaskedArray(data = np.empty(len(self.binner), 'float'),
                                                      mask = self.metricValues[metricName].mask,
                                                      fill_value=self.binner.badval)
        # Run through binpoints, applying all reduce functions.
        for i, b in enumerate(self.binner):
            if not self.metricValues[metricName].mask[i]:
                # Get (complex) metric values for this binpoint. 
                metricValuesPt = self.metricValues[metricName][i]
                # Evaluate reduced version of metric values.
                for rName, rFunc in zip(rNames, reduceFunc):
                    self.metricValues[rName].data[i] = rFunc(metricValuesPt)
        # Copy simdataName, metadata and comments for this reduced version of the metric data.
        for rName in rNames:
            try:
                self.simDataName[rName] = self.simDataName[metricName]
            except KeyError:
                pass
            try:
                self.metadata[rName] = self.metadata[metricName]
            except KeyError:
                pass
            try:
                self.comment[rName] = self.comment[metricName]
            except KeyError:
                pass

    def writeAll(self, outDir=None, outfileRoot=None, comment=''):
        """Write all metric values to disk."""
        for mk in self.metricValues.keys():
            dt = self.metricValues[mk].dtype
            self.writeMetric(mk, comment=comment, outDir=outDir, outfileRoot=outfileRoot, \
                             dt=dt)

        
    def writeMetric(self, metricName, comment='', outfileRoot=None, outDir=None, 
                    dt='float'):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
        outDir = directory to write output data (default '.').
        dt = data type.
       """
        outfile = self._buildOutfileName(metricName, outDir=outDir, outfileRoot=outfileRoot)
        self.binner.writeMetricData(outfile+'.fits', self.metricValues[metricName],
                                    metricName = metricName,
                                    simDataName = self.simDataName[metricName],
                                    metadata = self.metadata[metricName],
                                    comment = comment, dt=dt, 
                                    badval = self.binner.badval)

        #depreciated, now binner data written with metric
 #   def writeBinner(self,  binnerfile='binner.obj', outfileRoot=None, outDir=None):
 #       """Write a pickle of the binner to disk."""
 #       outfile = self._buildOutfileName(binnerfile, outDir=outDir, outfileRoot=outfileRoot)
 #       modbinner = self.binner
 #       if hasattr(modbinner,'opsimtree'):  delattr(modbinner,'opsimtree') 
 #       pickle.dump(modbinner, open(outfile,'w'))

 #   def readBinner(self, binnerfile='binner.obj'):
 #      self.binner = pickle.load(open(binnerfile, 'r'))
    
    def readMetric(self, filenames, checkBinner=True):
        """Read metric values and binner (pickle object) from disk.

        checkBinner =  check the binnertype and number of binpoints match the 
          properties of self.binner"""
        # Read metrics from disk
        for f in filenames:
            metricValues, binner, header \
              = self.binner.readMetricData(f)
            # Dedupe the metric name, if needed.
            metricName = self._deDupeMetricName(metricName)
            # Store the header values in variables
            self.metricValues[metricName] = metricValues
            self.metricValues[metricName].fill_value = self.binner.badval
            self.simDataName[metricName] = header['simDataName']
            self.metadata[metricName] = header['metadata'.upper()]
            self.comment[metricName] = header['comment'.upper()]
            if checkBinner:
                if binnertype != self.binner.binnertype:
                    raise Exception('Metrics not computed on currently loaded binner type.')           
                if np.size(metricValues) != self.binner.nbins:
                    raise Exception('Metric does not have the same number of points as loaded binner.')

                
    def plotAll(self, outDir='./', savefig=True, closefig=False):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mk in self.metricValues.keys():
            try:
                self.plotMetric(mk, outDir=outDir, savefig=savefig)
                if closefig:
                   plt.close('all')
            except ValueError:
                continue 

    def plotMetric(self, metricName, 
                   savefig=True, outDir=None, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        # Check that metricName refers to plottable ('float') data.
        if not ((self.metricValues[metricName].dtype == 'float') or 
                (self.metricValues[metricName].dtype=='int')):
            raise ValueError('Metric data in %s is not float or int type (%s).' 
                             %(metricName, self.metricValues[metricName].dtype))
        # Build plot title and label.
        mname = self._dupeMetricName(metricName)
        plotTitle = self.simDataName[metricName] + ' ' + self.metadata[metricName]
        plotTitle += ' : ' + mname
        plotLabel = mname
        # Plot the binned metric data, if relevant (oneD binners). 
        if hasattr(self.binner, 'plotBinnedData'):
            histfignum = self.binner.plotBinnedData(self.metricValues[metricName],
                                                    plotLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot,
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the histogram, if relevant. (spatial binners)
        if hasattr(self.binner, 'plotHistogram'):
            histfignum = self.binner.plotHistogram(self.metricValues[metricName].compressed(), 
                                                   plotLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the sky map, if able. (spatial binners)
        if hasattr(self.binner, 'plotSkyMap'):
            skyfignum = self.binner.plotSkyMap(self.metricValues[metricName].filled(self.binner.badval),
                                               plotLabel, title=plotTitle)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='sky')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the angular power spectrum, if able. (healpix binners)
        if hasattr(self.binner, 'plotPowerSpectrum'):
            psfignum = self.binner.plotPowerSpectrum(self.metricValues[metricName].filled(),
                                                     title=plotTitle, 
                                                     legendLabel=plotLabel)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)

    
    def plotComparisons(self, metricNameList, histBins=100, histRange=None, maxl=500.,
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
                    
    
    def computeSummaryStatistics(self, metricName, summaryMetric):
        """Compute single number summary of metric values in metricName, using summaryMetric."""
        # Because of the way the metrics are built, summaryMetric will require a numpy rec array.
        if len(self.metricValues[metricName]) == 1:
            return self.metricValues[metricName]
        else:
            # Create numpy rec array from metric data, with bad values removed. 
            rarr = np.array(zip(self.metricValues[metricName].compressed()), 
                            dtype=[('metricdata', self.metricValues[metricName].dtype)])
            metric = summaryMetric('metricdata')
            return metric.run(rarr)
        
