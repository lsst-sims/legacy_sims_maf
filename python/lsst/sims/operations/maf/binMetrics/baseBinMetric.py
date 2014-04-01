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
#  dictionaries keyed by the metric name.

##   todo ... swap metric name for a running index? but then how to indicate which value we mean
##   (have to track number externally, or specify which dictionary, as metricName could be duplicated)

import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf
import lsst.sims.operations.maf.binners as binners
from lsst.sims.operations.maf.utils.percentileClip import percentileClip


import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

binnertypeDict = {'UNI': binners.UniBinner,
                  'ONED': binners.OneDBinner,
                  'ND': binners.NDBinner,
                  'OPSIMFIELDS': binners.OpsimFieldBinner, 
                  'HEALPIX': binners.HealpixBinner}


class BaseBinMetric(object):
    def __init__(self, figformat='png'):
        """Instantiate binMetric object and set up (empty) dictionaries."""
        # Set figure format for output plot files.
        self.figformat = figformat
        self.metricNames = []
        self.metricObjs = {}
        self.plotParams = {}
        self.metricValues = {}
        self.simDataName = {}
        self.metadata = {}
        self.comment={}
        self.binner = None

    def _buildOutfileName(self, metricName,
                          outDir=None, outfileRoot=None, plotType=None):
        """Builds an automatic output file name for metric data or plots."""
        # Set output directory and root 
        if outDir == None:
            outDir = '.'
        # Start building output file name.
        if outfileRoot == None:
            if metricName in self.simDataName:
                # If metricName is the name of an actual metric, use its associated simdata ID.
                outfileRoot = self.simDataName[metricName]
            else:
                # metricName may have been a plot title, so try to find a good compromise.
                outfileRoot = list(set(self.simDataName.values()))
                if len(outfileRoot) > 1:
                    outfileRoot = 'comparison'
                else:
                    outfileRoot = outfileRoot[0]
        # Start building output file name. Strip trailing numerals from metricName.
        oname = outfileRoot + '_' + self._dupeMetricName(metricName)
        # Add summary of the metadata if it exists.
        if metricName in self.metadata:
            metadata_summary = self.metadata[metricName]
            if len(metadata_summary) > 0:        
                oname = oname + '_' + metadata_summary
        # Add letter to distinguish binner types
        #   (which otherwise might have the same output name).
        oname = oname + '_' + self.binner.binnertype[:3]
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

    def setBinner(self, binner, override=False):
        """Set binner for binMetric.

        If override = False (default), checks if binner already set, and if the two are equal."""
        if (self.binner == None) or (override):
            self.binner = binner
            return True        
        return (self.binner == binner)            

    def setMetrics(self, metricList):
        """Sets dictionaries for the metric objects and their plotting parameters."""
        # Keeping track of metric data values, plotting parameters, and metadata must
        # be done without depending on having the metric objects themselves, as the binMetric
        # may be populated with data by reading values from disk instead of calculating them.
        # All dictionaries are keyed by metric name
        #   (reduced metric data is originalmetricName.reduceFuncName). 
        if not hasattr(metricList, '__iter__'):
            metricList = [metricList,]
        for m in metricList:
            self.metricNames.append(self._deDupeMetricName(m.name))
        for m, mname in zip(metricList, self.metricNames):
            self.metricObjs[mname] = m
            self.plotParams[mname] = m.plotParams

    def readMetricValues(self, filenames, verbose=False):
        """Given a list of filenames, reads metric values and metadata from disk. """
        if not hasattr(filenames, '__iter__'):
            filenames = [filenames, ]        
        for f in filenames:
            if self.binner == None:            
                hdulist = pyf.open(f)
                header = hdulist[0].header
                if (header['NAXIS'] == 0):
                    header = hdulist[1].header
                binnertype = header['binnertype']
                hdulist.close()
                #  Instantiate a binner of the right type, and use its native read methods.
                if binnertype == 'HEALPIX':
                    self.binner = binnertypeDict[binnertype](nside=header['NSIDE'], 
                                                             verbose=verbose)
                else:
                    self.binner = binnertypeDict[binnertype]()
            metricValues, binner, header = self.binner.readMetricData(f, verbose=verbose)
            # Check that the binner from this file matches self.binner
            if not(self.setBinner(binner, override=False)):
                raise Exception('Binner for metric %s does not match existing binner.' 
                                % (header['metricName']))
            # Dedupe the metric name, if needed.
            metricName = self._deDupeMetricName(header['metricName'])
            self.metricNames.append(metricName)
            self.metricValues[metricName] = metricValues
            self.metricValues[metricName].fill_value = self.binner.badval
            self.simDataName[metricName] = header['simDataName']
            self.metadata[metricName] = header['metadata'.upper()]
            self.comment[metricName] = header['comment'.upper()]
            self.plotParams[metricName] = {}
            if 'plotParams' in header:
                for pp in header['plotParams']:
                    self.plotParams[metricName][pp] = header['plotParams'][pp]

    
    def validateMetricData(self, simData):
        """Validate that simData has the required data values for the metrics in self.metricObjs."""
        simCols = self.metricObjs[self.metricNames[0]].classRegistry.uniqueCols()
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n',
                                metricList[0].classRegistry)

    def runBins(self, simData, simDataName='opsim', metadata=''):
        """Run metric generation over binner, for metric objects in self.metricObjs.

        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.) """
        # Set metadata for each metric.
        for mname in self.metricObjs:
            self.simDataName[mname] = simDataName
            self.metadata[mname] = metadata
        # Set up (masked) arrays to store metric data. 
        for mname in self.metricObjs:
            self.metricValues[mname] = ma.MaskedArray(data = np.empty(len(self.binner), 
                                                      self.metricObjs[mname].metricDtype),
                                                      mask = np.zeros(len(self.binner), 'bool'),
                                                      fill_value=self.binner.badval)
        # Run through all binpoints and calculate metrics 
        #    (slicing the data once per binpoint for all metrics).
        for i, binpoint in enumerate(self.binner):
            idxs = self.binner.sliceSimData(binpoint)
            slicedata = simData[idxs]
            if len(slicedata)==0:
                # No data at this binpoint. Mask data values.
                for mname in self.metricObjs:
                    self.metricValues[mname].mask[i] = True
            else:
                for mname in self.metricObjs:
                    if hasattr(self.metricObjs[mname], 'needRADec'):
                        if self.metricObjs[mname].needRADec:
                            self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata, binpoint[1], binpoint[2])
                        else:
                            self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata)
                    else:
                        self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata)
        # Mask data where metrics could not be computed (according to metric bad value).
        for mname in self.metricObjs:
            self.metricValues[mname] = ma.masked_where(self.metricValues[mname] == 
                                                       self.metricObjs[mname].badval, 
                                                       self.metricValues[mname], copy=False)
            # For some reason, the mask fill value is not preserved, so reset.
            self.metricValues[mname].fill_value = self.binner.badval

    def reduceAll(self):
        """Run all reduce functions on all (complex) metrics."""
        for mname in self.metricObjs:
            # Check if there are reduce functions to apply.
            try:
                self.metricObjs[mname].reduceFuncs
            except Exception as e: 
                continue
            # Apply reduce functions
            self.reduceMetric(mname, self.metricObjs[mname].reduceFuncs.values())            
                
    def reduceMetric(self, metricName, reduceFunc):
        """Run 'reduceFunc' (method on metric object) on metric data 'metricName'. 

        reduceFunc can be a list of functions to be applied to the same metric data."""
        if not isinstance(reduceFunc, list):
            reduceFunc = [reduceFunc,]
        # Set up metric reduced value names.
        rNames = []
        for r in reduceFunc:
            rNames.append(metricName + '_' + r.__name__.replace('reduce',''))
        # Set up reduced metric values masked arrays, copying metricName's mask.
        for rName in rNames:
            self.metricValues[rName] = ma.MaskedArray(data = np.empty(len(self.binner), 'float'),
                                                      mask = self.metricValues[metricName].mask,
                                                      fill_value=self.binner.badval)
        # Run through binpoints, applying all reduce functions.
        for i, b in enumerate(self.binner):
            if np.array(self.metricValues[metricName].mask).size == 1:
                maskval = self.metricValues[metricName].mask
            else:
                maskval = self.metricValues[metricName].mask[i]
            if not maskval:
                # Get (complex) metric values for this binpoint. 
                metricValuesPt = self.metricValues[metricName][i]
                # Evaluate reduced version of metric values.
                for rName, rFunc in zip(rNames, reduceFunc):
                    self.metricValues[rName].data[i] = rFunc(metricValuesPt)
        # Copy simdataName, metadata and comments for this reduced version of the metric data.
        for rName in rNames:
            if metricName in self.simDataName:
                self.simDataName[rName] = self.simDataName[metricName]
            if metricName in self.metadata:
                self.metadata[rName] = self.metadata[metricName]
            if metricName in self.comment:
                self.comment[rName] = self.comment[metricName]
            if metricName in self.plotParams:
                self.plotParams[rName] = self.plotParams[metricName]

    def writeAll(self, outDir=None, outfileRoot=None, comment=''):
        """Write all metric values to disk."""
        for mname in self.metricValues:
            dt = self.metricValues[mname].dtype
            self.writeMetric(mname, comment=comment, outDir=outDir, outfileRoot=outfileRoot, \
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
                                    metricName = self._dupeMetricName(metricName),
                                    simDataName = self.simDataName[metricName],
                                    metadata = self.metadata[metricName],
                                    comment = comment, dt=dt, 
                                    badval = self.binner.badval)
        # Update this to write self.comment and self.plotParams

                
    def plotAll(self, outDir='./', savefig=True, closefig=False, outfileRoot=None, verbose=False):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mname in self.metricValues:
            if verbose:
                print 'Plotting %s' %(mname)
            try:
                self.plotMetric(mname, outDir=outDir, savefig=savefig, outfileRoot=outfileRoot)
                if closefig:
                   plt.close('all')
            except ValueError:
                continue 

    def plotMetric(self, metricName, 
                   savefig=True, outDir=None, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        # Check that metricName refers to plottable ('float') data.
        if not ((self.metricValues[metricName].dtype == 'float') or 
                (self.metricValues[metricName].dtype=='int') or (metricName == 'hourglass')):
            raise ValueError('Metric data in %s is not float or int type (%s).' 
                             %(metricName, self.metricValues[metricName].dtype))
        if metricName in self.plotParams:
            pParams = self.plotParams[metricName]
        else:
            pParams = None
        # Build plot title and label.
        mname = self._dupeMetricName(metricName)
        if 'title' in pParams:
            title = pParams['title']
        else:
            title = self.simDataName[metricName] + ' ' + self.metadata[metricName]
            title += ' : ' + mname
        if 'xlabel' in pParams:
            xlabel = pParams['xlabel']
        elif '_unit' in pParams:
            xlabel = mname+'('+pParams['_unit']+')'
        else:
            xlabel = mname
        if 'ylabel' in pParams:
            ylabel=pParams['ylabel']
        else:
            ylabel=None
        if 'units' in pParams:
            units = pParams['units']
        elif '_unit' in pParams:
            units = mname+'('+pParams['_unit']+')'
        else:
            units = mname
        if 'legendLabel' in pParams:
            legendLabel =  pParams['legendLabel']
        else:
            legendLabel = None
        if 'cmap' in pParams:
            cmap = getattr(cm, pParams['cmap'])
        else:
            cmap = None
        if 'cbarFormat' in pParams:
            cbarFormat = pParams['cbarFormat']
        else:
            cbarFormat = None
        # Set up for plot limits (used directly for clims for skyMaps, indirectly for histRange).
        plotMin = self.metricValues[metricName].compressed().min()
        plotMax = self.metricValues[metricName].compressed().max()
        # If percentile clipping is set, use it. 
        if 'percentileClip' in pParams:
            plotMin, plotMax = percentileClip(self.metricValues[metricName].compressed(),
                                              percentile=pParams['percentileClip'])
        # Use plot limits if they're set (min/max overrides percentile clipping).
        if 'plotMin' in pParams:
            plotMin = pParams['plotMin']
        if 'plotMax' in pParams:
            plotMax = pParams['plotMax']
        # Set 'histRange' parameter from pParams, if available.
        if 'histMax' in pParams:
            histRange = [pParams['histMin'], pParams['histMax']]
        #else:
        #    histRange = None
        else: # Otherwise use data from plotMin/Max or percentileClipping, if those were set.
            histRange = [plotMin, plotMax]
        # Determine if should data using log scale, using pParams if available
        if 'ylog' in pParams:
            ylog = pParams['ylog']
        else: # or if data spans > 3 decades if not.
            ylog = False
            if metricName != 'hourglass':
                if (np.log10(self.metricValues[metricName].max() -
                             self.metricValues[metricName].min()) > 3):
                    ylog = True
                    if self.metricValues[metricName].max() <= 0:
                        ylog = False
        # Okay, now that's all set .. go plot some data! 
        if hasattr(self.binner, 'plotBinnedData'):
            histfignum = self.binner.plotBinnedData(self.metricValues[metricName],
                                                    xlabel=xlabel, title=title, 
                                                    histRange=histRange, ylog=ylog,
                                                    legendLabel=legendLabel)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot,
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the histogram, if relevant. (spatial binners)
        if hasattr(self.binner, 'plotHistogram'):
            histfignum = self.binner.plotHistogram(self.metricValues[metricName], 
                                                   xlabel=xlabel, title=title, 
                                                   histRange=histRange, ylog=ylog)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='hist')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the sky map, if able. (spatial binners)
        if hasattr(self.binner, 'plotSkyMap'):
            # Make sure the color map leaves background white in healpy plots.  Need to debug this some more I think...
            if cmap:
                cmap0 = cmap
                cmap0.set_under('w')
                cmap0.set_bad('gray')
            if 'zp' in pParams: # Subtract off a zeropoint
                skyfignum = self.binner.plotSkyMap((self.metricValues[metricName] - pParams['zp']),
                                                   cmap=cmap, cbarFormat=cbarFormat,
                                                   units=units, title=title,
                                                   clims=[plotMin-pParams['zp'],
                                                          plotMax-pParams['zp']], ylog=ylog)
            elif 'normVal' in pParams: # Normalize by some value
                skyfignum = self.binner.plotSkyMap((self.metricValues[metricName]/pParams['normVal']),
                                                   cmap=cmap, cbarFormat=cbarFormat,
                                                   units=units, title=title,
                                                   clims=[plotMin/pParams['normVal'],
                                                          plotMax/pParams['normVal']], ylog=ylog)
            else: # Just plot raw values
                skyfignum = self.binner.plotSkyMap(self.metricValues[metricName],
                                                   cmap=cmap, cbarFormat=cbarFormat,
                                                   units=units, title=title,
                                                   clims=[plotMin, plotMax], ylog=ylog)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='sky')
                plt.savefig(outfile, figformat=self.figformat)
        # Plot the angular power spectrum, if able. (healpix binners)
        if hasattr(self.binner, 'plotPowerSpectrum'):
            psfignum = self.binner.plotPowerSpectrum(self.metricValues[metricName],
                                                     title=title, 
                                                     legendLabel=legendLabel)
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='ps')
                plt.savefig(outfile, figformat=self.figformat)

        # Plot the hourglass plot
        if hasattr(self.binner, 'plotHour'):
            if xlabel == None:
                xlabel='MJD (day)'
            if ylabel == None:
                ylabel = 'Hours from local midnight'
            psfignum = self.binner.plotHour(self.metricValues[metricName][0], title=title, xlabel=xlabel,ylabel=ylabel )
            if savefig:
                outfile = self._buildOutfileName(metricName, 
                                                 outDir=outDir, outfileRoot=outfileRoot, 
                                                 plotType='hr')
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
        
