# The sliceMetric class is used for running/generating metric output,
#  and can also be used for generating comparisons or summary statistics on 
#  already calculated metric outputs.
# In either case, there is only one slicer per sliceMetric, 
#   although there may be many metrics.
# 
# An important aspect of the sliceMetric is handling the metadata about each metric.
#  This includes the opsim run name, the sql constraint on the query that pulled the
#  input data (e.g. 'r band', 'X<1.2', 'WFD prop'), and the slicer type that the 
#  metric was run on. The metadata is important for
#  understanding what the metric means, and should be presented in plots & saved in the
#  output files. 
#
# Instantiate the sliceMetric object and set the slicer and (potentially) metrics.
# Only one slicer per baseSlicemetric!
# Then, the actual metric data can enter the sliceMetric through either running metrics on simData,
#  or reading metric values from files.
#
#  'runSlices' - generates metric data by running metrics over slicepoints in slicer.
#      pass list of metrics, simData, and metadata for the metric run;
#      validates that simData has needed cols, then runs metrics over the slicepoints in the slicer. 
#      Stores the results in a dictionary keyed by the metric names.
#
# 'readMetric' will read metric data from files. In this case, the metadata 
#   may not be the same for all metrics (e.g. comparing two different opsim runs). 
# To get multiple metric data into the sliceMetric, in this case run 'readMetric' 
#   multiple times (once for each metric data file) -- the metrics will be added
#   to an internal list, along with their metadata. 
#   Note that all metrics must use the same slicer. 
#
# A mixture of readMetric & runSlices can also be used to populate the data in the sliceMetric.
#
# runSlices applies to multiple metrics at once; most other methods apply to one metric 
#  at a time but a convenience method to run over all metrics is provided (i.e. reduceAll)
#
# Metric data values, as well as metadata for each metric, are stored in
#  dictionaries keyed by the metric name.


import os
import warnings
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.utils.outputUtils as outputUtils
from lsst.sims.maf.db import ResultsDb

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class BaseSliceMetric(object):
    def __init__(self, figformat='pdf', dpi=600, outDir='Output'):
        """Instantiate sliceMetric object and set up (empty) dictionaries."""
        # Set figure format for output plot files.
        self.figformat = figformat
        self.dpi = dpi
        self.metricNames = []
        self.metricObjs = {}
        self.plotParams = {}
        self.metricValues = {}
        self.simDataName = {}
        self.sqlconstraint = {}
        self.metadata = {}
        self.metricId = {}
        self.slicer = None
        self.outDir = outDir
        self.resultsDb = ResultsDb(outDir=outDir)

    def _buildOutfileName(self, metricName, outfileRoot=None, plotType=None):
        """Builds an automatic output file name for metric data or plots."""
        # Start building output file name.
        if outfileRoot is None:
            if metricName in self.simDataName:
                # If metricName is the name of an actual metric, use its associated simdata ID.
                outfileRoot = self.simDataName[metricName]
            else:
                # metricName may have been a multi-metric plot title, so try to find a good compromise.
                outfileRoot = list(set(self.simDataName.values()))
                if len(outfileRoot) > 1:
                    outfileRoot = 'comparison'
                else:
                    outfileRoot = outfileRoot[0]
        # Start building output file name from the root.
        # Strip trailing numerals from metricName (de-duplicate).
        oname = outfileRoot + '_' + self._dupeMetricName(metricName)
        # Add summary of the metadata if it exists.
        if metricName in self.metadata:
            if len(self.metadata[metricName]) > 0:        
                oname = oname + '_' + self.metadata[metricName]
        # Add letters to distinguish slicer types
        #   (which otherwise might have the same output name).
        oname = oname + '_' + self.slicer.slicerName[:4].upper()
        # Replace <, > and = signs.
        oname = oname.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
        # Strip white spaces (replace with underscores), strip '.'s and ','s
        oname = oname.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        # and strip quotes and double __'s
        oname = oname.replace('"','').replace("'",'').replace('__', '_')        
        # Add plot name, if plot.
        if plotType:
            oname = oname + '_' + plotType + '.' + self.figformat
        return oname

    def _deDupeMetricName(self, metricName):
        """In case of multiple metrics having the same 'metricName', add additional characters to de-dupe."""
        mname = metricName
        i = 0 
        while mname in self.metricNames:
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

    def setSlicer(self, slicer, override=False):
        """Set slicer for sliceMetric.

        If override = False (default), checks if slicer already set, and if the two are equal."""
        if (self.slicer is None) or (override):
            self.slicer = slicer
            return True        
        return (self.slicer == slicer)            

    def setMetrics(self, metricList):
        """Sets dictionaries for the metric objects and their plotting parameters."""
        # Keeping track of metric data values, plotting parameters, and metadata must
        # be done without depending on having the metric objects themselves, as the sliceMetric
        # may be populated with data by reading values from disk instead of calculating them.
        # All dictionaries are keyed by metric name
        #   (reduced metric data is originalmetricName.reduceFuncName). 
        if not hasattr(metricList, '__iter__'):
            metricList = [metricList,]
        newmetricNames = []
        for m in metricList:
            newmetricNames.append(self._deDupeMetricName(m.name))
        for m, mname in zip(metricList, newmetricNames):
            self.metricNames.append(mname)
            self.metricObjs[mname] = m
            self.plotParams[mname] = m.plotParams
        return newmetricNames

    def validateMetricData(self, simData):
        """Validate that simData has the required data values for the metrics in self.metricObjs."""
        simCols = self.metricObjs[self.metricNames[0]].colRegistry.colSet
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n')
        return True

    def runSlices(self, simData, simDataName='opsim', sqlconstraint='', metadata=''):
        """Run metric generation over slicer, for metric objects in self.metricObjs.

        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.) """
        # Set provenance information for each metric.
        for mname in self.metricObjs:
            self.simDataName[mname] = simDataName
            self.sqlconstraint[mname] = sqlconstraint
            self.metadata[mname] = metadata
            if len(self.metadata[mname]) == 0:                
                self.metadata[mname] = self.sqlconstraint[mname]
        # Set up (masked) arrays to store metric data. 
        for mname in self.metricObjs:
            self.metricValues[mname] = ma.MaskedArray(data = np.empty(len(self.slicer), 
                                                      self.metricObjs[mname].metricDtype),
                                                      mask = np.zeros(len(self.slicer), 'bool'),
                                                      fill_value=self.slicer.badval)
        # Set up an ordered dictionary to be the cache if needed:
        # Currently using OrderedDict, it might be faster to use 2 regular Dicts instead
        if self.slicer.cacheSize > 0:
           cacheDict = OrderedDict()
           cache = True
        else:
           cache = False
        # Run through all slicepoints and calculate metrics 
        #    (slicing the data once per slicepoint for all metrics).
        for i, slice_i in enumerate(self.slicer):
            slicedata = simData[slice_i['idxs']]
            if len(slicedata)==0:
                # No data at this slicepoint. Mask data values.
                for mname in self.metricObjs:
                    self.metricValues[mname].mask[i] = True
            else:
               if cache:
                  # Make the data idxs hashable.  
                  key = str(sorted(slice_i['idxs']))[1:-1].replace(' ','')
                  # If key exists, set flag to use it, otherwise add it
                  if key in cacheDict:
                     useCache = True 
                  else:
                     cacheDict[key] = i
                     useCache = False
                     # If we are above the cache size, drop the oldest element from the cache dict
                     if i > self.slicer.cacheSize:
                        pop = cacheDict.popitem(last=False) #remove 1st item
                  for mname in self.metricObjs:
                     if useCache:
                        self.metricValues[mname].data[i] = self.metricValues[mname].data[cacheDict[key]]
                     else:
                        self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata,
                                                                                      slicePoint=slice_i['slicePoint'])
               # Not using memoize, just calculate things normally
               else:
                  for mname in self.metricObjs:
                     self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata,
                                                                                   slicePoint=slice_i['slicePoint'])
    
           
        # Mask data where metrics could not be computed (according to metric bad value).
        for mname in self.metricObjs:            
            self.metricValues[mname].mask = np.where(self.metricValues[mname].data==self.metricObjs[mname].badval,
                                                     True, self.metricValues[mname].mask)


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
            self.metricValues[rName] = ma.MaskedArray(data = np.empty(len(self.slicer), 'float'),
                                                      mask = self.metricValues[metricName].mask,
                                                      fill_value=self.slicer.badval)
        # Run through slicepoints, applying all reduce functions.
        for i, b in enumerate(self.slicer):
            if not self.metricValues[metricName].mask[i]:
                # Get (complex) metric values for this slicepoint. 
                metricValuesPt = self.metricValues[metricName][i]
                # Evaluate reduced version of metric values.
                for rName, rFunc in zip(rNames, reduceFunc):
                    self.metricValues[rName].data[i] = rFunc(metricValuesPt)
        # Copy simdataName, metadata and comments for this reduced version of the metric data.
        for rName in rNames:
            self.simDataName[rName] = self.simDataName[metricName]
            self.metadata[rName] = self.metadata[metricName]
            self.sqlconstraint[rName] = self.sqlconstraint[metricName]
            self.plotParams[rName] = self.plotParams[metricName]

    def computeSummaryStatistics(self, metricName, summaryMetric):
        """Compute single number summary of metric values in metricName, using summaryMetric.
        summaryMetric must be an object (not a class), already instantiated.
         """
        # Cannot generally run summary statistics on object metric data, so test and skip.
        if self.metricValues[metricName].dtype == 'object':
            warnings.warn('Cannot compute simple scalar summary metric %s on "object" type metric value for %s'
                              % (summaryMetric.name, metricName))
            return None
        # To get (clear, non-confusing) result from unislicer, try running this with 'Identity' metric.
        # Create numpy structured array from metric data, with bad values removed. 
        rarr = np.array(zip(self.metricValues[metricName].compressed()), 
                dtype=[('metricdata', self.metricValues[metricName].dtype)])
        # The summary metric colname should already be set to 'metricdata', but in case it's not:
        summaryMetric.colname = 'metricdata'
        summaryValue = summaryMetric.run(rarr)
        # All summary values should be floats or ints.
        if not isinstance(summaryValue, float) and not isinstance(summaryValue, int):
            warnings.warn('Summary Metric should return float/int. Just sending this back without recording.')
            return summaryValue
        # Add summary metric info to results database.
        self.resultsDb.addSummaryStats(self.metricId[metricName],
                                       summaryName=summaryMetric.name.replace(' metricdata', ''),
                                       summaryValue=summaryValue)
        return summaryValue
        
                        
    def readMetricValues(self, filenames, verbose=False):
        """Given a list of filenames, reads metric values and metadata from disk. """
        if not hasattr(filenames, '__iter__'):
            filenames = [filenames, ]        
        for f in filenames:
            baseslicer = slicers.BaseSlicer()
            metricData, slicer, header = baseslicer.readData(f)
            # Check that the slicer from this file matches self.slicer (ok if self.slicer is None)
            if not(self.setSlicer(slicer, override=False)):
                raise Exception('Slicer for metric %s does not match existing slicer.' 
                                % (header['metricName']))
            # Dedupe the metric name, if needed.
            metricName = self._deDupeMetricName(header['metricName'])
            self.metricNames.append(metricName)
            self.metricValues[metricName] = metricData
            self.metricValues[metricName].fill_value = self.slicer.badval
            self.simDataName[metricName] = header['simDataName']
            self.metadata[metricName] = header['metadata']
            self.sqlconstraint[metricName] = header['sqlconstraint']
            self.plotParams[metricName] = {}
            if 'plotParams' in header:
                for pp in header['plotParams']:
                    self.plotParams[metricName][pp] = header['plotParams'][pp]
            if verbose:
                print 'Read data from %s, got metric data for metricName %s' %(f, header['metricName'])
            
    def writeAll(self, outfileRoot=None, comment=''):
        """Write all metric values to disk."""
        for mname in self.metricValues:
            dt = self.metricValues[mname].dtype
            outfilename = self.writeMetric(mname, comment=comment,
                                           outfileRoot=outfileRoot)
        
    def writeMetric(self, metricName, comment='', outfileRoot=None):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
       """
        outfile = self._buildOutfileName(metricName, outfileRoot=outfileRoot) + '.npz'
        self.slicer.writeData(os.path.join(self.outDir, outfile),
                              self.metricValues[metricName],
                              metricName = self._dupeMetricName(metricName),
                              simDataName = self.simDataName[metricName],
                              sqlconstraint = self.sqlconstraint[metricName],
                              metadata = self.metadata[metricName] + comment)
        self.metricId[metricName] = self.resultsDb.addMetric(self._dupeMetricName(metricName),
                                                              self.slicer.slicerName,
                                                              self.simDataName[metricName],
                                                              self.sqlconstraint[metricName],
                                                              self.metadata[metricName],
                                                              outfile)
        # for driver merged histograms - update this later.
        # Add an attribute to the metric class object letting it know its output file.
        self.metricObjs[metricName].saveFile = outfile

                  
    def plotAll(self, savefig=True, closefig=False, outfileRoot=None, verbose=True):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mname in self.metricValues:            
            plotfigs = self.plotMetric(mname, savefig=savefig, outfileRoot=outfileRoot)
            if closefig:
               plt.close('all')
            if plotfigs is None and verbose:
                warnings.warn('Not plotting metric data for %s' %(mname))
            
    def plotMetric(self, metricName, savefig=True, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        outfile = self._buildOutfileName(metricName, outfileRoot=outfileRoot)
        # Get the metric plot parameters. 
        pParams = self.plotParams[metricName].copy()
        # Build plot title and label.
        mname = self._dupeMetricName(metricName)
        # Units always in pParams, but might be '' (are the physical units). 
        if 'title' not in pParams:
            # Build default title. 
            pParams['title'] = self.simDataName[metricName] + ' ' + self.metadata[metricName]
            pParams['title'] += ': ' + mname
        if 'ylabel' not in pParams:
            # Build default y label if needed (i.e. oneDSlicer)
            if self.slicer.slicerName == 'OneDSlicer':
                pParams['ylabel'] = mname + ' (' + pParams['units'] + ')'
        if 'xlabel' not in pParams:
            # Build a default x label if needed
            if self.slicer.slicerName == 'OneDSlicer':
                pParams['xlabel'] = self.slicer.sliceColName + ' (' + self.slicer.sliceColUnits + ')'
            else:
                pParams['xlabel'] = mname + ' (' + pParams['units'] + ')'
        # Plot the data. Plotdata for each slicer returns a dictionary with the filenames, filetypes, and fig nums.
        plotResults = self.slicer.plotData(self.metricValues[metricName], savefig=savefig,
                                           figformat=self.figformat, dpi=self.dpi,
                                           filename=os.path.join(self.outDir, outfile), **pParams)
        # Save information about the plotted files into the output file list.
        for filename, filetype in  zip(plotResults['filenames'], plotResults['filetypes']):
           # filetype = 'Histogram' or 'SkyMap', etc.
           self.resultsDb.addPlot(metricId=self.metricId[metricName], plotType=filetype, plotFile=filename)
        return plotResults['figs']
