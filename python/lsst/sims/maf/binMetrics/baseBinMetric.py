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
# Instantiate the binMetric object and set the binner and (potentially) metrics.
# Only one binner per baseBinMetric!
# Then, the actual metric data can enter the binMetric through either running metrics on simData,
#  or reading metric values from files.
#
#  'runBins' - generates metric data by running metrics over binpoints in binner.
#      pass list of metrics, simData, and metadata for the metric run;
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


import os
import warnings
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lsst.sims.maf.binners as binners
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.utils.outputUtils as outputUtils

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


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
        self.sqlconstraint = {}
        self.metadata = {}
        self.binner = None
        self.outputFiles = {}

    def _buildOutfileName(self, metricName,
                          outDir=None, outfileRoot=None, plotType=None):
        """Builds an automatic output file name for metric data or plots."""
        # Set output directory and root 
        if outDir is None:
            outDir = '.'
        # Start building output file name.
        if outfileRoot is None:
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
            if len(self.metadata[metricName]) > 0:        
                oname = oname + '_' + self.metadata[metricName]
        # Add letter to distinguish binner types
        #   (which otherwise might have the same output name).
        oname = oname + '_' + self.binner.binnerName[:4].upper()
        # Add plot name, if plot.
        if plotType:
            oname = oname + '_' + plotType + '.' + self.figformat
        # Build outfile (with path) and strip white spaces (replace with underscores) and strip quotes. 
        outfile = os.path.join(outDir, oname.replace(' ', '_').replace("'",'').replace('"',''))
        return outfile

    def _addOutputFiles(self, metricName, key, value):
        """Add outputfilename to internal dictionary of dictionaries (keyed per metricName)
        with output filenames (plus plots) and summary metrics.

        Expected keys for each metricName dictionary are:
        metricName (de-duped), binnerName, sqlconstraint, metadata, simDataName,
        dataFile (for metric data), histFile, skyFile, psFile, (other plots),
        summary metric - metricName / summaryValue  [can be repeated]
        """
        if metricName not in self.outputFiles:
            self.outputFiles[metricName] = {}
            self.outputFiles[metricName]['metricName'] = self._dupeMetricName(metricName)
            self.outputFiles[metricName]['binnerName'] = self.binner.binnerName
            self.outputFiles[metricName]['simDataName'] = self.simDataName[metricName]
            self.outputFiles[metricName]['sqlconstraint'] = self.sqlconstraint[metricName]
            self.outputFiles[metricName]['metadata'] = self.metadata[metricName]
        if key == 'summaryStat':
            if 'summaryStat' not in self.outputFiles[metricName]:
                self.outputFiles[metricName]['summaryStat'] = {}
            self.outputFiles[metricName]['summaryStat'][value[0]] = value[1]
        else:
            # Strip out directories and leave only final file name if relevant
            if key.endswith('File') or key.endswith('Plot'):
                head, tail = os.path.split(value)
                value = tail
            self.outputFiles[metricName][key] = value

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

    def setBinner(self, binner, override=False):
        """Set binner for binMetric.

        If override = False (default), checks if binner already set, and if the two are equal."""
        if (self.binner is None) or (override):
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
        simCols = self.metricObjs[self.metricNames[0]].classRegistry.uniqueCols()
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n',
                                metricList[0].classRegistry)
        return True

    def runBins(self, simData, simDataName='opsim', sqlconstraint='', metadata=''):
        """Run metric generation over binner, for metric objects in self.metricObjs.

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
                            self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata,
                                                                                          binpoint[1], binpoint[2])
                        else:
                            self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata)
                    else:
                        self.metricValues[mname].data[i] = self.metricObjs[mname].run(slicedata)
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
            self.simDataName[rName] = self.simDataName[metricName]
            self.metadata[rName] = self.metadata[metricName]
            self.sqlconstraint[rName] = self.sqlconstraint[metricName]
            self.plotParams[rName] = self.plotParams[metricName]

    def computeSummaryStatistics(self, metricName, summaryMetric):
        """Compute single number summary of metric values in metricName, using summaryMetric.
        summaryMetric must be an object (not a class), already instantiated.
         """
        # To get (clear, non-confusing) result from unibinner, try running this with 'Identity' metric.
        # Most of the summary metrics are simpleScalarMetrics: test if this is the case, and if
        #  metricValue is compatible, in order to avoid exceptions.
        if issubclass(summaryMetric.__class__, metrics.SimpleScalarMetric):
            if self.metricValues[metricName].dtype == 'object':
                warnings.warn('Cannot compute simple scalar summary metric %s on "object" type metric value for %s'
                              % (summaryMetric.name, metricName))
                return None
        # Because of the way the metrics are built, summaryMetric will require a numpy rec array.
        # Create numpy rec array from metric data, with bad values removed. 
        rarr = np.array(zip(self.metricValues[metricName].compressed()), 
                dtype=[('metricdata', self.metricValues[metricName].dtype)])
        # The summary metric colname should already be set to 'metricdata', but in case it's not:
        summaryMetric.colname = 'metricdata'
        summaryValue = summaryMetric.run(rarr)
        # Convert to numpy array if not, for uniformity in final use.
        if isinstance(summaryValue, float) or isinstance(summaryValue, int):
            summaryValue = np.array(summaryValue)
        # Add summary metric info to outputFiles.
        self._addOutputFiles(metricName, 'summaryStat', [summaryMetric.name.replace(' metricdata', ''), summaryValue])
        return summaryValue
        
    def returnOutputFiles(self, verbose=True):
        """Return list of output file information (which is a list of dictionaries)
        If 'verbose' then prints in somewhat pretty fashion to stdout."""
        if verbose:
            subkeyorder = ['metricName', 'simDataName', 'binnerName', 'metadata', 'sqlconstraint', 'dataFile']
            outputUtils.printSimpleDict(self.outputFiles, subkeyorder)
        return self.outputFiles      
                        
    def readMetricValues(self, filenames, verbose=False):
        """Given a list of filenames, reads metric values and metadata from disk. """
        if not hasattr(filenames, '__iter__'):
            filenames = [filenames, ]        
        for f in filenames:
            basebinner = binners.BaseBinner()
            metricData, binner, header = basebinner.readData(f)
            # Check that the binner from this file matches self.binner (ok if self.binner is None)
            if not(self.setBinner(binner, override=False)):
                raise Exception('Binner for metric %s does not match existing binner.' 
                                % (header['metricName']))
            # Dedupe the metric name, if needed.
            metricName = self._deDupeMetricName(header['metricName'])
            self.metricNames.append(metricName)
            self.metricValues[metricName] = metricData
            self.metricValues[metricName].fill_value = self.binner.badval
            self.simDataName[metricName] = header['simDataName']
            self.metadata[metricName] = header['metadata']
            self.sqlconstraint[metricName] = header['sqlconstraint']
            self.plotParams[metricName] = {}
            if 'plotParams' in header:
                for pp in header['plotParams']:
                    self.plotParams[metricName][pp] = header['plotParams'][pp]
            if verbose:
                print 'Read data from %s, got metric data for metricName %s' %(f, header['metricName'])
            
    def writeAll(self, outDir=None, outfileRoot=None, comment=''):
        """Write all metric values to disk."""
        for mname in self.metricValues:
            dt = self.metricValues[mname].dtype
            self.writeMetric(mname, comment=comment, outDir=outDir, outfileRoot=outfileRoot)

        
    def writeMetric(self, metricName, comment='', outfileRoot=None, outDir=None):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
        outDir = directory to write output data (default '.').        
       """
        outfile = self._buildOutfileName(metricName, outDir=outDir, outfileRoot=outfileRoot)
        self.binner.writeData(outfile+'.npz', self.metricValues[metricName],
                              metricName = self._dupeMetricName(metricName),
                              simDataName = self.simDataName[metricName],
                              sqlconstraint = self.sqlconstraint[metricName],
                              metadata = self.metadata[metricName] + comment)
        self._addOutputFiles(metricName, 'dataFile', outfile+'.npz')

                  
    def plotAll(self, outDir='./', savefig=True, closefig=False, outfileRoot=None, verbose=True):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mname in self.metricValues:            
            plotfigs = self.plotMetric(mname, outDir=outDir, savefig=savefig, outfileRoot=outfileRoot)
            if closefig:
               plt.close('all')
            if plotfigs is None and verbose:
                warnings.warn('Not plotting metric data for %s' %(mname))
            
    def plotMetric(self, metricName, savefig=True, outDir=None, outfileRoot=None):
        """Create all plots for 'metricName' ."""
        outfile = self._buildOutfileName(metricName, outDir=outDir, outfileRoot=outfileRoot)
        # Get the metric plot parameters. 
        pParams = self.plotParams[metricName].copy()
        # Build plot title and label.
        mname = self._dupeMetricName(metricName)
        if 'title' not in pParams:
           pParams['title'] = self.simDataName[metricName] + ' ' + self.metadata[metricName]
           pParams['title'] += ': ' + mname
        if 'units' not in pParams:
           pParams['units'] = mname
           if '_units' in pParams:
              pParams['units'] += ' ('+ pParams['_units'] + ')'
        if 'xlabel' not in pParams:
           pParams['xlabel'] = pParams['units']
        # Plot the data. Plotdata for each binner returns a dictionary with the filenames, filetypes, and fig nums.
        plotResults = self.binner.plotData(self.metricValues[metricName], savefig=savefig,
                                            filename=outfile, **pParams)
        # Save information about the plotted files into the output file list.
        for filename, filetype in  zip(plotResults['filenames'], plotResults['filetypes']):
           # filetype = 'Histogram' or 'SkyMap', etc. -- add 'Plot' for output file key.
           filetype += 'Plot'
           self._addOutputFiles(metricName, filetype, filename)
        return plotResults['figs']
