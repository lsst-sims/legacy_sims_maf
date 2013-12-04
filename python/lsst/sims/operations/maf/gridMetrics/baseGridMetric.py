# Base class for all grid & metrics. 
# The gridMetric class is used for running/generating metric output,
#  and can also be used for generating comparisons or summary statistics on 
#  already calculated metric outputs.
# In either case, there is only one grid per gridMetric, 
#   although there may be many metrics.
# 
# An important aspect of the gridMetric is handling the metadata about each metric.
#  This includes the opsim run name, the sql constraint on the query that pulled the
#  input data (e.g. 'r band', 'X<1.2', 'WFD prop'), and the grid type that the 
#  metric was run on (global vs spatial, timestep..). The metadata is important for
#  understanding what the metric means, and should be presented in plots & saved in the
#  output files. 
#
# Instantiate the gridMetric object by providing a grid object (for spatial metrics,
#   this does not have to be 'set up' - it does not need the kdtree to be built). 
# Then, Metric data can enter the gridMetric through either running metrics on simData,
#  or reading metric values from files. 
# To run metrics on simData, 
#  runGrid - pass list of metrics, simData, and metadata for the metric run;
#      validates that simData has needed cols, then runs metrics over the grid. 
#      Stores the results in a dictionary keyed by the metric names.
#
# 'readMetric' will read metric data from files. In this case, the metadata 
#   may not be the same for all metrics (e.g. comparing two different opsim runs). 
# To get multiple metric data into the gridMetric, in this case run 'readMetric' 
#   multiple times (once for each metric data file) -- the metrics will be added
#   to an internal list, along with their metadata. 
#   Note that all metrics must use the same grid. 
#
# A mixture of readMetric & runGrid can be used to populate the data in the gridMetric!
#
# runGrid applies to multiple metrics at once; most other methods apply to one metric 
#  at a time but a convenience method to run over all metrics is usually provided (i.e. reduceAll)
#
# Metric data values, as well as metadata for each metric, are stored in
#  dictionaries keyed by the metric names (a property of the metric). 

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyfits as pyf

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class BaseGridMetric(object):
    def __init__(self, figformat='png'):
        """Instantiate gridMetric object and set up (empty) dictionaries."""
        # Set up dictionaries to hold metric values, reduced metric values,
        #   simDataName(s) and metadata(s). All dictionary keys should be
        #   metric name -- and then for reduceValues is [metric name][reduceFuncName]
        self.metricValues = {}
        self.simDataName = {}
        self.metadata = {}
        self.comment={}
        # Set figure format for output plot files.
        self.figformat = figformat
        return

    def _buildOutfileName(self, metricName,
                          outDir=None, outfileRoot=None, plotType=None):
        """Builds output file name for 'metricName'.

        Output filename uses outDir and outfileRoot (defaults are to use '.' and simDataName).
        For plots, builds outfile name for plot, with addition of plot type at start."""
        # Set output directory and root 
        if outDir == None:
            outDir = '.'
        if outfileRoot == None:
            outfileRoot = self.simDataName[metricName]
        # Start building output file name. Strip trailing numerals from metricName.
        oname = outfileRoot + '_' + self._dupeMetricName(metricName)
        # Add summary of the metadata if exists.
        try:
            self.metadata[metricName]    
            if len(self.metadata[metricName]) > 0:        
                oname = oname + '_' + self.metadata[metricName][:5]
        except KeyError:
            pass
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
            return ''.join(mname.split('__')[:-1])
        else:
            return metricName

    def setGrid(self, grid):
        """Set grid object for gridMetric."""
        self.grid = grid
        return
        
    def runGrid(self, metricList, simData, 
                simDataName='opsim', metadata='', sliceCol=None):
        """Run metric generation over grid.

        metricList = list of metric objects
        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.)
        sliceCol = column for slicing grid, if needed (default None)"""
        # I'm going to assume that we will never get duplicate metricNames from this method, 
        #  as metricList would give the same results for the same metric run on same data.
        # Set metrics (convert to list if not iterable). 
        if hasattr(metricList, '__iter__'):
            self.metrics = metricList
        else:
            self.metrics = [metricList,]        
        # Validate that simData has all the required data values. 
        # The metrics have saved their required columns in the classRegistry.
        simCols = self.metrics[0].classRegistry.uniqueCols()
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n',
                                self.metrics[0].classRegistry)
        # And verify that sliceCol is part of simData too.
        if sliceCol != None:
            if sliceCol not in simData.dtype.names:
                raise Exception('Simdata slice column', sliceCol, 'not in simData.')
        # Set metadata for each metric.
        for m in self.metrics:
            self.simDataName[m.name] = simDataName
            self.metadata[m.name] = metadata
        # Set up arrays to store metric data. 
        for m in self.metrics:
            self.metricValues[m.name] = np.empty(len(self.grid), m.metricDtype) 
        # SliceCol is needed for global grids, but only has to be a specific
        #  column if the grid needs a specific column (for time slicing, for example).
        if sliceCol==None:
            sliceCol = simData.dtype.names[0]
        # Run through all gridpoints and calculate metrics 
        #    (slicing the data once per gridpoint for all metrics).
        for i, g in enumerate(self.grid):
            idxs = self.grid.sliceSimData(g, simData[sliceCol])
            slicedata = simData[idxs]
            if len(idxs)==0:
                # No data at this gridpoint.
                for m in self.metrics:
                    self.metricValues[m.name][i] = self.grid.badval
            else:
                for m in self.metrics:
                    self.metricValues[m.name][i] = m.run(slicedata)
        return

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
            for reduceFunc in m.reduceFuncs.values():
                # Apply reduce function.
                self.reduceMetric(m.name, reduceFunc)
        return
                
    def reduceMetric(self, metricName, reduceFunc):
        """Run 'reduceFunc' (method on metric object) on metric data 'metricName'. """
        # Run reduceFunc on metricValues[metricName]. 
        rName = metricName + '_' + reduceFunc.__name__.lstrip('reduce')
        self.metricValues[rName] = np.zeros(len(self.grid), 'float')
        for i, g in enumerate(self.grid):
            # Get (complex) metric values for this gridpoint. 
            metricValuesPt = self.metricValues[metricName][i]
            # Evaluate reduced version of metric values.
            if metricValuesPt == self.grid.badval:
                self.metricValues[rName][i] = self.grid.badval
            else:
                self.metricValues[rName][i] = reduceFunc(metricValuesPt)
        return

    def writeAll(self, outDir=None, outfileRoot=None, comment='',  gridfile='grid.obj'):
        """Write all metric values to disk."""
        for mk in self.metricValues.keys():
            dt = self.metricValues[mk].dtype
            self.writeMetric(mk, comment=comment, outDir=outDir, outfileRoot=outfileRoot, \
                             gridfile=self._buildOutfileName(gridfile, outDir=outDir, 
                                                             outfileRoot=outfileRoot),
                                                             dt=dt)
        self.writeGrid(gridfile=gridfile, outfileRoot=outfileRoot,outDir=outDir)
        return
    
    def writeMetric(self, metricName, comment='', outfileRoot=None, outDir=None, 
                    dt='float', gridfile=None):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
        outDir = directory to write output data (default '.').
        dt = data type.
        gridfile = the filename for the pickled grid"""
        outfile = self._buildOutfileName(metricName, outDir=outDir, outfileRoot=outfileRoot)
        self.grid.writeMetricData(outfile, self.metricValues[metricName],
                                  metricName = metricName,
                                  simDataName = self.simDataName[metricName],
                                  metadata = self.metadata[metricName],
                                  comment = comment, dt=dt, gridfile=gridfile,
                                  badval = self.grid.badval)
        return
        
    def writeGrid(self,  gridfile='grid.obj',outfileRoot=None, outDir=None):
        """Write a pickle of the grid to disk."""
        outfile = self._buildOutfileName(gridfile, outDir=outDir, outfileRoot=outfileRoot)
        modgrid = self.grid
        if hasattr(modgrid,'opsimtree'):  delattr(modgrid,'opsimtree') #some kdtrees can't be pickled
        pickle.dump(modgrid, open(outfile,'w'))
        return

    def readGrid(self, gridfile='grid.obj'):
       self.grid = pickle.load(open(gridfile, 'r'))
       return
    
    def readMetric(self, filenames, checkGrid=True):
        """Read metric values and grid (pickle object) from disk.
        checkGrid:  make sure the gridtype and number of points match the properties of self.grid"""
        # Here we can get duplicate metric names however, as we could
        #  have the same metric with different opsim or metadata values.

        # Read metrics from disk
        for f in filenames:
           metricValues, metricName, simDataName, metadata, \
               comment,gridfile,gridtype \
               = self.grid.readMetricData(f)
           # Dedupe the metric name, if needed.
           if metricName != self._deDupeMetricName(metricName):
              metricName = self._deDupeMetricName(metricName)
              print '# Read multiple metrics with same name - using %s' %(metricName)
           # Store the header values in variables
           self.metricValues[metricName] = metricValues
           self.simDataName[metricName] = simDataName
           self.metadata[metricName] = metadata
           self.comment[metricName] = comment
           if checkGrid:
              if gridtype != self.grid.gridtype:
                 raise Exception('Metrics not computed on currently loaded grid type.')           
              if np.size(metricValues) != self.grid.npix:
                 raise Exception('Metric does not have the same number of points as loaded grid.')
        return    

    def plotAll(self, savefig=True):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mk in self.metricValues.keys():
            try:
                self.plotMetric(mk, savefig=savefig)
            except ValueError:
                continue 
        return        

    def plotMetric(self, metricName, *args, **kwargs):
        """Create all plots for 'metricName'."""
        # Implemented in spatialGridMetric or globalGridMetric.
        raise NotImplementedError()

    def plotComparisons(self, metricNameList, *args, **kwargs):
        """Create comparison plots of all metricValues in metricNameList."""
        # Implemented in spatialGridMetric or globalGridMetric.
        raise NotImplementedError()

    def computeSummaryStatistics(self, metricName, summaryMetric):
        # Implemented in spatialGridMetric or globalGridMetric.
        raise NotImplementedError()
