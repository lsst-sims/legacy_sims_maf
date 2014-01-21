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
# Instantiate the gridMetric object by providing a grid object. 
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
# A mixture of readMetric & runGrid can also be used to populate the data in the gridMetric.
#
# runGrid applies to multiple metrics at once; most other methods apply to one metric 
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


class BaseGridMetric(object):
    def __init__(self, figformat='png'):
        """Instantiate gridMetric object and set up (empty) dictionaries."""
        # Set up dictionaries to hold metric values, reduced metric values,
        #   simDataName(s) and metadata(s). All dictionary keys should be
        #   metric name (reduced metric data is originalmetricName.reduceFuncName). 
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
                    # (the gridMetric doesn't have self.metrics set .. perhaps because have just
                    #  read back in the metric values.)
                    outfileRoot = 'comparison'
        # Start building output file name. Strip trailing numerals from metricName.
        oname = outfileRoot + '_' + self._dupeMetricName(metricName)
        # Add summary of the metadata if it exists.
        try:
            self.metadata[metricName]    
            if len(self.metadata[metricName]) > 0:        
                oname = oname + '_' + self.metadata[metricName][:5]
        except KeyError:
            pass
        # Add letter to distinguish spatial grid metrics from global grid metrics 
        #   (which otherwise might have the same output name).
        if self.grid.gridtype == 'SPATIAL':
            oname = oname + '_sp'
        elif self.grid.gridtype == 'GLOBAL':
            oname = oname + '_gl'        
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

    def setGrid(self, grid):
        """Set grid object for gridMetric."""
        self.grid = grid
        return
    
    def validateMetricData(self, metricList, simData):
        """Validate that simData has the required data values for the metrics in metricList."""
        simCols = self.metrics[0].classReigstry.uniqueCols()
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n',
                                self.metrics[0].classRegistry)
        return

    def runGrid(self, metricList, simData, 
                simDataName='opsim', metadata='', sliceCol=None, **kwargs):
        """Run metric generation over grid.

        metricList = list of metric objects
        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data
        metadata = further information from config files ('WFD', 'r band', etc.)
        sliceCol = column for slicing grid, if needed (default None)"""
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
            self.metricValues[m.name] = ma.MaskedArray(data = np.empty(len(self.grid), m.metricDtype),
                                                       mask = np.zeros(len(self.grid), 'bool'),
                                                       fill_value=self.grid.badval)
        # SliceCol is needed for global grids, but only has to be a specific
        #  column if the grid needs a specific column (for time slicing, for example).
        if sliceCol==None:
            sliceCol = simData.dtype.names[0]
        if sliceCol not in simData.dtype.names:
            raise Exception('Simdata slice column', sliceCol, 'not in simData.')
        # Run through all gridpoints and calculate metrics 
        #    (slicing the data once per gridpoint for all metrics).
        for i, g in enumerate(self.grid):
            idxs = self.grid.sliceSimData(g, simData[sliceCol])
            slicedata = simData[idxs]
            if len(idxs)==0:
                # No data at this gridpoint. Mask data values.
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
            self.metricValues[m.name].fill_value = self.grid.badval
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
            # Apply reduce functions
            self.reduceMetric(m.name, m.reduceFuncs.values())            
        return
                
    def reduceMetric(self, metricName, reduceFunc):
        """Run 'reduceFunc' (method on metric object) on metric data 'metricName'. 

        reduceFunc can be a list of functions to be applied to the same metric data."""
        # Run reduceFunc(s) on metricValues[metricName]. 
        # Turn reduceFunc into a list if it wasn't, to make everything consistent.
        if not isinstance(reduceFunc, list):
            reduceFunc = [reduceFunc,]
        # Set up metric reduced value names.
        rNames = []
        for r in reduceFunc:
            rNames.append(metricName + '_' + r.__name__.lstrip('reduce'))
        # Set up reduced metric values masked arrays, copying metricName's mask.
        for rName in rNames:
            self.metricValues[rName] = ma.MaskedArray(data = np.empty(len(self.grid), 'float'),
                                                      mask = self.metricValues[metricName].mask,
                                                      fill_value=self.grid.badval)
        # Run through grid, applying all reduce functions.
        for i, g in enumerate(self.grid):
            if not self.metricValues[metricName].mask[i]:
                # Get (complex) metric values for this gridpoint. 
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
        return

    def writeAll(self, outDir=None, outfileRoot=None, comment='',  gridfile='grid.obj'):
        """Write all metric values to disk."""
        for mk in self.metricValues.keys():
            dt = self.metricValues[mk].dtype
            gridfilename = self._buildOutfileName(gridfile, outDir=outDir, 
                                                  outfileRoot=outfileRoot)
            self.writeMetric(mk, comment=comment, outDir=outDir, outfileRoot=outfileRoot, \
                             gridfile=gridfilename, dt=dt)
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
        print metricName, outfile
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

        checkGrid =  make sure the gridtype and number of points match the properties of self.grid"""
        # Here we can get duplicate metric names however, as we could
        #  have the same metric with different opsim or metadata values.

        # Read metrics from disk
        for f in filenames:
           metricValues, metricName, simDataName, metadata, \
               comment,gridfile,gridtype, metricHistValues,metricHistBins \
               = self.grid.readMetricData(f)
           # Dedupe the metric name, if needed.
           metricName = self._deDupeMetricName(metricName)
           # Store the header values in variables
           self.metricValues[metricName] = ma.MaskedArray(data = metricValues,
                                                          mask = np.where(metricValues == 
                                                                          self.grid.badval, True, 
                                                                          False),
                                                            fill_value = self.grid.badval)
           if hasattr(self,'metricHistValues'):
              self.metricHistValues[metricName] = metricHistValues
              self.metricHistBins[metricName] = metricHistBins
           self.simDataName[metricName] = simDataName
           self.metadata[metricName] = metadata
           self.comment[metricName] = comment
           if checkGrid:               
              if gridtype != self.grid.gridtype:
                 raise Exception('Metrics not computed on currently loaded grid type.')           
              if np.size(metricValues) != self.grid.npix:
                 raise Exception('Metric does not have the same number of points as loaded grid.')
        return    

    def plotAll(self, outDir='./', savefig=True, closefig=False):
        """Plot histograms and skymaps (where relevant) for all metrics."""
        for mk in self.metricValues.keys():
            try:
                self.plotMetric(mk, outDir=outDir, savefig=savefig)
                if closefig:
                   plt.close('all')
            except ValueError:
                continue 
        return        

    def plotMetric(self, metricName, outDir='./', *args, **kwargs):
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
