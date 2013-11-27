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
# runGrid applies to multiple metrics at once; all other methods apply to one metric 
#  at a time.
#
# Metric data values, as well as metadata for each metric, are stored in
#  dictionaries keyed by the metric names (a property of the metric). 

import os
import numpy as np

class BaseGridMetric(object):
    def __init__(self, grid):
        """Instantiate gridMetric object and set grid."""
        self.grid = grid
        # Set up dictionaries to hold metric values, reduced metric values,
        #   simDataName(s) and metadata(s). All dictionary keys should be
        #   metric name -- and then for reduceValues is [metric name][reduceFuncName]
        self.metricValues = {}
        self.reduceValues = {}
        self.simDataName = {}
        self.metadata = {}
        return

    def runGrid(self, metricList, simData, 
                simDataName='opsim', metadata='', sliceCol=None):
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
            self.metricValues[m.name] = np.empty(len(self.grid), 'object') 
        # SliceCol is needed for global grids, but only has to be a specific
        #  column if the grid needs a specific column (for time slicing, for example).
        if sliceCol==None:
            sliceCol = simData.dtype.names[0]
        # Run through all gridpoints and calculate metrics 
        #    (slicing the data once per gridpoint for all metrics).
        for i, g in enumerate(self.grid):
            idxs = self.grid.sliceSimData(g, simData[sliceCol])
            for m in self.metrics:
                if len(idxs)==0:
                    # No data at this gridpoint.
                    self.metricValues[m.name][i] = self.grid.badval
                else:
                    self.metricValues[m.name][i] = m.run(simData[idxs])
        return

    def reduceAll(self):
        """Run all reduce functions on all (complex) metrics."""
        for m in self.metrics:
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
        """Run 'reduceFunc' on metric data 'metricName'. """
        # Check for a dictionary to hold the reduced values for this particular metric.
        try:
            self.reduceValues[metricName]
        except:
            self.reduceValues[metricName] = {}
        # Run reduceFunc on metricValues[metricName]. 
        rName = reduceFunc.__name__.lstrip('reduce')
        self.reduceValues[metricName][rName] = np.zeros(len(self.grid), 'float')
        for i, g in enumerate(self.grid):
            metricValuesPt = self.metricValues[metricName][i]
            if metricValuesPt == self.grid.badval:
                self.reduceValues[metricName][rName][i] = self.grid.badval
            else:
                self.reduceValues[metricName][rName][i] = reduceFunc(metricValuesPt)
        return

    def writeAll(self, outdir=None, outfile_root=None, comment=''):
        """Write all metric values to disk."""
        for m in self.metrics:
            self.writeMetric(m.name, comment=comment, outdir=outdir, outfile_root=outfile_root)
        return
    
    def writeMetric(self, metricName, comment='', outfile_root=None, outdir=None):
        """Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond 
           metric name, simDataName, and metadata).
        outfile_root = root of the output files (default simDataName).
        outdir = directory to write output data (default simDataName).  """
        if outdir == None:
            outdir = self.simDataName
        if outfile_root == None:
            outfile_root = self.simDataName
        outfile = os.path.join(outdir, outfile_root + metricName + metadata[metricName])
        self.grid.writeMetricData(outfile, self.metricValues[metricName],
                                      metricName = metricName
                                      simDataName = self.simDataName[metricName],
                                      metadata = self.metadata[metricName],
                                      comment = comment)
        return

    def readMetric(self, filename):
        """Read metric values from disk. """
        # read metrics from disk
        metricValues, metricName, simDataName, metadata, comment \
          = self.grid.readMetricData(filename)
        # Store results.
        self.metricValues[metricName] = metricValues
        self.simDataName[metricName] = simDataName
        self.metadata[metricName] = metadata + comment
        ### What do we do about complex metrics -- does name alone give enough info
        ### to instantiate a new object to access 'reduce' functions? (possibly new 
        ### reduce functions as the results of old ones should be stored with the data)?
        ### Or is that a user's problem, that if they're using
        ### a complex metric and have added new reduce functions, then they ought to
        ### know where the data should be going. 
        return    
        
    def plotMetric(self, metricName, reduceName=None):
        # Plot the sky map, if available.         
        # Plot the histogram.
        pass

    def computeSummaryStatistics(self):
        # compute the summary statistics .. note can pass metric values into
        # another global grid and then pass any metric to be evaluated on the 
        # GlobalGrid! (mean/min/rms/...). 
        pass 
