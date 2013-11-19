# Base class for all grid & metrics. 
# The gridMetric class is used for running/generating metric output,
#  and can also be used for generating comparisons or summary statistics on 
#  already calculated metric outputs.
# In either case, there is only one grid per group of metrics.
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
#  setupRun - pass list of metrics and simData; validates that simData has needed cols.
#  buildKDTree (if needed/not already set up for grid)
#  runGrid  - runs metrics over grid & stores metric data.
#
# 'readMetrics' will read metric data from files. In this case, the metadata 
#   may not be the same for all metrics (e.g. comparing two different opsim runs).  
#  For readMetrics, provide a list of filenames and readMetrics will
#   read the metric data values from the files using the methods in the grid class
#   (Note this means the user has to know the grid type at the start**).


# Pass these (in a list) to the gridMetric and it will:
#   run the metrics on the grid
#   store the metric values at each gridpoint 
#   read/write them to files (using the methods in grid)
#   plot the metric values (using the methods in grid)
#   

import os
import numpy as np

class BaseGridMetric(object):
    def __init__(self, grid):
        """Instantiate gridMetric object and set grid."""
        self.grid = grid
        return

    def setupRun(self, metricList, simData, simDataName='opsim', sqlconstraint=''):
        """Set metrics and metadata for metric generation, validate that simData has columns needed for metrics.

        metricList = a list of metrics to run. 
        simData = the simData to evaluate.
        simDataName = a tag to identify the simData (i.e. 'opsim3.61').
        sqlconstraint = characteristics of the simData (i.e. 'r', 'X<1.2', 'WFD')."""
        # Set metrics (convert to list if not iterable). 
        if hasattr(metricList, '__iter__'):
            self.metrics = metricList
        else:
            self.metrics = [metricList,]
        self.setSimData(simData, simDataName, sqlconstraint)
        # Validate that simData has all the required data values. 
        # The metrics have saved their required columns in the classRegistry.
        simCols = self.metrics[0].classRegistry.uniqueCols()
        for c in simCols:
            if c not in self.simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n ',
                                self.metrics[0].classRegistry)
        return    

    def setSimData(simData, simDataName='opsim', sqlconstraint=''):
        """Set simData and metadata for metric generation. """
        # Set simData.
        self.simData = simData
        # Set metadata. 
        self.simDataName = simDataName
        self.sqlconstraint = sqlconstraint
        return
        
    def buildKDTree(self, racol='fieldra', deccol='fielddec', leafsize=500, radius=1.8):
        """Build kdtree if needed to run spatial grid on metrics. Calls buildTree from baseSpatialGrid.

        racol = RA column name to use for building kdtree.
        deccol = Dec column name to use for building kdtree.
        leafsize = number of observations to leave in leafnodes of tree.
        radius = match radius for kdtree (in degrees). """
        try: 
            self.simData
        except:
            raise Exception('Set simData first.')        
        self.grid.buildTree(self.simData[racol], self.deccol[deccol], 
                            leafsize=leafsize, radius=radius)
        return

    def runGrid(self, sliceCol=None):
        """Run metrics in metricList over the grid and store the results. """
        # Set up arrays to store metric data.
        self.metricValues = {}
        for m in self.metrics:
            self.metricValues[m.name] = np.zeros(len(self.grid), 'float')
        # Run through gridpoints and evaluate metrics at each gridpoint.
        if sliceCol==None:
            sliceCol = self.simData.dtype.names[0]
        for i, g in enumerate(self.grid):
            idxs = self.grid.sliceSimData(g, self.simData[sliceCol])
            for m in self.metrics:
                if len(idxs)==0:
                    # No data at this gridpoint.
                    self.metricValues[m.name][i] = self.grid.badval
                self.metricValues[m.name][i] = m.run(self.simData[idxs])
        return

    def writeMetrics(self, comment=None, outfile_root=None, outdir=None):
        """Write metric values to disk.

        comment = string describing the metricValue provenance (default None).
        outfile_root = root of the output files (default simDataName).
        outdir = directory to write output data (default simDataName).  """
        if outdir == None:
            outdir = self.simDataName
        if outfile_root == None:
            outfile_root = self.simDataName
        for m in self.metrics:
            outfile = os.path.join(outdir, outfile_root + m.name)
            self.grid.writeMetricData(outfile, self.metricValues[m.name], 
                                      comment=comment)
        return

    def readMetrics(self):
        # read metrics from disk
        pass
    
    def plotGrid(self):
        # Plot the sky map, if available. 
        # Plot the histograms.
        pass

    def computeSummaryStatistics(self):
        # compute the summary statistics .. note can pass metric values into
        # another global grid and then pass any metric to be evaluated on the 
        # GlobalGrid! (mean/min/rms/...). 
        pass 
