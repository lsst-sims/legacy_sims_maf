# Base class for all grid & metrics. 
# Set up your grid (one grid per gridMetric) 
#     (including building the KDtree if a spatial metric)
# Then set up your metrics that are to be evaluated on this grid 
# Pass these (in a list) to the gridMetric and it will:
#   run the metrics on the grid
#   store the metric values at each gridpoint that result
#   read/write them to files (using the methods in grid)
#   plot the metric values (using the methods in grid)
#   

import os
import numpy as np

class BaseGridMetric(object):
    def __init__(self, grid, metricList, simData, simDataName='opsim'):
        """Set up BaseGridMetric object and validate that simData has all necessary columns for metrics.

        grid = the (single) grid that these metrics will be run over.
        metricList = a list of metrics to run. 
        simData = the simData to evaluate.
        simDataName = a tag to identify the simData (i.e. 'opsim3.61'). """
        self.grid = grid
        if hasattr(metricList, '__iter__'):
            self.metrics = metricList
        else:
            self.metrics = [metricList,]
        self.simData = simData
        # Validate that simData has all the required data values. 
        # The metrics have saved their required columns in the classRegistry.
        simCols = self.metrics[0].classRegistry.uniqueCols()
        for c in simCols:
            if c not in self.simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n ',
                                self.metrics[0].classRegistry)
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
