# Base class for all grid objects. 
# 
# Grid objects must know slice the complete set of incoming opsim data for the metrics:
#  * for spatial metrics, this means slicing in RA/Dec space and iterating over the sky)
#  * for global metrics this means handing over the entire (or part of the) data column to the metric)
# To facilitate metric calculation, the grid should be iterable and indexable:
#  (for spatial metrics, this means iterating over the RA/Dec points)
#  (for global metrics, this means iterating over the visits with a particular 'stepsize' -- for the base global grid, the stepsize is all the visits.)
# Grid metrics must also know how to set themselves up ('set up the grid'),
# read and write metric data, and generate plot representations of the metric data. 
# In order to compare metrics calculated on various grids, they must also be
#  able to at least check if two grids are equal. 

# TODO add read/write sql constraint & metric name

import numpy as np

# Might have to fiddle with the next few lines in case of non-interactive shell.
import matplotlib
#matplotlib.use('Agg')
import pylab

class BaseGrid(object):
    """Base class for all grid objects: sets required methods and implements common functionality."""

    def __init__(self, verbose=True, *args, **kwargs):
        """Set up grid."""
        self.verbose = verbose
        self.badval = np.nan
        return

    def __len__(self):
        """Return npix, the number of pixels in the grid."""
        return self.npix

    def __iter__(self):
        """Iterate over the grid."""
        raise NotImplementedError()

    def next(self):
        """Set the gridvalues to return when iterating over grid."""
        raise NotImplementedError()

    def __getitem__(self):
        """Make grid indexable."""
        raise NotImplementedError()
    
    def __eq__(self, othergrid):
        """Evaluate if two grids are equivalent."""
        raise NotImplementedError()
    
    def sliceSimData(self, gridpoint, simDataCol, **kwargs):
        """Slice the simulation data appropriately for the grid.

        This slice of data should be the indices of the numpy rec array (the simData)
        which are appropriate for the metric to be working on, at that gridpoint."""
        raise NotImplementedError()

    def writeMetricData(self, outfilename, metricValues, comment=None):
        """Save metric data to disk."""
        # Individual grids can and should overwrite this with more appropriate methods.
        if len(metricValues) != len(self):
            raise Exception('Length of metric values must match length of grid points.')
        if self.verbose:
            print 'Writing metric values to ascii file %s' %(outfilename)
        f = open(outfilename, 'w')
        print >>f, "#", comment
        for gridpoint, metricValue in zip(self, metricValues):
            print >>f, gridpoint, metricValue
        f.close()
        return

    def readMetricData(self, infilename):
        """Read metric data from disk."""
        # Individual grids can and should overwrite this with more appropriate methods.
        if self.verbose:
            print 'Reading (single) metric values from ascii file %s' %(infilename)
        f = open(infilename)
        metricValues = []
        gridpoints = []
        for line in f:
            if line.startswith('#') or line.startswith('!'):
                continue
            # Assume that gridpoint and metric are single values.
            gridpoints.append(line.split()[0])
            metricValues.append(line.split()[1])
        gridpoints = np.array(gridpoints)
        metricValues = np.array(metricValues)
        f.close()
        self.gridpoints = gridpoints
        self.npix = len(self.gridpoints)
        return metricValues

    def plotHistogram(self, metricValue, metricLabel, title=None, 
                      fignum=None, legendLabel=None, addLegend=False, 
                      bins=None, cumulative=False, histRange=None, flipXaxis=False,
                      scale=1.0):
        """Plot a histogram of metricValue, labelled by metricLabel.

        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, try to set)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by pylab.hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)"""
        # Histogram metricValues. 
        if fignum != None:
            fig = pylab.figure(fignum)
        else:
            fig = pylab.figure()
        # Estimate number of bins needed (unless passed bins).
        if bins == None:
            bins = int(self.npix/2000.0)
            if bins < 20.:
                bins = self.npix
        # Need to only use 'good' values in histogram.
        good = np.where(metricValue != self.badval)
        n, b, p = pylab.hist(metricValue[good], bins=bins, histtype='step', 
                             cumulative=cumulative, range=histRange, label=label)
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(x,  pos):        
            return "%.3f" % (x * scale)
        ax = pylab.gca()
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(mjrFormatter))
        pylab.ylabel('Area (1000s of square degrees)')
        pylab.xlabel(metricUnit)
        if flipXaxis:
            # Might be useful for magnitude scales.
            x0, x1 = pylab.xlim()
            pylab.xlim(x1, x0)
        if addLegend:
            pylab.legend(fancybox=True, fontsize='smaller', loc='upper left')
        if title!=None:
            pylab.title(title)
        # Return figure number (so we can reuse this if desired).         
        return fig.number
            
