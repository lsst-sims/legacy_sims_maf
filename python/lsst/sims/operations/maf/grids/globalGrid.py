# Global grid class for metrics.
# This kind of grid considers all visits / sequences / etc. regardless of RA/Dec.
# Metrics are calculable on a global grid - the data they will receive will be
#  all of the relevant data, which in this case is all of the data values in a 
#  particular simData column. 
# Subclasses of the global metric could slice on other aspects of the simData (but not
#  RA/Dec as that is done more efficiently in the spatial classes).

import numpy as np
from baseGrid import BaseGrid

class GlobalGrid(BaseGrid):
    """Global grid"""
    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate object and call set up global grid method."""
        super(GlobalGrid, self).__init__(verbose=verbose)
        self._setupGrid(*args, **kwargs)
        self.gridtype = 'GLOBAL'
        return
    
    def _setupGrid(self, *args, **kwargs):
        """Set up global grid.

        For base GlobalGrid class, this does nothing. For subclasses this could
        (for example) split the grid by time. """
        # Set number of 'pixels' in the grid. 
        # Corresponds to the number of metric values returned from the metric.
        self.npix = 1
        return

    def __iter__(self):
        """Iterate over the grid."""
        self.ipix = 0
        return self

    def next(self):
        """Set the gridvalues to return when iterating over grid."""
        if self.ipix >= self.npix:
            raise StopIteration
        ipix = self.ipix
        self.ipix += 1
        return ipix

    def __getitem__(self, ipix):
        """Make global grid indexable."""  
        return ipix
    
    def __eq__(self, otherGrid):
        """Evaluate if grids are equivalent."""
        if isinstance(otherGrid, GlobalGrid):
            return True
        else:
            return False
            
    def sliceSimData(self, gridpoint, simDataCol):
        """Return relevant indices in simData for 'gridpoint'. 

        For base GlobalGrid, this is all data."""
        indices = np.where(simDataCol)
        return indices

    def plotHistogram(self, simDataCol, simDataColLabel, title=None, fignum=None, 
                      legendLabel=None, addLegend=False, bins=None, cumulative=False,
                      histRange=None, flipXaxis=False, scale=1.0):
        """Plot a histogram of simDataCol values, labelled by simDataColLabel.

        simDataCol = the data values for generating the histogram
        simDataColLabel = the units for the simDataCol ('m5', 'airmass', etc.)
        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, try to set)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)"""
        super(GlobalGrid, self).plotHistogram(simDataCol, simDataColLabel, 
                                              title=title, fignum=fignum, 
                                              legendLabel=label, addLegend=addLegend,
                                              bins=bins, cumulative=cumulative,
                                              histRange=histRange, flipXaxis=flipXaxis,
                                              scale=scale)
