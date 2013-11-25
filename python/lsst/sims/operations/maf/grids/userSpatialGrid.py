# Class for User-defined spatial grid. 
# User can read the grid from a file or database. 
# The method to read and write the metric encodes the grid in the data. 

import numpy as np
import matplotlib.pyplot as plt

from .baseSpatialGrid import BaseSpatialGrid

class UserSpatialGrid(BaseSpatialGrid):
    """User defined spatial grid."""
    def __init__(self, filename, verbose=True):
        """Set up healpix grid object."""
        # Bad metric data values should be set to badval
        super(UserSpatialGrid, self).__init__(verbose=verbose)
        self.badval = hp.UNSEEN 
        self._setupGrid(filename = filename)
        return

    def _setupGrid(self, filename):
        """Read the user grid from filename. """
        # Read RA and Dec points from filename.
        self.ra = numpy.zeros(100)
        self.dec = numpy.zeros(100)
        # Set self.npix
        self.npix = len(self.ra)
        return

    def __iter__(self):
        """Iterate over the grid."""
        self.ipix = 0
        return self
    
    def next(self):
        """Return RA/Dec values when iterating over grid."""
        # To make __iter__ work, you need next. 
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.npix:
            raise StopIteration
        radec = self.ra[self.ipix], self.dec[self.ipix]
        self.ipix += 1
        return radec

    def __getitem__(self, ipix):
        """Make healpix grid indexable."""
        radec = self.ra[self.ipix], self.dec[self.ipix]
        return radec

    def __eq__(self, otherGrid):
        """Evaluate if two grids are equivalent."""
        # If the two grids are both user spatial grids, check points.
        if isinstance(otherGrid, UserSpatialGrid):
            return ((numpy.all(otherGrid.ra == self.ra)) 
                    and (numpy.all(otherGrid.dec) == self.dec))
        else:
            return False
        
    def plotSkyMap(self, metricValue, metricLabel, title='', 
                   clims=None, cbarFormat='%.2g'):
        """Plot the sky map of metricValue using healpy Mollweide plot."""
        # Generate a plot.
        raise NotImplementedError()

