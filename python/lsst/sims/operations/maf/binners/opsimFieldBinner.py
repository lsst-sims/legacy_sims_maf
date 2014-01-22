# Class for opsim field based binner.

import numpy as np
import matplotlib.pyplot as plt

from .baseSpatialBinner import BaseSpatialBinner

class opsimFieldBinner(BaseSpatialBinner):
    """Opsim Field based binner."""
    def __init__(self, filename, verbose=True):
        """Set up opsim field binner object."""
        super(opsimFieldBinner, self).__init__(verbose=verbose)
        # Read RA and Dec points from filename.
        ### 
        # Set self.npix
        self.npix = len(self.ra)
        return

    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self
    
    def next(self):
        """Return RA/Dec values when iterating over binpoints."""
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.npix:
            raise StopIteration
        radec = self.ra[self.ipix], self.dec[self.ipix]
        self.ipix += 1
        return radec

    def __getitem__(self, ipix):
        radec = self.ra[self.ipix], self.dec[self.ipix]
        return radec

    def __eq__(self, otherBinner):
        """Evaluate if two grids are equivalent."""
        if isinstance(otherBinner, opsimFieldBinner):
            return ((np.all(otherBinner.ra == self.ra)) 
                    and (np.all(otherBinner.dec) == self.dec))
        else:
            return False
        
    def plotSkyMap(self, metricValue, metricLabel, title='', 
                   clims=None, cbarFormat='%.2g'):
        """Plot the sky map of metricValue."""
        # Generate a plot.
        raise NotImplementedError()

