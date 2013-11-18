# The base class for all spatial grids.
# Grids are 'data slicers' at heart; spatial grids slice data by RA/Dec and 
#  return the relevant indices in the simData to the metric. 
# 
# The primary things added here are the methods to slice the data (for any spatial grid)

import numpy as np
try:
    # Try cKDTree first, as it's supposed to be faster.
    from scipy.spatial import cKDTree as kdtree 
except:
    # But some computers in department only have KDTree.
    from scipy.spatial import KDTree as kdtree
# Check API is the same. (was on my laptop).

from baseGrid import BaseGrid

class BaseSpatialGrid(BaseGrid):
    """Base grid object, with added slicing functions for spatial grids."""

    def _treexyz(self, ra, dec):
        """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
        # Note ra/dec can be arrays.
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z
    
    def buildTree(self, simDataRa, simDataDec, 
                  leafsize=500, radius=1.8):
        """Build KD tree on simDataRA/Dec values. 

        leafsize corresponds to the number of Ra/Dec pointings in each leaf node.
        radius corresponds to the distance at which matches between the simData kdtree
         and the gridpoint RA/Dec value will be produced. (Can be set independently via setRad method).  """
        x, y, z = self._treexyz(simDataRa, simDataDec)
        data = zip(x,y,z)
        self.opsimtree = kdtree(data, leafsize=leafsize)
        self.setRad(radius)
        return

    def setRad(self, radius=1.8):
        """Set radius (in degrees) for kdtree search.
        
        kdtree queries will return pointings within rad."""        
        x0, y0, z0 = (1, 0, 0)
        x1, y1, z1 = self._treexyz(np.radians(radius), 0)
        self.rad = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
        return
    
    def sliceSimData(self, gridpoint, simDataCol):
        """Return indexes for relevant opsim data at gridpoint (gridpoint=ra/dec)."""
        # SimData not needed here, but keep interface the same for all grids.
        gridx, gridy, gridz = self._treexyz(gridpoint[0], gridpoint[1])
        if isinstance(gridx, np.ndarray):
            indices = self.opsimtree.query_ball_point(zip(gridx, gridy, gridz), 
                                                      self.rad)
        else:
            indices = self.opsimtree.query_ball_point((gridx, gridy, gridz), 
                                                      self.rad)
        return indices


