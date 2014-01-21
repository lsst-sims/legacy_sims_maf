# The base class for all spatial binners.
# Binners are 'data slicers' at heart; spatial binners slice data by RA/Dec and 
#  return the relevant indices in the simData to the metric. 
# The primary things added here are the methods to slice the data (for any spatial binner)
#  as this uses a KD-tree built on spatial (RA/Dec type) indexes. 

import numpy as np

try:
    # Try cKDTree first, as it's supposed to be faster.
    from scipy.spatial import cKDTree as kdtree
    #current stack scipy has a bad version of cKDTree.  
    if not hasattr(kdtree,'query_ball_point'): 
        from scipy.spatial import KDTree as kdtree
except:
    # But older scipy may not have cKDTree.
    from scipy.spatial import KDTree as kdtree


from .baseBinner import BaseBinner

class BaseSpatialBinner(BaseBinner):
    """Base binner object, with added slicing functions for spatial binner."""
    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate the base spatial binner object."""
        super(BaseSpatialBinner, self).__init__(verbose=verbose)
        self.binnertype = 'SPATIAL'
        return
    
    def _treexyz(self, ra, dec):
        """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
        # Note ra/dec can be arrays.
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z
    
    def buildTree(self, simDataRa, simDataDec, 
                  leafsize=500, radius=1.8):
        """Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

        simDataRA, simDataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node.
        radius = the distance (in degrees) at which matches between the simData kdtree
        and the binpoint RA/Dec value will be produced. """
        if np.any(simDataRa > np.pi*2.0) or np.any(simDataDec> np.pi*2.0):
            raise Exception('Expecting RA and Dec values to be in radians.')
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
    
    def sliceSimData(self, binpoint):
        """Return indexes for relevant opsim data at binpoint (binpoint=ra/dec value)."""
        binx, biny, binz = self._treexyz(binpoint[0], binpoint[1])
        # If we were given more than one binpoint, try multiple query against the tree.
        if isinstance(binx, np.ndarray):
            indices = self.opsimtree.query_ball_point(zip(binx, biny, binz), 
                                                      self.rad)
        # If we were given one binpoint, do a single query against the tree.
        else:
            indices = self.opsimtree.query_ball_point((binx, biny, binz), 
                                                      self.rad)
        return indices

        
