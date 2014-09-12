from lsst.sims.maf.maps import BaseMap
from .EBVhp import EBVhp

class DustMap(BaseMap):
    """
    Compute the E(B-V) for each point in a given slicePoint
    """

    def __init__(self, interp=False, nside=128):
        """
        interp: should the dust map be interpolated (True) or just use the nearest value (False).
        """
        self.keynames = ['ebv']
        self.interp = interp
        self.nside = nside
        
    def run(self, slicePoints):
        # If the slicer has nside, it's a healpix slicer so we can read the map directly
        if 'nside' in slicePoints.keys():
            slicePoints['ebv'] = EBVhp(slicePoints['nside'], pixels=slicePoints['sid'])
        # Not a healpix slicer, look up values based on RA,dec with possible interpolation
        else:
            slicePoints['ebv'] = EBVhp(self.nside, ra=self.slicePoints['ra'],
                                            dec=self.slicePoints['dec'], interp=self.interp)
        
        return slicePoints
    
