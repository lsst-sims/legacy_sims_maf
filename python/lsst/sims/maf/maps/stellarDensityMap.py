from lsst.sims.maf.maps import BaseMap
import numpy as np
from lsst.utils import getPackageDir
import os

__all__ = ['StellarDensityMap']

class StellarDensityMap(BaseMap):
    """
    Return the stellar density for each slicepoint. Units of stars per sq degree
    """
    def __init__(self, nside=64, rmag=25, keyname='stellarDensity'):
        """
        rmag: The r-band magnitude limit to return the stellar density at
        """
        self.keyNames = [keyname]
        self.nside = nside
        self.rmag = rmag
        self.mapDir = os.path.join(getPackageDir('sims_dustmaps'),'StarMaps')

    def run(self, slicePoints):
        if 'nside' in slicePoints.keys():
            if slicePoints['nside'] == 64:
                mapfile = os.path.join(self.mapDir, 'starDensity_nside_64_rmagLimit_%i.npz' % self.rmag)
                data = np.load(mapfile)
                starMap = data['starMap'].copy()
                slicePoints[self.keyNames[0]] = starMap
        else:
            # Compute the healpix for each slicepoint on the nside=64 grid
            pass

        return slicePoints
