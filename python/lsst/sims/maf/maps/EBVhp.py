import numpy as np
import healpy as hp
import os
from lsst.sims.maf.utils import radec2pix



def EBVhp(nside, ra=None,dec=None, pixels=None, interp=False):
    """
    Read in a healpix dust map and return values for given RA, Dec values

    nside: Healpixel resolution (2^x)
    ra: RA (can take numpy array)
    dec: Dec (can take numpy array)
    pixles: Healpixel IDs
    interp: Should returned values be interpolated (True) or just nearest neighbor(False)
    """

    if (ra is None) & (dec is None) & (pixels is None):
        raise RuntimeError("Need to set ra,dec or pixels.")

    # Load the map
    ebvDataDir=os.environ.get("SIMS_DUSTMAPS_DIR")
    filename = 'DustMaps/dust_nside_%i.npz'%nside
    dustMap = np.load(os.path.join(ebvDataDir,filename))['ebvMap']

    # If we are interpolating to arbitrary positions
    if interp:
        result = hp.get_interp_val(dustMap, dec + np.pi/2.,
                                  -ra % (np.pi*2.))
    else:
        # If we know the pixel indices we want
        if pixels is not None:
            result = dustMap[pixels]
        # Look up 
        else:
            pixels = radec2pix(nside,ra,dec)
            result = dustMap[pixels]

    return result

    
