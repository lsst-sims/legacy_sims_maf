import numpy as np
import healpy as hp

def radec2pix(nside, ra, dec):
    """Convert ra,dec to the nearest heaplixel id."""
    lat = dec + np.pi/2. #theta
    lon = -ra % (np.pi*2.) #phi 
    hpid = hp.ang2pix(nside, lat, lon )
    return hpid
