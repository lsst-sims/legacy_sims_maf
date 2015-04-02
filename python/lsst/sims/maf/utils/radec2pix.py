import numpy as np
import healpy as hp

__all__ = ['radec2pix']

def radec2pix(nside, ra, dec):
    """Convert ra,dec to the nearest heaplixel id."""
    lat = np.pi/2. - dec
    hpid = hp.ang2pix(nside, lat, ra )
    return hpid
