import numpy as np
import healpy as hp
from lsst.sims.photUtils import EBV


# Read in the Schelgel dust maps and convert them to a healpix map
dustmap = EBV.EBVbase()
dustmap.load_ebvMapNorth()
dustmap.load_ebvMapSouth()

# Set up the Healpixel map
nsides = [2,4,8,16,32,64,128,256,512,1024]
for nside in nsides:
    lat, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    # Move dec to +/- 90 degrees
    dec = np.pi/2.0 - lat
    ebvMap = dustmap.calculateEbv(ra=ra, dec=dec, interp=False)
    # When new API for calculateEbv is out:
    # ebvMap = dustmap.calculateEbv(np.array([ra,dec]), interp=False)
    np.savez('dust_nside_%s.npz'%nside, ebvMap=ebvMap)
