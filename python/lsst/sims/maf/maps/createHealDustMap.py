import numpy as np
import healpy as hp
from lsst.sims.photUtils import EBV
from lsst.sims.coordUtils import AstrometryBase


# Read in the Schelgel dust maps and convert them to a healpix map
dustmap = EBV.EBVmixin()
dustmap.load_ebvMapNorth()
dustmap.load_ebvMapSouth()

# Set up the Healpixel map
nsides = [2,4,8,16,32,64,128,256,512,1024]
for nside in nsides:
    lat, lon = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    # Move dec to +/- 90 degrees
    dec = lat - np.pi/2.0
    # Flip ra from latitude to RA (increasing eastward rather than westward)
    ra = -lon % (np.pi*2)

    # Convert RA/Dec to galactic coords
    gall, galb =  AstrometryBase.equatorialToGalactic(ra,dec)

    ebvMap = dustmap.calculateEbv(gLat=galb,gLon=gall, interp=False)

    np.savez('dust_nside_%s.npz'%nside, ebvMap=ebvMap)
