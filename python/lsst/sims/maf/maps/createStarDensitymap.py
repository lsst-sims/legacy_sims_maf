import numpy as np
from lsst.sims.utils import ObservationMetaData
import healpy as hp
from lsst.sims.catalogs.generation.db import CatalogDBObject
from lsst.sims.catUtils.exampleCatalogDefinitions import RefCatalogStarBase
from lsst.sims.catUtils.exampleCatalogDefinitions import RefCatalogGalaxyBase, PhoSimCatalogPoint,\
                                                         PhoSimCatalogZPoint, PhoSimCatalogSersic2D
import lsst.sims.catUtils.baseCatalogModels as bcm
import sys
import glob

# Use the catsim framework to loop over a healpy map and generate a stellar density map

# Connect to fatboy with: ssh -L 51433:fatboy-private.phys.washington.edu:1433 gateway.astro.washington.edu

# Set up healpy map and ra,dec centers
nside = 64
# set the min to 15 since we saturate there. CatSim max is 28
bins = np.arange(15.,28.2,.2)
starDensity = np.zeros((hp.nside2npix(nside),np.size(bins)-1), dtype=float)
lat, ra = hp.pix2ang(nside,np.arange(0,hp.nside2npix(nside)))
dec = np.pi/2.-lat

# square root of pixel area.
hpsizeDeg = hp.nside2resol(nside, arcmin=True)/60.

# Limit things to a 1 sq degree area
hpsizeDeg = np.min([1., hpsizeDeg] )

# options include galaxyBase, cepheidstars, wdstars, rrlystars, msstars, bhbstars, allstars, and more...
dbobj = CatalogDBObject.from_objid('allstars')

indxMin = 0

restoreFile = glob.glob('starDensity_nside_%i.npz' % (nside))
if len(restoreFile) > 0:
    data = np.load(restoreFile[0])
    starDensity = data['starDensity'].copy()
    indxMin = data['icheck']


print ''
# Look at a cirular area the same area as the healpix it's centered on.
boundLength = hpsizeDeg/np.pi**0.5

blockArea = hpsizeDeg**2 # sq deg

checksize = 500
printsize = 10
npix=float(hp.nside2npix(nside))

for i in np.arange(indxMin,npix):
    lastCP = ''
    # wonder what the units of boundLength are...degrees! And it's a radius
    obs_metadata = ObservationMetaData(boundType='circle',
                                       unrefractedRA=np.degrees(ra[i]),
                                       unrefractedDec=np.degrees(dec[i]),
                                       boundLength=boundLength)
    t = dbobj.getCatalog('ref_catalog_star', obs_metadata=obs_metadata)

    # let's see if I can just querry
    chunks = t.db_obj.query_columns(colnames=['rmag'], obs_metadata=obs_metadata,
                                    constraint=None, chunk_size=10000)

    # I could think of setting the chunksize to something really large, then only doing one chunk?
    # Or maybe setting up a way to break out of the loop if everything gets really dense?
    tempHist = np.zeros(np.size(bins)-1, dtype=float)
    for chunk in chunks:
        chunkHist,bins = np.histogram(chunk['rmag'],bins)
        tempHist += chunkHist

    starDensity[i] = np.add.accumulate(tempHist)/blockArea
        #for key in densityMaps.keys():
        #    good = np.where(chunk['rmag'] < key)[0]
        #    densityMaps[key][i] += good.size/blockArea
    # if there were no stars, set to -1
    #for key in densityMaps.keys():
    #    if densityMaps[key][i] == 0:
    #        densityMaps[key][i] = -1

    # Checkpoint
    if (i % checksize == 0) & (i != 0):
        np.savez('starDensity_nside_%i.npz' % (nside) , starDensity=starDensity, bins=bins, icheck=i)
        lastCP = 'Checkpointed at i=%i of %i' % (i,npix)
    if i % printsize == 0:
        sys.stdout.write('\r')
        perComplete = float(i)/npix*100
        sys.stdout.write(r'%.2f%% complete. ' %(perComplete) + lastCP)
        sys.stdout.flush()


np.savez('starDensity_nside_%i.npz' % (nside) , starDensity=starDensity, bins=bins)
