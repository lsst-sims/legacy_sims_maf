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
#nside = 8
# set the min to 15 since we saturate there. CatSim max is 28
bins = np.arange(15.,28.2,.2)
starDensity = np.zeros((hp.nside2npix(nside),np.size(bins)-1), dtype=float)
overMaxMask = np.zeros(hp.nside2npix(nside), dtype=bool)
lat, ra = hp.pix2ang(nside,np.arange(0,hp.nside2npix(nside)))
dec = np.pi/2.-lat

filterName = 'r'
colName = 'rmag'

# square root of pixel area.
hpsizeDeg = hp.nside2resol(nside, arcmin=True)/60.

# Limit things to a 10 arcmin radius
hpsizeDeg = np.min([10./60., hpsizeDeg] )
#import pdb ; pdb.set_trace()

# options include galaxyBase, cepheidstars, wdstars, rrlystars, msstars, bhbstars, allstars, and more...
dbobj = CatalogDBObject.from_objid('allstars')

indxMin = 0

restoreFile = glob.glob('starDensity_%s_nside_%i.npz' % (filterName,nside))
if len(restoreFile) > 0:
    data = np.load(restoreFile[0])
    starDensity = data['starDensity'].copy()
    indxMin = data['icheck'].copy()
    overMaxMask = data['overMaxMask'].copy()


print ''
# Look at a cirular area the same area as the healpix it's centered on.
boundLength = hpsizeDeg/np.pi**0.5

blockArea = hpsizeDeg**2 # sq deg

checksize = 1000
printsize = 10
npix=float(hp.nside2npix(nside))

# If the area has more than this number of objects, flag it as a max
breakLimit = 1e6
chunk_size=10000
for i in np.arange(indxMin,npix):
    lastCP = ''
    # wonder what the units of boundLength are...degrees! And it's a radius
    obs_metadata = ObservationMetaData(boundType='circle',
                                       pointingRA=np.degrees(ra[i]),
                                       pointingDec=np.degrees(dec[i]),
                                       boundLength=boundLength, mjd=5700)
    t = dbobj.getCatalog('ref_catalog_star', obs_metadata=obs_metadata)

    # let's see if I can just querry
    chunks = t.db_obj.query_columns(colnames=[colName], obs_metadata=obs_metadata,
                                    constraint=None, chunk_size=chunk_size)

    # I could think of setting the chunksize to something really large, then only doing one chunk?
    # Or maybe setting up a way to break out of the loop if everything gets really dense?
    tempHist = np.zeros(np.size(bins)-1, dtype=float)
    counter = 0
    for chunk in chunks:
        chunkHist,bins = np.histogram(chunk[colName],bins)
        tempHist += chunkHist
        counter += chunk_size
        if counter >= breakLimit:
            overMaxMask[i] = True
            break

    starDensity[i] = np.add.accumulate(tempHist)/blockArea

    # Checkpoint
    if (i % checksize == 0) & (i != 0):
        np.savez('starDensity_%s_nside_%i.npz' % (filterName,nside),
                 starDensity=starDensity, bins=bins, icheck=i, overMaxMask=overMaxMask)
        lastCP = 'Checkpointed at i=%i of %i' % (i,npix)
    if i % printsize == 0:
        sys.stdout.write('\r')
        perComplete = float(i)/npix*100
        sys.stdout.write(r'%.2f%% complete. ' %(perComplete) + lastCP)
        sys.stdout.flush()


np.savez('starDensity_%s_nside_%i.npz' % (filterName,nside), starDensity=starDensity,
         bins=bins,overMaxMask=overMaxMask )
print ''
print 'Completed!'
