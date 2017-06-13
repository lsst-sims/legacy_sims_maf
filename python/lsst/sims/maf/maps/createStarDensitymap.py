#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from lsst.sims.utils import ObservationMetaData
import healpy as hp
import sys
import glob
import argparse

# Use the catsim framework to loop over a healpy map and generate a stellar density map

# Connect to fatboy with: ssh -L 51433:fatboy.phys.washington.edu:1433 gateway.astro.washington.edu
# If non-astro user, use simsuser@gateway.astro.washington.edu

if __name__ == '__main__':

    # Hide imports here so documentation builds
    from lsst.sims.catalogs.db import CatalogDBObject
    # Import the bits needed to get the catalog to work
    from lsst.sims.catUtils.baseCatalogModels import *
    from lsst.sims.catUtils.exampleCatalogDefinitions import *


    parser = argparse.ArgumentParser(description="Build a stellar density healpix map")
    parser.add_argument("--filtername", type=str, default='r', help="which filter: u, g, r, i, z, y")
    parser.add_argument("--stars", type=str, default='allstars', help="the stellar type to pull from CatSim")
    parser.add_argument("--mag_min", type=flaot, default=15., help="How bright to go")

    args = parser.parse_args()

    if args.stars == 'allstars':
        starNames = ''
    else:
        starNames = args.stars+'_'

    filterName = args.filtername
    colName = filterName+'mag'
    mag_min = args.mag_min

    # Set up healpy map and ra, dec centers
    nside = 64

    # Set the min to 15 by default since we saturate there. CatSim max is 28
    bins = np.arange(mag_min, 28.2, .2)
    starDensity = np.zeros((hp.nside2npix(nside), np.size(bins)-1), dtype=float)
    overMaxMask = np.zeros(hp.nside2npix(nside), dtype=bool)
    lat, ra = hp.pix2ang(nside, np.arange(0, hp.nside2npix(nside)))
    dec = np.pi/2.-lat

    # Square root of pixel area.
    hpsizeDeg = hp.nside2resol(nside, arcmin=True)/60.

    # Limit things to a 10 arcmin radius
    hpsizeDeg = np.min([10./60., hpsizeDeg])

    # Options include galaxyBase, cepheidstars, wdstars, rrlystars, msstars, bhbstars, allstars, and more...
    dbobj = CatalogDBObject.from_objid(args.stars)

    indxMin = 0

    restoreFile = glob.glob('starDensity_%s_%s_nside_%i.npz' % (filterName, starNames, nside))
    if len(restoreFile) > 0:
        data = np.load(restoreFile[0])
        starDensity = data['starDensity'].copy()
        indxMin = data['icheck'].copy()
        overMaxMask = data['overMaxMask'].copy()

    print('')
    # Look at a cirular area the same area as the healpix it's centered on.
    boundLength = hpsizeDeg/np.pi**0.5

    blockArea = hpsizeDeg**2  # sq deg

    checksize = 1000
    printsize = 10
    npix = float(hp.nside2npix(nside))

    # If the area has more than this number of objects, flag it as a max
    breakLimit = 1e6
    chunk_size = 10000
    for i in np.arange(indxMin, npix):
        lastCP = ''
        # wonder what the units of boundLength are...degrees! And it's a radius
        # The newer interface:
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
            chunkHist, bins = np.histogram(chunk[colName], bins)
            tempHist += chunkHist
            counter += chunk_size
            if counter >= breakLimit:
                overMaxMask[i] = True
                break

        starDensity[i] = np.add.accumulate(tempHist)/blockArea

        # Checkpoint
        if (i % checksize == 0) & (i != 0):
            np.savez('starDensity_%s_%snside_%i.npz' % (filterName, starNames, nside),
                     starDensity=starDensity, bins=bins, icheck=i, overMaxMask=overMaxMask)
            lastCP = 'Checkpointed at i=%i of %i' % (i, npix)
        if i % printsize == 0:
            sys.stdout.write('\r')
            perComplete = float(i) / npix * 100
            sys.stdout.write(r'%.2f%% complete. ' % (perComplete) + lastCP)
            sys.stdout.flush()

    np.savez('starDensity_%s_%snside_%i.npz' % (filterName, starNames, nside), starDensity=starDensity,
             bins=bins, overMaxMask=overMaxMask)
    print('')
    print('Completed!')
