# Class for HealpixSlicer (healpixel-based spatial slicer).
# User can select resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import warnings
import healpy as hp

from lsst.sims.maf.utils import percentileClipping
from lsst.sims.maf.plots.spatialPlotters import HealpixSkyMap, HealpixHistogram, HealpixPowerSpectrum

from .baseSpatialSlicer import BaseSpatialSlicer


__all__ = ['HealpixSlicer']

class HealpixSlicer(BaseSpatialSlicer):
    """Healpix spatial slicer."""
    def __init__(self, nside=128, lonCol ='fieldRA' , latCol='fieldDec', verbose=True,
                 useCache=True, radius=1.75, leafsize=100, 
                 useCamera=False, rotSkyPosColName='rotSkyPos', mjdColName='expMJD'):
        """Instantiate and set up healpix slicer object."""
        super(HealpixSlicer, self).__init__(verbose=verbose,
                                            lonCol=lonCol, latCol=latCol,
                                            badval=hp.UNSEEN, radius=radius, leafsize=leafsize,
                                            useCamera=useCamera, rotSkyPosColName=rotSkyPosColName,
                                            mjdColName=mjdColName)
        # Valid values of nside are powers of 2.
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise ValueError('Valid values of nside are powers of 2.')
        self.nside = int(nside)
        self.pixArea = hp.nside2pixarea(self.nside)
        self.nslice = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Healpix slicer using NSIDE=%d, '%(self.nside) + \
            'approximate resolution %f arcminutes'%(hp.nside2resol(self.nside,arcmin=True))
        # Set variables so slicer can be re-constructed
        self.slicer_init = {'nside':nside, 'lonCol':lonCol, 'latCol':latCol,
                            'radius':radius}
        if useCache:
            # useCache set the size of the cache for the memoize function in sliceMetric.
            binRes = hp.nside2resol(nside) # Pixel size in radians
            # Set the cache size to be ~2x the circumference
            self.cacheSize = int(np.round(4.*np.pi/binRes))
        # Set up slicePoint metadata.
        self.slicePoints['nside'] = nside
        self.slicePoints['sid'] = np.arange(self.nslice)
        self.slicePoints['ra'], self.slicePoints['dec'] = self._pix2radec(self.slicePoints['sid'])
        # Set the default plotting functions.
        self.plotFuncs = [HealpixSkyMap, HealpixHistogram, HealpixPowerSpectrum]

    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value.
        if isinstance(otherSlicer, HealpixSlicer):
            if otherSlicer.nside == self.nside:
                if (otherSlicer.lonCol == self.lonCol and otherSlicer.latCol == self.latCol):
                    if otherSlicer.radius == self.radius:
                        return True
        else:
            return False

    def _pix2radec(self, islice):
        """Given the pixel number / sliceID, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix slicer.
        # Note that ipix could be an array,
        # in which case RA/Dec values will be an array also.
        lat, ra = hp.pix2ang(self.nside, islice)
        # Move dec to +/- 90 degrees
        dec = np.pi/2.0 - lat
        return ra, dec
