import numpy as np
import healpy as hp
from .healpixSlicer import HealpixSlicer
import warnings
from functools import wraps
import lsst.sims.utils as simsUtils


__all__ = ['HealpixComCamSlicer']


# The names of the chips in the central raft, aka, ComCam
center_raft_chips = ['R:2,2 S:0,0', 'R:2,2 S:0,1', 'R:2,2 S:0,2'
                     'R:2,2 S:1,0', 'R:2,2 S:1,1', 'R:2,2 S:1,2'
                     'R:2,2 S:2,0', 'R:2,2 S:2,1', 'R:2,2 S:2,2']

class HealpixComCamSlicer(HealpixSlicer):
    """Slicer that uses the ComCam footprint to decide if observations overlap a healpixel center
    """

    def __init__(self, nside=128, lonCol ='fieldRA',
                 latCol='fieldDec', latLonDeg=True, verbose=True, badval=hp.UNSEEN,
                 useCache=True, leafsize=100, radius=0.35,
                 useCamera=False, rotSkyPosColName='rotSkyPos',
                 mjdColName='observationStartMJD', chipNames=center_raft_chips):
        super(HealpixComCamSlicer, self).__init__(nside=nside, lonCol=lonCol, latCol=latCol,
                                                  latLonDeg=latLonDeg,
                                                  verbose=verbose, badval=badval, useCache=useCache,
                                                  leafsize=leafsize, radius=radius, useCamera=useCamera,
                                                  rotSkyPosColName=rotSkyPosColName,
                                                  mjdColName=mjdColName, chipNames=chipNames)

    def setupSlicer(self, simData, maps=None):
        """Use simData[self.lonCol] and simData[self.latCol] (in radians) to set up KDTree.

        Parameters
        -----------
        simData : numpy.recarray
            The simulated data, including the location of each pointing.
        maps : list of lsst.sims.maf.maps objects, optional
            List of maps (such as dust extinction) that will run to build up additional metadata at each
            slicePoint. This additional metadata is available to metrics via the slicePoint dictionary.
            Default None.
        """
        if maps is not None:
            if self.cacheSize != 0 and len(maps) > 0:
                warnings.warn('Warning:  Loading maps but cache on.'
                              'Should probably set useCache=False in slicer.')
            self._runMaps(maps)
        self._setRad(self.radius)
        if self.useCamera:
            self._setupLSSTCamera()
            self._presliceFootprint(simData)
        else:
            if self.latLonDeg:
                self._buildTree(np.radians(simData[self.lonCol]),
                                np.radians(simData[self.latCol]), self.leafsize)
            else:
                self._buildTree(simData[self.lonCol], simData[self.latCol], self.leafsize)

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=lonCol/latCol value .. usually ra/dec)."""

            # Build dict for slicePoint info
            slicePoint = {}
            if self.useCamera:
                indices = self.sliceLookup[islice]
                slicePoint['chipNames'] = self.chipNames[islice]
            else:
                sx, sy, sz = simsUtils._xyz_from_ra_dec(self.slicePoints['ra'][islice],
                                                        self.slicePoints['dec'][islice])
                # Query against tree.
                indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
                xxx -- these are the initial indices. Need to draw polygon and check which are inside

            # Loop through all the slicePoint keys. If the first dimension of slicepoint[key] has
            # the same shape as the slicer, assume it is information per slicepoint.
            # Otherwise, pass the whole slicePoint[key] information. Useful for stellar LF maps
            # where we want to pass only the relevant LF and the bins that go with it.
            for key in self.slicePoints:
                if len(np.shape(self.slicePoints[key])) == 0:
                    keyShape = 0
                else:
                    keyShape = np.shape(self.slicePoints[key])[0]
                if (keyShape == self.nslice):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs': indices, 'slicePoint': slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)