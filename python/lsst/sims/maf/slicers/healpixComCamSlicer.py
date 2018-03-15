import numpy as np
import healpy as hp
from .healpixSlicer import HealpixSlicer
import warnings
from functools import wraps
import lsst.sims.utils as simsUtils
import matplotlib.path as mplPath
from lsst.sims.maf.utils.mafUtils import gnomonic_project_toxy


__all__ = ['HealpixComCamSlicer']


# The names of the chips in the central raft, aka, ComCam
center_raft_chips = ['R:2,2 S:0,0', 'R:2,2 S:0,1', 'R:2,2 S:0,2',
                     'R:2,2 S:1,0', 'R:2,2 S:1,1', 'R:2,2 S:1,2',
                     'R:2,2 S:2,0', 'R:2,2 S:2,1', 'R:2,2 S:2,2']


def corner_positions(ra, dec, rotation, radius=np.radians(0.495)):
    """Compute the RA,dec positions of the comcam corners. Using equations from:
    https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    ra : 
    dec :
    rotation :
    radius :
    
    """
    corner_angles = np.array([.25, .75, 1.25, 1.75])*np.pi
    corner_angles += rotation % (2.*np.pi)
    decs = np.arcsin(np.sin(ra)*np.cos(radius)+np.cos(ra)*np.sin(radius)*np.cos(corner_angles))
    # Need to check order on atan2
    ras = ra + np.arctan2(np.sin(corner_angles)*np.sin(radius)*np.cos(dec), np.cos(radius)-np.sin(dec)*np.sin(decs))
    return ras, decs


class HealpixComCamSlicer(HealpixSlicer):
    """Slicer that uses the ComCam footprint to decide if observations overlap a healpixel center
    """

    def __init__(self, nside=128, lonCol='fieldRA',
                 latCol='fieldDec', latLonDeg=True, verbose=True, badval=hp.UNSEEN,
                 useCache=True, leafsize=100, radius=0.49497,
                 useCamera=False, rotSkyPosColName='rotSkyPos',
                 mjdColName='observationStartMJD', chipNames=center_raft_chips, side_length=0.7):
        """
        Parameters
        ----------
        radius : float (0.49497)
            The radius to check for healpixels. Default set by assuming we want to include the
            corner of the raft. Given the full FoV is 5 rafts, with a radius of 1.75 degrees, the
            distance to the corner of 1 raft comes out to sqrt(2*(1.75/5)**2).
        side_length : float (0.7)
            How large is a side of the raft (degrees)
        """
        super(HealpixComCamSlicer, self).__init__(nside=nside, lonCol=lonCol, latCol=latCol,
                                                  latLonDeg=latLonDeg,
                                                  verbose=verbose, badval=badval, useCache=useCache,
                                                  leafsize=leafsize, radius=radius, useCamera=useCamera,
                                                  rotSkyPosColName=rotSkyPosColName,
                                                  mjdColName=mjdColName, chipNames=chipNames)
        self.side_length = np.radians(side_length)
        # Need the rotation even if not using the camera
        self.columnsNeeded.append(rotSkyPosColName)
        self.columnsNeeded = list(set(self.columnsNeeded))

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
                initial_indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
                # XXX I don't have to check if the radius is small enough, only the border cases

                indices = []
                if self.latLonDeg:
                    lat = np.radians(simData[self.latCol][initial_indices])
                    lon = np.radians(simData[self.lonCol][initial_indices])
                    rotSky_rad = np.radians(simData[self.rotSkyPosColName][initial_indices])
                else:
                    lat = simData[self.latCol][initial_indices]
                    lon = simData[self.lonCol][initial_indices]
                    rotSky_rad = simData[self.rotSkyPosColName][initial_indices]
                for i, ind in enumerate(initial_indices):
                    corner_ra, corner_dec = corner_positions(lon[i], lat[i], rotSky_rad[i], radius=self.side_length/np.sqrt(2.))
                    # Project to plane
                    corner_x, corner_y = gnomonic_project_toxy(corner_ra, corner_dec,
                                                               self.slicePoints['ra'][islice],
                                                               self.slicePoints['dec'][islice])
                    # Use matplotlib to make a polygon
                    bbPath = mplPath.Path(np.array([[corner_x[0], corner_y[0]],
                                                   [corner_x[1], corner_y[1]],
                                                   [corner_x[2], corner_y[2]],
                                                   [corner_x[3], corner_y[3]],
                                                   [corner_x[0], corner_y[0]]]))
                    # Check if the slicepoint is inside the image corners and append to list if it is
                    if bbPath.contains_point((0., 0.)) == 1:
                        indices.append(ind)

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
