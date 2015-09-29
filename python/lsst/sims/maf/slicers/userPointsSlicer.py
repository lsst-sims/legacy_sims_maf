import numpy as np
from lsst.sims.maf.plots.spatialPlotters import BaseSkyMap, BaseHistogram
from .baseSpatialSlicer import BaseSpatialSlicer

__all__ = ['UserPointsSlicer']

class UserPointsSlicer(BaseSpatialSlicer):
    """Use spatial slicer on a user provided point """
    def __init__(self, ra, dec, bins=None, binCol='night', verbose=True, lonCol='fieldRA', latCol='fieldDec',
                 badval=-666, leafsize=100, radius=1.75,
                 useCamera=False, rotSkyPosColName='rotSkyPos', mjdColName='expMJD',
                 chipNames=None):
        """
        ra = list of ra points to use
        dec = list of dec points to use
        """

        super(UserPointsSlicer,self).__init__(verbose=verbose,bins=bins, binCol=binCol,
                                                lonCol=lonCol, latCol=latCol,
                                                badval=badval, radius=radius, leafsize=leafsize,
                                                useCamera=useCamera, rotSkyPosColName=rotSkyPosColName,
                                                mjdColName=mjdColName, chipNames=chipNames)

        # check that ra and dec are iterable, if not, they are probably naked numbers, wrap in list
        if not hasattr(ra, '__iter__'):
            ra = [ra]
        if not hasattr(dec, '__iter__'):
            dec = [dec]
        if len(ra) != len(dec):
            raise ValueError('RA and Dec must be the same length')
        self.nslice = np.size(ra)
        self.shape = self.nslice
        self.slicePoints['sid'] = np.arange(np.size(ra))
        self.slicePoints['ra'] = np.array(ra)
        self.slicePoints['dec'] = np.array(dec)
        self.plotFuncs = [BaseSkyMap, BaseHistogram]
        if bins is not None:
            self._setup2d(bins, binCol)

    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        result = False
        if isinstance(otherSlicer, UserPointsSlicer):
            if otherSlicer.nslice == self.nslice:
                if np.all(otherSlicer.ra == self.ra) and np.all(otherSlicer.dec == self.dec):
                    if (otherSlicer.lonCol == self.lonCol and otherSlicer.latCol == self.latCol):
                        if otherSlicer.radius == self.radius:
                            if otherSlicer.useCamera == self.useCamera:
                                if otherSlicer.chipsToUse == self.chipsToUse:
                                    if otherSlicer.rotSkyPosColName == self.rotSkyPosColName:
                                        if np.all(otherSlicer.shape == self.shape):
                                            result = True
        return result
