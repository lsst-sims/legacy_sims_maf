import numpy as np
from lsst.sims.maf.plots.spatialPlotters import BaseSkyMap
from .baseSpatialSlicer import BaseSpatialSlicer

__all__ = ['UserPointsSlicer']

class UserPointsSlicer(BaseSpatialSlicer):
    """Use spatial slicer on a user provided point """
    def __init__(self, verbose=True, lonCol='fieldRA', latCol='fieldDec',
                 badval=-666, leafsize=100, radius=1.75, ra=None, dec=None):
        """
        ra = list of ra points to use
        dec = list of dec points to use
        """

        super(UserPointsSlicer,self).__init__(verbose=verbose,
                                                lonCol=lonCol, latCol=latCol,
                                                badval=badval, radius=radius, leafsize=leafsize)

        # check that ra and dec are iterable, if not, they are probably naked numbers, wrap in list
        if not hasattr(ra, '__iter__'):
            ra = [ra]
        if not hasattr(dec, '__iter__'):
            dec = [dec]
        self.nslice = np.size(ra)
        self.slicePoints['sid'] = np.arange(np.size(ra))
        self.slicePoints['ra'] = np.array(ra)
        self.slicePoints['dec'] = np.array(dec)
        self.plotFuncs = [BaseSkyMap,]
