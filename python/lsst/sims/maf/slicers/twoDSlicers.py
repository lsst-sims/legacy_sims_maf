import numpy as np
from .healpixSlicer import HealpixSlicer
from .opsimFieldSlicer import OpsimFieldSlicer
from .userPointsSlicer import UserPointsSlicer

__all__ = ['Healpix2dSlicer', 'Opsim2dSlicer', 'User2dSlicer']

class Healpix2dSlicer(HealpixSlicer):
    """
    Just like the healpix slicer, but to be used with metrics that return
    a vector, either histogramming or accumulating results.

    bins: For histogram vector metrics, the bins are passed through and are
          treated as bin edges (with the usual numpy/scipy conventions on open/closed bins).
          For accumulated vector metrics, the first bin element is ignored, and the rest of the
          bin values are treated as right-side limits.  For example, bins=[0, 1, 2, 3] will
          result in a histogram with 3 values (0-1, 1-2, 2-3), or accumulated values after
          1,2, and 3 nights.
    """

    def __init__ (self, bins=None, binCol='night', nside=128,
                  lonCol ='fieldRA' , latCol='fieldDec', verbose=True,
                  useCache=True, radius=1.75, leafsize=100,
                  useCamera=False, chipNames='all',
                  rotSkyPosColName='rotSkyPos', mjdColName='expMJD',**kwargs):
        """
        Set a bin that the metric will histogram or accumulate on.
        """
        super(Healpix2dSlicer,self).__init__(nside=nside, lonCol=lonCol, latCol=latCol, verbose=verbose,
                                             useCache=useCache, radius=radius, leafsize=leafsize,
                                             useCamera=useCamera,chipNames=chipNames,
                                             rotSkyPosColName=rotSkyPosColName,mjdColName=mjdColName,**kwargs)

        self._setup2d(bins, binCol)


class Opsim2dSlicer(OpsimFieldSlicer):
    """
    Just like the opsim slicer, but to be used with metrics that return a vector.
    """
    def __init__(self, bins=None, binCol='night', verbose=True, simDataFieldIDColName='fieldID',
                 simDataFieldRaColName='fieldRA', simDataFieldDecColName='fieldDec',
                 fieldIDColName='fieldID', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                 badval=-666, **kwargs):
        super(Opsim2dSlicer,self).__init__(verbose=verbose, simDataFieldIDColName=simDataFieldIDColName,
                                           simDataFieldRaColName=simDataFieldRaColName,
                                           simDataFieldDecColName=simDataFieldDecColName,
                                           fieldIDColName=fieldIDColName,fieldRaColName=fieldRaColName,
                                           fieldDecColName=fieldDecColName, badval=badval, **kwargs)
        self._setup2d(bins, binCol)

    def setupSlicer(self, simData, fieldData, maps=None):
        super(Opsim2dSlicer,self).setupSlicer(simData, fieldData, maps=maps)
        self.shape = (self.nslice, np.size(self.slicePoints['bins'])-1)
        self.spatialExtent = [simData[self.simDataFieldIDColName].min(),
                              simData[self.simDataFieldIDColName].max()]

class User2dSlicer(UserPointsSlicer):
    """
    Just like the UserPointSlicer, but to be used with metrics that return a vector.
    """
    def __init__(self, ra, dec, bins=None, binCol='night', verbose=True,
                 lonCol='fieldRA', latCol='fieldDec',
                 badval=-666, leafsize=100, radius=1.75,
                 useCamera=False, rotSkyPosColName='rotSkyPos', mjdColName='expMJD',
                 chipNames=None):

        super(User2dSlicer,self).__init__(ra,dec,verbose=verbose,
                                          lonCol=lonCol, latCol=latCol,
                                          badval=badval, radius=radius, leafsize=leafsize,
                                          useCamera=useCamera, rotSkyPosColName=rotSkyPosColName,
                                          mjdColName=mjdColName, chipNames=chipNames)
        self._setup2d(bins, binCol)
