
import numpy as np
from .baseSpatialSlicer import BaseSpatialSlicer

class UserPointsSlicer(BaseSpatialSlicer):
    """Use spatial slicer on a user provided point """
    def __init__(self, verbose=True, spatialkey1='fieldRA', spatialkey2='fieldDec',
                 badval=-666, leafsize=100, radius=1.75, plotFuncs='all', ra=None, dec=None):
        super(UserPointsSlicer,self).__init__(verbose=verbose,
                                            spatialkey1=spatialkey1, spatialkey2=spatialkey2,
                                            badval=badval, radius=radius, leafsize=leafsize)
        self.slicePoints['sid'] = np.arange(np.size(ra))
        self.slicePoints['ra'] = ra
        self.slicePoints['dec'] = dec
