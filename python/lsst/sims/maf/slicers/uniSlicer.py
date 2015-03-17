# UniSlicer class.
# This slicer simply returns the indexes of all data points. No slicing done at all.

import numpy as np
from functools import wraps

from .baseSlicer import BaseSlicer

__all__ = ['UniSlicer']

class UniSlicer(BaseSlicer):
    """UniSlicer."""
    def __init__(self, verbose=True, badval=-666, plotFuncs=None):
        """Instantiate unislicer. """
        super(UniSlicer, self).__init__(verbose=verbose, badval=badval,
                                        plotFuncs=plotFuncs)
        self.nslice = 1
        self.slicePoints['sid'] = np.array([0,], int)

    def setupSlicer(self, simData):
        """Use simData to set indexes to return."""
        simDataCol = simData.dtype.names[0]
        self.indices = np.ones(len(simData[simDataCol]),  dtype='bool')
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return all indexes in simData. """
            idxs = self.indices
            return {'idxs':idxs,
                    'slicePoint':{'sid':islice}}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(otherSlicer, UniSlicer):
            return True
        else:
            return False
