# UniSlicer class.
# This slicer simply returns the indexes of all data points. No slicing done at all.

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

from .baseSlicer import BaseSlicer

class UniSlicer(BaseSlicer):
    """UniSlicer."""
    def __init__(self, verbose=True, **kwargs):
        """Instantiate unislicer. """
        super(UniSlicer, self).__init__(verbose=verbose, **kwargs)
        self.nbins = 1
        self.bins = None

    def setupSlicer(self, simData):
        """Use simData to set indexes to return."""
        simDataCol = simData.dtype.names[0]
        self.indices = np.ones(len(simData[simDataCol]),  dtype='bool')
                
    def _sliceSimData(self, ipix):
        """Return all indexes in simData. """
        idxs = self.indices
        slicePoint={'pid':ipix}
        return {'idxs':idxs, 'slicePoint':slicePoint}

    def __iter__(self):
        """Iterate over the slicepoints."""
        self.ipix = 0
        return self

    def next(self):
        """Set the slicepoints to return when iterating over slicer."""
        if self.ipix >= self.nbins:
            raise StopIteration
        ipix = self.ipix
        self.ipix += 1
        return self._sliceSimData(ipix)

    def __getitem__(self, ipix):
        return self._sliceSimData(ipix)
    
    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(otherSlicer, UniSlicer):
            return True
        else:
            return False
            
