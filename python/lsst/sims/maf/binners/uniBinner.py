# UniBinner class.
# This binner simply returns the indexes of all data points. No slicing done at all.

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

from .baseBinner import BaseBinner

class UniBinner(BaseBinner):
    """UniBinner."""
    def __init__(self, verbose=True, **kwargs):
        """Instantiate unibinner. """
        super(UniBinner, self).__init__(verbose=verbose, **kwargs)
        self.nbins = 1
        self.bins = None

    def setupBinner(self, simData):
        """Use simData to set indexes to return."""
        simDataCol = simData.dtype.names[0]
        self.indices = np.ones(len(simData[simDataCol]),  dtype='bool')
        # Build sliceSimData method here.
        @wraps(self.sliceSimData)
        def sliceSimData(binpoint):
            """Return all indexes in simData. """
            return self.indices
        setattr(self, 'sliceSimData', sliceSimData)
        
    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self

    def next(self):
        """Set the binpoints to return when iterating over binner."""
        if self.ipix >= self.nbins:
            raise StopIteration
        ipix = self.ipix
        self.ipix += 1
        return ipix

    def __getitem__(self, ipix):
        return ipix
    
    def __eq__(self, otherBinner):
        """Evaluate if binners are equivalent."""
        if isinstance(otherBinner, UniBinner):
            return True
        else:
            return False
            
    def plotData(self, metricValues, figformat='png', filename=None,
                 savefig=True, **kwargs):
        """Override plotData, to be sure it returns None (no plots available)."""

        super(UniBinner, self).plotData(metricValues, 
                                        figformat=figformat, 
                                        filename=filename, savefig=savefig, **kwargs)
            
        return None
