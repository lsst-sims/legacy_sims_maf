# nd Binner slices data on N columns in simData

import numpy as np
from .baseBinner import BaseBinner

class NDBinner(BaseBinner):
    """Nd binner (N dimensions)"""
    def __init__(self, verbose=True):  
        """Instantiate object."""
        super(NDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ND'
    
    def setupBinner(self, sliceDataColList, binsList=None, nbinsList=100):
        """Set up bins.

        sliceDataColList is a list of the columns for slicing data. 
        binsList can be a list of numpy arrays with the respective binpoints for sliceDataColList,
        or can be left 'None' (default) in which case nbinsList will be used together with data 
        min/max values to set bins. """
        self.nD = len(sliceDataColList)
        self.sliceDataCols = sliceDataColList[i]
        if binsList != None:
            # User set the bins themselves.
            if len(binsList) != self.nD:
                raise Exception('binsList must be same length as sliceDataColList')
            self.bins = binsList
        else:
            # We should set the bins.
            if isinstance(nbinsList, list):
                if len(nbinsList) != self.nD:
                    raise Exception('nbinsList must be same length as sliceDataColList if providing a list')
            else: # have a number of bins, but it's just a single number to be applied to all cols
                nbinsList = [nbinsList for i in range(self.nd)]
            self.bins = []
            for i in range(self.nD):
                binsize = (self.sliceDataCols[i].max() - self.sliceDataCols[i].min()) \
                    / float(nbinsList[i])
                self.bins[i] = np.arange(self.sliceDataCols[i].min(), 
                                         self.sliceDataCols[i].max() + binsize,
                                         binsize, 'float')
        self.allbins = []
        for i in range(self.nD):
            self.allbins.append(np.meshgrid(*self.bins)[i].flatten())
        self.nbins = np.array(map(len, self.bins)).prod()
    
    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self

    def next(self):
        """Return the binvalues at this binpoint."""
        ### This is not correctly implemented yet (binpoints not being properly defined)
        if self.ipix >= self.nbins-1:
            raise StopIteration
        binlo = np.zeros(self.nD, 'float')
        binhi = np.zeros(self.nD, 'float')
        for i in range(self.nD):
            binlo[i] = self.allbins[i][self.ipix]
            binhi[i] = self.allbins[i][self.ipix+1]
        self.ipix += 1
        return (binlo, binhi)

    def __getitem__(self, ipix):
        binlo = np.zeros(self.nD, 'float')
        binhi = np.zeros(self.nD, 'float')
        for i in range(self.nD):
            binlo[i] = self.allbins[i][ipix]
            binhi[i] = self.allbins[i][ipix+1]
        return (binlo, binhi)
    
    def __eq__(self, otherBinner):
        """Evaluate if grids are equivalent."""
        if isinstance(otherBinner, NDBinner):
            if otherBinner.nD != self.nD:
                return False
            for i in range(self.nD):
                if np.all(otherBinner.bins[i] != self.bins[i]):
                    return False                
            return True
        else:
            return False
            
    def sliceSimData(self, binpoint):
        """Slice simData to return relevant indexes for binpoint."""
        condition = True
        for i in range(self.nD):
            condition = (condition & (self.simDataCols[i] >= binpoint[0][i])
                         & (self.simDataCols[i] < binpoint[1][i]))
        return condition
