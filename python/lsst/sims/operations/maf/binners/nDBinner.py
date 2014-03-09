# nd Binner slices data on N columns in simData

import numpy as np
import matplotlib.pyplot as plt
import itertools
try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf

from .baseBinner import BaseBinner

    
class NDBinner(BaseBinner):
    """Nd binner (N dimensions)"""
    def __init__(self, sliceDataColList=None, verbose=True):  
        """Instantiate object."""
        super(NDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ND'
        self.bins = None 
        self.nbins = None
        self.sliceDataColList = sliceDataColList
        self.columnsNeeded = self.sliceDataColList
        if self.sliceDataColList != None:
            self.nD = len(self.sliceDataColList)
        else:
            self.nD = None

    def setupBinner(self, simData, binsList=None, nbinsList=100):
        """Set up bins.

        binsList can be a list of numpy arrays with the respective binpoints for sliceDataColList,
            (default 'None' uses nbinsList together with data min/max values to set bins). """
        # Parse input bins choices.
        if binsList != None:
            if len(binsList) != self.nD:
                raise Exception('BinsList must be same length as sliceDataColNames')
            self.bins = binsList
            for b in self.bins:
                b = np.sort(b)
        else:
            if isinstance(nbinsList, list):
                if len(nbinsList) != self.nD:
                        raise Exception('nbinsList must be same length as sliceDataColList')
            else:  # we have an nbins but it's a single number to apply to all cols
                nbinsList = [nbinsList for i in range(self.nD)]
            # Set the bins.
            self.bins = []
            for sliceColName, nbins in zip(self.sliceDataColList, nbinsList):
                sliceDataCol = simData[sliceColName]
                binsize = (sliceDataCol.max() - sliceDataCol.min()) / float(nbins)
                bins = np.arange(sliceDataCol.min(), sliceDataCol.max() + binsize/2.0,
                                 binsize, 'float')
                self.bins.append(bins)
        # Count how many bins we have total (not counting last 'RHS' bin values, as in oneDBinner).
        self.nbins = (np.array(map(len, self.bins))-1).prod()
        # Set up data slicing.
        self.simIdxs = []
        self.lefts = []
        for sliceColName, bins in zip(self.sliceDataColList, self.bins):
            simIdxs = np.argsort(simData[sliceColName])
            simFieldsSorted = np.sort(simData[sliceColName])
            # "left" values are location where simdata == bin value
            left = np.searchsorted(simFieldsSorted, bins[:-1], 'left')
            left = np.concatenate((left, np.array([len(simIdxs),])))
            # Add these calculated values into the class lists of simIdxs and lefts.
            self.simIdxs.append(simIdxs)
            self.lefts.append(left)
            
    def __iter__(self):
        """Iterate over the binpoints."""
        # Order of iteration over bins: go through bins in each sliceCol in the sliceColList in order.
        self.ipix = 0
        binsForIteration = []
        for b in self.bins:
            binsForIteration.append(b[:-1])
        self.biniterator = itertools.product(*binsForIteration)
        return self

    def next(self):
        """Return the binvalues at this binpoint."""
        if self.ipix >= self.nbins:
            raise StopIteration
        binlo = self.biniterator.next()
        self.ipix += 1        
        return binlo

    def __getitem__(self, ipix):
        # There's probably a better way to do this.
        binsForIteration = []
        for b in self.bins:
            binsForIteration.append(b[:-1])
        biniterator = itertools.product(*binsForIteration)
        for i, b in zip(range(ipix), biniterator):
            pass
        return b
    
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
        # Identify relevant pointings in each dimension.
        simIdxsList = []
        for d in range(self.nD):
            i = (np.where(binpoint[d] == self.bins[d]))[0]
            simIdxsList.append(set(self.simIdxs[d][self.lefts[d][i]:self.lefts[d][i+1]]))
        return list(set.intersection(*simIdxsList))
    
    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """

        header_dict = dict(comment=comment, metricName=metricName, simDataName=simDataName,
                           metadata=metadata, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval, nD=self.nD)
        base = BaseBinner()
        base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        hdulist = pyf.open(outfilename, mode='update')
        for key in header_dict.keys():
            hdulist[0].header[key] = header_dict[key]
        hdulist.close()
        #now to append the bins
        hdulist = pyf.open(outfilename,mode='append')
        binHDU = pyf.PrimaryHDU(data=self.bins)
        hdulist.append(binHDU)
        hdulist.flush()
        hdulist.close()
        return outfilename

    def readMetricData(self, infilename):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        
        self.bins = hdulist[1].data.copy()
        
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)
        
        binner = NDBinner()
        binner.bins = self.bins
        binner.badval = header['badval'.upper()]
        binner.int_badval = header['int_badval']
        binner.nD = header['ND']
        _setupAllbins()
        
        return metricValues, binner, header

