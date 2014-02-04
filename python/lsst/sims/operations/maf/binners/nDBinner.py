# nd Binner slices data on N columns in simData

import numpy as np
from .baseBinner import BaseBinner
import pyfits as pyf

class NDBinner(BaseBinner):
    """Nd binner (N dimensions)"""
    def __init__(self, verbose=True):  
        """Instantiate object."""
        super(NDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ND'
    
    def setupBinner(self, data, sliceDataColList=None, binsList=None, nbinsList=100):
        """Set up bins.

        data is the input data. Does not need to be passed, but keeps API same for all binners.
        sliceDataColList is a list of the columns for slicing data. 
        binsList can be a list of numpy arrays with the respective binpoints for sliceDataColList,
        or can be left 'None' (default) in which case nbinsList will be used together with data 
        min/max values to set bins. """
        if sliceDataColList == None:
            sliceDataColList = data.keys()
        self.nD = len(sliceDataColList)
        self.sliceDataCols = sliceDataColList
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
                nbinsList = [nbinsList for i in range(self.nD)]
            self.bins = [ [] for i in range(self.nD)]
            for i in range(self.nD):
                binsize = (self.sliceDataCols[i].max() - self.sliceDataCols[i].min()) \
                    / float(nbinsList[i])
                self.bins[i] = np.arange(self.sliceDataCols[i].min(), 
                                         self.sliceDataCols[i].max() + binsize,
                                         binsize, 'float')
        _setupAllbins()

    def _setupAllbins(self):
        self.allbins = []
        for i in range(self.nD):
            self.allbins.append(np.meshgrid(*self.bins)[i].flatten())
        self.nbins = np.array(map(len, self.bins)).prod()
        return
    
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

