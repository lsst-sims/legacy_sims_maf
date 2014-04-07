# UniBinner class.
# This binner simply returns the indexes of all data points. No slicing done at all.

import numpy as np
import matplotlib.pyplot as plt
import warnings
import pyfits as pyf


from .baseBinner import BaseBinner

class UniBinner(BaseBinner):
    """UniBinner."""
    def __init__(self, verbose=True):
        """Instantiate unibinner. """
        super(UniBinner, self).__init__(verbose=verbose)
        self.binnertype = 'UNI'
        self.nbins = 1
        self.binnerName='UniBinner'
        self.bins=np.array([0.])

    def setupBinner(self, simData):
        """Use simData to set indexes to return."""
        simDataCol = simData.dtype.names[0]
        self.indices = np.ones(len(simData[simDataCol]),  dtype='bool')
        
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
            
    def sliceSimData(self, binpoint):
        """Return all indexes in simData. """
        return self.indices

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """

        header_dict = dict(comment=comment, metricName=metricName, simDataName=simDataName,
                           metadata=metadata, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval)
        base=BaseBinner()
        base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        #update the header
        hdulist = pyf.open(outfilename, mode='update')
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key in header_dict.keys():
                    hdulist[0].header[key] = header_dict[key]
        hdulist.close()
        return outfilename

    def readMetricData(self, infilename, verbose=False):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)
        
        binner = UniBinner()
        binner.badval = header['badval'.upper()]
        binner.int_badval = header['int_badval']
                
        return metricValues, binner, header
