# Class for opsim field based binner.

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pyf
from .baseBinner import BaseBinner

from .baseSpatialBinner import BaseSpatialBinner

class OpsimFieldBinner(BaseSpatialBinner):
    """Opsim Field based binner."""
    def __init__(self, verbose=True):
        super(opsimFieldBinner, self).__init__(verbose=verbose)
        self.binnertype = 'OPSIMFIELDS'

    def setupBinner(self, simData, simFIdColName, 
                    fieldData, fieldFIdColName, fieldRaColName, fieldDecColName):
        """Set up opsim field binner object.

        simData = numpy rec array with simulation pointing history,
        simFIdColName = the column name of the fieldIDs (in simData)
        fieldData = numpy recarray with fieldID information
        fieldFIdColName = the column name with the fieldIDs (in fieldData)
        fieldRaColName = the column name with the RA values (in fieldData)
        fieldDecColname = the column name with the Dec values (in fieldData)."""
        # Set basic properties for tracking field information, in sorted order.
        idxs = np.argsort(fieldData[fieldFIdColName])
        self.fieldId = fieldData[fieldFIdColName][idxs]
        self.ra = fieldData[fieldRaColName][idxs]
        self.dec = fieldData[fieldDecColName][idxs]
        self.nbins = len(self.fieldID)
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[simFIdColName])
        simFieldsSorted = np.sort(simData[simFIdColName])
        self.left = np.searchsorted(simFieldsSorted, self.fieldId, 'left')
        self.right = np.searchsorted(simFieldsSorted, self.fieldId, 'right')        

    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self
    
    def next(self):
        """Return RA/Dec values when iterating over binpoints."""
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.npix:
            raise StopIteration
        fieldidradec = self.fieldid[self.ipix], self.ra[self.ipix], self.dec[self.ipix]
        self.ipix += 1
        return fieldidradec

    def __getitem__(self, ipix):
        fieldidradec = self.fieldid[ipix], self.ra[self.ipix], self.dec[self.ipix]
        return fieldidradec
    
    def __eq__(self, otherBinner):
        """Evaluate if two grids are equivalent."""
        if isinstance(otherBinner, OpsimFieldBinner):
            return ((np.all(otherBinner.ra == self.ra)) 
                    and (np.all(otherBinner.dec == self.dec)))
        else:
            return False

    def sliceSimData(self, binpoint):
        """Slice simData on fieldID, to return relevant indexes for binpoint."""
        i = np.where(binpoint[0] == self.fieldId)
        return self.simIdxs[self.left[i]:self.right[i]]   

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """
        header_dict = dict(comment=comment, metricName=metricName,
                           simDataName=simDataName,
                           metadata=metadata, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval)
        base = BaseBinner()
        base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        #update the header
        hdulist = pyf.open(outfilename, mode='update')
        for key in header_dict.keys():
            hdulist[0].header[key] = header_dict[key]
        hdulist.close()

        #append the bins
        hdulist = pyf.open(outfilename, mode='append')
        raHDU = pyf.PrimaryHDU(data=self.ra)
        decHDU =  pyf.PrimaryHDU(data=self.dec)
        hdulist.append(raHDU)
        hdulist.append(decHDU)
        hdulist.flush()
        hdulist.close()
        
        return outfilename

    def readMetricData(self, infilename):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        
        ra = hdulist[1].data.copy()
        dec = hdulist[2].data.copy()        
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)
        
        binner = OpsimFieldBinner(ra, dec)
       
        binner.badval = header['badval'.upper()]
        binner.int_badval = header['int_badval']
                
        return metricValues, binner,header
