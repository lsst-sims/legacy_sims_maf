# Class for opsim field based binner.

import numpy as np
import matplotlib.pyplot as plt

from .baseSpatialBinner import BaseSpatialBinner

class opsimFieldBinner(BaseSpatialBinner):
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
        if isinstance(otherBinner, opsimFieldBinner):
            return ((np.all(otherBinner.ra == self.ra)) 
                    and (np.all(otherBinner.dec) == self.dec))
        else:
            return False

    def sliceSimData(self, binpoint):
        """Slice simData on fieldID, to return relevant indexes for binpoint."""
        i = np.where(binpoint[0] == self.fieldId)
        return self.simIdxs[self.left[i]:self.right[i]]

   

