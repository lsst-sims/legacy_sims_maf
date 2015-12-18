# Class for opsim field based slicer.

import numpy as np
from functools import wraps
import warnings
from lsst.sims.maf.plots.spatialPlotters import OpsimHistogram, BaseSkyMap

from .baseSpatialSlicer import BaseSpatialSlicer

__all__ = ['OpsimFieldSlicer']

class OpsimFieldSlicer(BaseSpatialSlicer):
    """Index-based slicer, matched ID's between simData and fieldData.

    Slicer uses fieldData RA and Dec values to do sky map plotting, but could be used more
    generally for any kind of data slicing where the match is based on a simple ID value.

    Note that this slicer uses the fieldID of the opsim fields to generate spatial matches,
    thus this slicer is not suitable for use in evaluating dithering or high resolution metrics
    (use the healpix slicer instead for those use-cases). """

    def __init__(self, verbose=True, simDataFieldIDColName='fieldID',
                 simDataFieldRaColName='fieldRA', simDataFieldDecColName='fieldDec',
                 fieldIDColName='fieldID', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                 badval=-666):
        """Instantiate opsim field slicer (an index-based slicer that can do spatial plots).

        simDataFieldIDColName = the column name in simData for the field ID
        simDataFieldRaColName = the column name in simData for the field RA
        simDataFieldDecColName = the column name in simData for the field Dec
        fieldIDcolName = the column name in the fieldData for the field ID (to match with simData)
        fieldRaColName = the column name in the fieldData for the field RA (for plotting only)
        fieldDecColName = the column name in the fieldData for the field Dec (for plotting only).
        """
        super(OpsimFieldSlicer, self).__init__(verbose=verbose, badval=badval)
        self.fieldID = None
        self.simDataFieldIDColName = simDataFieldIDColName
        self.fieldIDColName = fieldIDColName
        self.fieldRaColName = fieldRaColName
        self.fieldDecColName = fieldDecColName
        self.columnsNeeded = [simDataFieldIDColName, simDataFieldRaColName, simDataFieldDecColName]
        while '' in self.columnsNeeded: self.columnsNeeded.remove('')
        self.fieldColumnsNeeded = [fieldIDColName, fieldRaColName, fieldDecColName]
        self.slicer_init={'simDataFieldIDColName':simDataFieldIDColName,
                          'simDataFieldRaColName':simDataFieldRaColName,
                          'simDataFieldDecColName':simDataFieldDecColName,
                          'fieldIDColName':fieldIDColName,
                          'fieldRaColName':fieldRaColName,
                          'fieldDecColName':fieldDecColName, 'badval':badval}
        self.plotFuncs = [BaseSkyMap, OpsimHistogram]
        self.needsFields = True


    def setupSlicer(self, simData, fieldData, maps=None):
        """Set up opsim field slicer object.

        simData = numpy rec array with simulation pointing history,
        fieldData = numpy rec array with the field information (ID, RA, Dec),
        Values for the column names are set during 'init'.
        """
        if hasattr(self,'slicePoints'):
            warning_msg = 'Warning: this OpsimFieldSlicer was already set up once. '
            warning_msg += 'Re-setting up an OpsimFieldSlicer can change the field information. '
            warning_msg += 'Rerun metrics if this was intentional. '
            warnings.warn(warning_msg)
        # Set basic properties for tracking field information, in sorted order.
        idxs = np.argsort(fieldData[self.fieldIDColName])
        # Set needed values for slice metadata.
        self.slicePoints['sid'] = fieldData[self.fieldIDColName][idxs]
        self.slicePoints['ra'] = fieldData[self.fieldRaColName][idxs]
        self.slicePoints['dec'] = fieldData[self.fieldDecColName][idxs]
        self.nslice = len(self.slicePoints['sid'])
        self._runMaps(maps)
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.simDataFieldIDColName])
        simFieldsSorted = np.sort(simData[self.simDataFieldIDColName])
        self.left = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'left')
        self.right = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'right')

        self.spatialExtent = [simData[self.simDataFieldIDColName].min(),
                                  simData[self.simDataFieldIDColName].max()]
        self.shape = self.nslice

        @wraps(self._sliceSimData)

        def _sliceSimData(islice):
            idxs = self.simIdxs[self.left[islice]:self.right[islice]]
            # Build dict for slicePoint info
            slicePoint={}
            for key in self.slicePoints.keys():
                if (np.shape(self.slicePoints[key])[0] == self.nslice) & (key is not 'bins') & (key is not 'binCol'):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs':idxs, 'slicePoint':slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if two grids are equivalent."""

        result = False
        if isinstance(otherSlicer, OpsimFieldSlicer):
            if np.all(otherSlicer.shape == self.shape):
                # Check if one or both slicers have been setup
                if (self.slicePoints['ra'] is not None) or (otherSlicer.slicePoints['ra'] is not None):
                    if (np.array_equal(self.slicePoints['ra'], otherSlicer.slicePoints['ra']) &
                        np.array_equal(self.slicePoints['dec'], otherSlicer.slicePoints['dec']) &
                        np.array_equal(self.slicePoints['sid'], otherSlicer.slicePoints['sid'])):
                        result = True
                # If they have not been setup, check that they have same fields
                elif ((otherSlicer.fieldIDColName == self.fieldIDColName) &
                      (otherSlicer.fieldRaColName == self.fieldRaColName) &
                      (otherSlicer.fieldDecColName == self.fieldDecColName)):
                    result = True
        return result
