# Class for opsim field based slicer.

import numpy as np
from functools import wraps
import warnings
from lsst.sims.maf.plots.spatialPlotters import OpsimHistogram, BaseSkyMap

from .baseSpatialSlicer import BaseSpatialSlicer

__all__ = ['OpsimFieldSlicer']


class OpsimFieldSlicer(BaseSpatialSlicer):
    """A spatial slicer that evaluates pointings based on matched IDs between the simData and fieldData.

    Note that this slicer uses the fieldID of the simulated data fields to generate the spatial matches.
    Thus, it is not suitable for use in evaluating dithering or high resolution metrics
    (use the HealpixSlicer instead for those use-cases).

    When the slicer is set up, it takes two arrays: fieldData and simData. FieldData is a numpy.recarray
    containing the information about the fields - this is the basis for slicing.
    The simData is a numpy.recarray that holds the information about the pointings - this is the data that
    is matched against the fieldData.

    Parameters
    ----------
    simDataFieldIDColName : str, optional
        Name of the column in simData for the fieldID
        Default fieldID.
    simDataFieldRaColName : str, optional
        Name of the column in simData for the fieldRA.
        Default fieldRA.
    simDataFieldDecColName : str, optional
        Name of the column in simData for the fieldDec.
        Default fieldDec.
    fieldIDColName : str, optional
        Name of the column in the fieldData for the fieldID (to match with simData).
        Default fieldID.
    fieldRAColName : str, optional
        Name of the column in the fieldData for the fieldRA (used for plotting).
        Default fieldDec.
    fieldDecColName : str, optional
        Name of the column in the fieldData for the fieldDec (used for plotting).
        Default fieldRA.
    verbose : boolean, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : float, optional
        Bad value flag, relevant for plotting. Default -666.
    """
    def __init__(self, simDataFieldIDColName='fieldID',
                 simDataFieldRaColName='fieldRA', simDataFieldDecColName='fieldDec',
                 fieldIDColName='fieldID', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                 verbose=True, badval=-666):
        super(OpsimFieldSlicer, self).__init__(verbose=verbose, badval=badval)
        self.fieldID = None
        self.simDataFieldIDColName = simDataFieldIDColName
        self.fieldIDColName = fieldIDColName
        self.fieldRaColName = fieldRaColName
        self.fieldDecColName = fieldDecColName
        self.columnsNeeded = [simDataFieldIDColName, simDataFieldRaColName, simDataFieldDecColName]
        while '' in self.columnsNeeded:
            self.columnsNeeded.remove('')
        self.fieldColumnsNeeded = [fieldIDColName, fieldRaColName, fieldDecColName]
        self.slicer_init = {'simDataFieldIDColName': simDataFieldIDColName,
                            'simDataFieldRaColName': simDataFieldRaColName,
                            'simDataFieldDecColName': simDataFieldDecColName,
                            'fieldIDColName': fieldIDColName,
                            'fieldRaColName': fieldRaColName,
                            'fieldDecColName': fieldDecColName, 'badval': badval}
        self.plotFuncs = [BaseSkyMap, OpsimHistogram]
        self.needsFields = True

    def setupSlicer(self, simData, fieldData, maps=None):
        """Set up opsim field slicer object.

        Parameters
        -----------
        simData : numpy.recarray
            Contains the simulation pointing history.
        fieldData : numpy.recarray
            Contains the field information (ID, Ra, Dec) about how to slice the simData.
            For example, only fields in the fieldData table will be matched against the simData.
        maps : list of lsst.sims.maf.maps objects, optional
            Maps to run and provide additional metadata at each slicePoint. Default None.
        """
        if hasattr(self, 'slicePoints'):
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
            slicePoint = {}
            for key in self.slicePoints.keys():
                if (np.shape(self.slicePoints[key])[0] == self.nslice) & \
                        (key is not 'bins') & (key is not 'binCol'):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs': idxs, 'slicePoint': slicePoint}
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
