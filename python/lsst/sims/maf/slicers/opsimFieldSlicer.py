# Class for opsim field based slicer.

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
   
from .baseSlicer import BaseSlicer
from .baseSpatialSlicer import BaseSpatialSlicer

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
        

    def setupSlicer(self, simData, fieldData):
        """Set up opsim field slicer object.
        
        simData = numpy rec array with simulation pointing history,
        fieldData = numpy rec array with the field information (ID, RA, Dec),
        Values for the column names are set during 'init'. 
        """
        # Set basic properties for tracking field information, in sorted order.
        idxs = np.argsort(fieldData[self.fieldIDColName])
        # Set needed values for slice metadata.
        self.slicePoints['sid'] = fieldData[self.fieldIDColName][idxs]
        self.slicePoints['ra'] = fieldData[self.fieldRaColName][idxs]
        self.slicePoints['dec'] = fieldData[self.fieldDecColName][idxs]
        self.nslice = len(self.slicePoints['sid'])
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.simDataFieldIDColName])
        simFieldsSorted = np.sort(simData[self.simDataFieldIDColName])
        self.left = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'left')
        self.right = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'right')
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            idxs = self.simIdxs[self.left[islice]:self.right[islice]]  
            return {'idxs':idxs,
                    'slicePoint':{'sid':self.slicePoints['sid'][islice],
                                  'ra':self.slicePoints['ra'][islice], 'dec':self.slicePoints['dec'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if two grids are equivalent."""
        if isinstance(otherSlicer, OpsimFieldSlicer):
            return (np.all(otherSlicer.slicePoints['ra'] == self.slicePoints['ra']) and
                    np.all(otherSlicer.slicePoints['dec'] == self.slicePoints['dec']))        
        else:
            return False

    # Add some 'rejiggering' to base histogram to make it look nicer for opsim fields.
    def plotHistogram(self, metricValue, title=None, xlabel=None, ylabel='Number of Fields',
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      bins=None, binsize=None, cumulative=False, xMin=None, xMax=None,
                      logScale=False, flipXaxis=False,
                      scale=None, color='b', linestyle='-', **kwargs):
        """Histogram metricValue over the healpix bin points.

        title = the title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel = y axis label (default 'Number of Fields')** 
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, uses Freedman-Diaconis rule to set binsize)
        binsize = size of bins to use.  Will override "bins" if both are set.
        cumulative = make histogram cumulative (default False)
        xMin/Max = histogram range (default None, set by matplotlib hist)
        logScale = use log for y axis (default False)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        if ylabel is None:
            ylabel = 'Number of Fields'
        fignum = super(OpsimFieldSlicer, self).plotHistogram(metricValue,  xlabel=xlabel,
                                                             ylabel=ylabel,
                                                             title=title, fignum=fignum, 
                                                             label=label, 
                                                             addLegend=addLegend, legendloc=legendloc,
                                                             bins=bins, binsize=binsize, cumulative=cumulative,
                                                             xMin=xMin, xMax=xMax, logScale=logScale,
                                                             flipXaxis=flipXaxis, 
                                                             scale=1, yaxisformat='%d', color=color, **kwargs)
        return fignum
