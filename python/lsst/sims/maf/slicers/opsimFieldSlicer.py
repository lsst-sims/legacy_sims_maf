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
    
    def __init__(self, verbose=True, simDataFieldIdColName='fieldID',
                 simDataFieldRaColName='fieldRA', simDataFieldDecColName='fieldDec',
                 fieldIDColName='fieldID', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                 badval=-666):
        """Instantiate opsim field slicer (an index-based slicer that can do spatial plots).

        simDataFieldIdColName = the column name in simData for the field ID
        simDataFieldRaColName = the column name in simData for the field RA 
        simDataFieldDecColName = the column name in simData for the field Dec
        fieldIDcolName = the column name in the fieldData for the field ID (to match with simData)
        fieldRaColName = the column name in the fieldData for the field RA (for plotting only)
        fieldDecColName = the column name in the fieldData for the field Dec (for plotting only).
        """
        super(OpsimFieldSlicer, self).__init__(verbose=verbose, badval=badval)
        self.bins = {}
        self.bins['fieldID'] = None
        self.bins['ra'] = None
        self.bins['dec'] = None
        self.nbins = None
        self.simDataFieldIdColName = simDataFieldIdColName
        self.fieldIDColName = fieldIDColName
        self.fieldRaColName = fieldRaColName
        self.fieldDecColName = fieldDecColName
        self.columnsNeeded = [simDataFieldIdColName, simDataFieldRaColName, simDataFieldDecColName]
        while '' in self.columnsNeeded: self.columnsNeeded.remove('')
        self.fieldColumnsNeeded = [fieldIDColName, fieldRaColName, fieldDecColName]
        self.slicer_init={'simDataFieldIdColName':simDataFieldIdColName,
                          'simDataFieldRaColName':simDataFieldRaColName,
                          'simDataFieldDecColName':simDataFieldDecColName,
                          'fieldIDColName':fieldIDColName,
                          'fieldRaColName':fieldRaColName,
                          'fieldDecColName':fieldDecColName}
        

    def setupSlicer(self, simData, fieldData):
        """Set up opsim field slicer object.
        
        simData = numpy rec array with simulation pointing history,
        fieldData = numpy rec array with the field information (ID, RA, Dec),
        Values for the column names are set during 'init'. 
        """
        # Set basic properties for tracking field information, in sorted order.
        idxs = np.argsort(fieldData[self.fieldIDColName])
        self.bins['fieldID'] = fieldData[self.fieldIDColName][idxs]
        self.bins['ra'] = fieldData[self.fieldRaColName][idxs]
        self.bins['dec'] = fieldData[self.fieldDecColName][idxs]
        self.nbins = len(self.bins['fieldID'])
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.simDataFieldIdColName])
        simFieldsSorted = np.sort(simData[self.simDataFieldIdColName])
        self.left = np.searchsorted(simFieldsSorted, self.bins['fieldID'], 'left')
        self.right = np.searchsorted(simFieldsSorted, self.bins['fieldID'], 'right')
        # Build slicing method.     
        @wraps(self.sliceSimData)
        def sliceSimData(binpoint):
            """Slice simData on fieldID, to return relevant indexes for binpoint."""
            i = np.where(self.bins['fieldID'] == binpoint)
            return self.simIdxs[self.left[i]:self.right[i]]   
        setattr(self, 'sliceSimData', sliceSimData)
        
    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self
    
    def _resultsDict(self,ipix):
        metadata = {'ra':self.bins['ra'][ipix], 'dec':self.bins['dec'][ipix], 'fieldID': self.bins['fieldID'][ipix]}
        idxs = self.sliceSimData(self.bins['fieldID'][ipix])
        return {'idxs':idxs, 'metadata':metadata}
    
    def next(self):
        """Return RA/Dec values when iterating over binpoints."""
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.nbins:
            raise StopIteration
        result = self._resultsDict(self.ipix) 
        self.ipix += 1
        return result

    def __getitem__(self, ipix):
        return self._resultsDict(ipix)
    
    def __eq__(self, otherSlicer):
        """Evaluate if two grids are equivalent."""
        if isinstance(otherSlicer, OpsimFieldSlicer):
            return (np.all(otherSlicer.bins['fieldID'] == self.bins['fieldID']) and
                    np.all(otherSlicer.bins['ra'] == self.bins['ra']) and
                    np.all(otherSlicer.bins['dec'] == self.bins['dec']))        
        else:
            return False

    # Add some 'rejiggering' to base histogram to make it look nicer for opsim fields.
    def plotHistogram(self, metricValue, title=None, xlabel=None, ylabel='Number of Fields',
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      bins=None, cumulative=False, histMin=None, histMax=None, ylog=False, flipXaxis=False,
                      scale=None, color='b', **kwargs):
        """Histogram metricValue over the healpix bin points.

        title = the title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel = y axis label (default 'Number of Fields')** 
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, uses Freedman-Diaconis rule to set binsize)
        cumulative = make histogram cumulative (default False)
        histMin/Max = histogram range (default None, set by matplotlib hist)
        ylog = use log for y axis (default False)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        fignum = super(OpsimFieldSlicer, self).plotHistogram(metricValue,  xlabel=xlabel,
                                                             ylabel=ylabel,
                                                             title=title, fignum=fignum, 
                                                             label=label, 
                                                             addLegend=addLegend, legendloc=legendloc,
                                                             bins=bins, cumulative=cumulative,
                                                             histMin=histMin,histMax=histMax, ylog=ylog,
                                                             flipXaxis=flipXaxis, 
                                                             scale=1, yaxisformat='%d', color=color, **kwargs)
        return fignum
