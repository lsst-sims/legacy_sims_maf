# Subclass of Global grid class for metrics, which splits data based on TIME.

import numpy as np
from globalGrid import GlobalGrid

class TimeGlobalGrid(GlobalGrid):
    """Global grid"""
    def __init__(self, timesteps, verbose=True):
        """Instantiate object and call set up timestep global grid method.

        'timesteps' is a numpy array with the timesteps 
         (from start of survey, in days) to split the opsim data ('expmjd') values -
        for example [0, 365, 2*365...] would split the survey by years. """
        super(TimeGlobalGrid, self).__init__(verbose=verbose)
        self._setupGrid(timesteps)
        return
    
    def _setupGrid(self, timesteps):
        """Set up global grid with timesteps. """
        # Set number of 'pixels' in the grid. 
        # Corresponds to the number of metric values returned from the metric.
        self.timesteps = timesteps
        self.npix = len(timesteps)
        return

    def __iter__(self):
        """Iterate over the grid."""
        self.ipix = 0
        return self

    def next(self):
        """Set the gridvalues to return when iterating over grid."""
        if self.ipix >= self.npix:
            raise StopIteration
        # This returns ipix - an index in the timestep array. 
        # Could instead rearrange to return timestep value itself.
        return self.ipix

    def __getitem__(self, ipix):
        """Make timestep global grid indexable."""
        return ipix
    
    def __eq__(self, otherGrid):
        """Evaluate if grids are equivalent."""
        if isinstance(otherGrid, TimeGlobalGrid):
            return np.all(otherGrid.timesteps == self.timesteps)
        else:
            return False
            
    def sliceSimData(self, gridpoint, simDataTime):
        """Slice simData on simDataTime, to return relevant indexes for gridpoint."""
        # Timesteps measure time elapsed from start of survey; translate simDataTime.
        timesurvey = simDataTime - simDataTime[0]
        # Set the starting time interesting for this gridpoint.
        timestart = self.timesteps[gridpoint]
        # Try to set the ending time interesting for this gridpoint.
        try:
            timeend = self.timesteps[gridpoint + 1]
        except:
            timeend = timesurvey.max() + 1.0
        indices = np.where((timesurvey >= timestart) & (timesurvey < timeend))
        return indices

    def plotHistogram(self, simDataCol, simDataColLabel, title=None, fignum=None, 
                      legendLabel=None, addLegend=False, bins=None, cumulative=False,
                      histRange=None, flipXaxis=False, scale=1.0):
        """Plot a histogram of simDataCol values, labelled by simDataColLabel.

        simDataCol = the data values for generating the histogram
        simDataColLabel = the units for the simDataCol ('m5', 'airmass', etc.)
        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, try to set)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)"""
        super(GlobalGrid, self).plotHistogram(simDataCol, simDataColLabel, 
                                              title=title, fignum=fignum, 
                                              legendLabel=label, addLegend=addLegend,
                                              bins=bins, cumulative=cumulative,
                                              histRange=histRange, flipXaxis=flipXaxis,
                                              scale=scale)
