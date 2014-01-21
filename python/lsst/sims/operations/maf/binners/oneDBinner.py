# oneDBinner - slices simData based on values in one data column. 

import numpy as np

from .baseBinner import BaseBinner

class OneDBinner(BaseBinner):
    """oneD Binner."""
    def __init__(self, sliceDataCol, bins=None, nbins=100, verbose=True):
        """Instantiate object and call set up method.

        'bins' can be a numpy array with the binpoints for sliceDataCol 
           or can be left 'None' in which case nbins will be used together with data min/max values
           to slice data in 'sliceDataCol'. """
        super(OneDBinner, self).__init__(verbose=verbose)
        if bins == None:
            binsize = (sliceDataCol.max() - sliceDataCol.min()) / float(nbins)
            bins = np.arange(sliceDataCol.min(), sliceDataCol.max() + binsize, binsize, 'float')
        self._setupBinner(bins)
        return
    
    def _setupGrid(self, bins):
        """Set up one D binner."""
        # Set number of 'pixels' in the grid. 
        # Corresponds to the number of metric values returned from the metric.
        self.bins = bins
        self.nbins = len(bins)
        return

    def __iter__(self):
        self.ipix = 0
        return self

    def next(self):
        """Set the binvalues to return when iterating over binpoints."""
        if self.ipix >= self.nbins:
            raise StopIteration
        (binlo, binhi) = (self.bins[self.ipix], self.bins[self.ipix+1])
        self.ipix += 1
        return binlo, binhi

    def __getitem__(self, ipix):
        return ipix
    
    def __eq__(self, otherBinner):
        """Evaluate if binners are equivalent."""
        if isinstance(otherBinner, OneDBinner):
            return np.all(otherBinner.bins == self.bins)
        else:
            return False
            
    def sliceSimData(self, binpoint):
        """Slice simData on oneD slice column, to return relevant indexes for gridpoint."""
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

    def plotBinnedData(self, metricValues, metricLabel, title=None, fignum=None,
                       legendLabel=None, addLegend=False, legendloc='upper left', 
                       filled=False, alpha=0.5):
        """Plot a set of oneD binned metric data.

        metricValues = the values to be plotted at each bin
        metricLabel = metric label (label for x axis)
        title = title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        filled = flag to plot histogram as filled bars or lines (default False = lines)
        alpha = alpha value for plot bins if filled (default 0.5). """
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        left = self.bins[:-1]
        width = np.diff(self.bins)
        if filled:
            plt.bar(left, metricValues[:-1], width, label=legendLabel, linewidth=0, alpha=alpha)
        else:
            x = np.ravel(zip(left, left+width))
            y = np.ravel(zip(histvalues[:-1], histvalues[:-1]))
            plt.plot(x, y, label=legendLabel)
        plt.xlabel(xlabel)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if title!=None:
            plt.title(title)
        return fig.number
