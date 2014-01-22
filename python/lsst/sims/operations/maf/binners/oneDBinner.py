# oneDBinner - slices based on values in one data column in simData.

import numpy as np
import matplotlib.pyplot as plt

from .baseBinner import BaseBinner

class OneDBinner(BaseBinner):
    """oneD Binner."""
    def __init__(self,  verbose=True):
        """Instantiate. """
        super(OneDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ONED'

    def setupBinner(self, sliceDataCol, bins=None, nbins=100):
        """Set up bins in binner.        

        'bins' can be a numpy array with the binpoints for sliceDataCol 
        or can be left 'None' in which case nbins will be used together with data min/max values
        to slice data in 'sliceDataCol'. """
        self.sliceDataCol = sliceDataCol
        if bins == None:
            binsize = (sliceDataCol.max() - sliceDataCol.min()) / float(nbins)
            bins = np.arange(sliceDataCol.min(), sliceDataCol.max() + binsize, binsize, 'float') 
        self.bins = bins
        self.nbins = len(bins)

    def __iter__(self):
        self.ipix = 0
        return self

    def next(self):
        """Return the binvalues for this binpoint."""
        if self.ipix >= self.nbins-1:
            raise StopIteration
        (binlo, binhi) = (self.bins[self.ipix], self.bins[self.ipix+1])
        self.ipix += 1
        return (binlo, binhi)

    def __getitem__(self, ipix):
        (binlo, binhi) = (self.bins[ipix], self.bins[ipix+1])
        return (binlo, binhi)
    
    def __eq__(self, otherBinner):
        """Evaluate if binners are equivalent."""
        if isinstance(otherBinner, OneDBinner):
            return np.all(otherBinner.bins == self.bins)
        else:
            return False
            
    def sliceSimData(self, binpoint):
        """Slice simData on oneD sliceDataCol, to return relevant indexes for binpoint."""
        indices = np.where((self.sliceDataCol >= binpoint[0]) & (self.sliceDataCol < binpoint[1]))
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
            y = np.ravel(zip(metricValues[:-1], metricValues[:-1]))
            plt.plot(x, y, label=legendLabel)
        plt.xlabel(metricLabel)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if title!=None:
            plt.title(title)
        return fig.number
