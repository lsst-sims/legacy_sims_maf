# oneDSlicer - slices based on values in one data column in simData.

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import warnings
from lsst.sims.maf.utils import optimalBins, ColInfo
from .baseSlicer import BaseSlicer


class OneDSlicer(BaseSlicer):
    """oneD Slicer."""
    def __init__(self, sliceColName=None, verbose=True, badval=-666, bins=None, binMin=None, binMax=None, binsize=None, sliceColUnits=None):
        """ 'bins' can be a numpy array with the binpoints for sliceCol or a single integer value
        (if a single value, this will be used as the number of bins, together with data min/max (or binMin/Max)),
        as in numpy's histogram function.
        If 'binsize' is used, this will override the bins value and will be used together with the data min/max
        (or binMin/Max) to set the binpoint values.

        Bins work like numpy histogram bins: the last 'bin' value is end value of last bin;
          all bins except for last bin are half-open ([a, b>), the last one is ([a, b]). """
        super(OneDSlicer, self).__init__(verbose=verbose, badval=badval)
        self.bins = None
        self.nbins = None
        self.sliceColName = sliceColName
        self.columnsNeeded = [sliceColName]
        self.slicer_init = {'sliceColName':self.sliceColName, 'badval':badval, 'bins':bins,
                            'binMin':binMin, 'binMax':binMax, 'binsize':binsize}
        self.bins = bins
        self.binMin = binMin
        self.binMax = binMax
        self.binsize = binsize
        if sliceColUnits is None:
            co = ColInfo()
            self.sliceColUnits = co.getUnits(self.sliceColName)
        
    def setupSlicer(self, simData): 
        """Set up bins in slicer.        

        'bins' can be a numpy array with the slicepoints for sliceCol or a single integer value
          (if a single value, this will be used as the number of bins, together with data min/max (or binMin/Max)),
          as in numpy's histogram function.
        If 'binsize' is used, this will override the bins value and will be used together with the data min/max
         (or binMin/Max) to set the slicepoint values.

        Bins work like numpy histogram bins: the last 'bin' value is end value of last bin;
          all bins except for last bin are half-open ([a, b>), the last one is ([a, b]).
          """
        if self.sliceColName is None:
            raise Exception('sliceColName was not defined when slicer instantiated.')
        sliceCol = simData[self.sliceColName]
        # Set bin min/max values.
        if self.binMin is None:
            self.binMin = sliceCol.min()
        if self.binMax is None:
            self.binMax = sliceCol.max()
        # Set bins.
        # Using binsize.
        if self.binsize is not None:  
            if self.bins is not None:
                warnings.warn('Both binsize and bins have been set; Using binsize %f only.' %(self.binsize))
            self.bins = np.arange(self.binMin, self.binMax+self.binsize/2.0, self.binsize, 'float')
        # Using bins value.
        else:
            # Bins was a sequence (np array or list)
            if hasattr(self.bins, '__iter__'):  
                self.bins = np.sort(self.bins)
            # Or bins was a single value. 
            else:
                if self.bins is None:
                    self.bins = optimalBins(sliceCol)
                nbins = int(self.bins)
                self.binsize = (self.binMax - self.binMin) / float(nbins)
                self.bins = np.arange(self.binMin, self.binMax+self.binsize/2.0, self.binsize, 'float')
        # Set nbins to be one less than # of bins because last binvalue is RH edge only
        self.nbins = len(self.bins) - 1
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.sliceColName])
        simFieldsSorted = np.sort(simData[self.sliceColName])
        # "left" values are location where simdata == bin value
        self.left = np.searchsorted(simFieldsSorted, self.bins[:-1], 'left')
        self.left = np.concatenate((self.left, np.array([len(self.simIdxs),])))
        
    def __iter__(self):
        self.ipix = 0
        return self

    def _sliceSimData(self, ipix):
        """Slice simData on oneD sliceCol, to return relevant indexes for slicepoint."""
        # Find the index of this slicepoint in the bins array, then use this to identify
        #  the relevant 'left' values, then return values of indexes in original data array
        idxs = self.simIdxs[self.left[ipix]:self.left[ipix+1]]
        slicePoint = {'pid':ipix, 'left':self.left[ipix],
                      'right':self.left[ipix+1], 'bin':self.bins[ipix]}
        return {'idxs':idxs, 'slicePoint':slicePoint}
         
    def next(self):
        """Return the binvalues for this slicepoint."""
        if self.ipix >= self.nbins:
            raise StopIteration
        result = self._sliceSimData(self.ipix)
        self.ipix += 1
        return result

    def __getitem__(self, ipix):
        result = self._sliceSimData(ipix) 
        return result
    
    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(otherSlicer, OneDSlicer):
            return np.all(otherSlicer.bins == self.bins)
        else:
            return False

    def plotBinnedData(self, metricValues, title=None,
                       fignum=None, units=None,
                       label=None, addLegend=False,
                       legendloc='upper left', 
                       filled=False, alpha=0.5, ylog=False,
                       ylabel=None, xlabel=None, yMin=None, yMax=None,
                       histMin=None, histMax=None, color='b', **kwargs):
        """Plot a set of oneD binned metric data.

        metricValues = the values to be plotted at each bin
        title = title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel =  y axis label (default None)
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        filled = flag to plot histogram as filled bars or lines (default False = lines)
        alpha = alpha value for plot bins if filled (default 0.5).
        ylog = make the y-axis log (default False)
        yMin/Max = min/max for y-axis 
        histMin/Max = min/max for x-axis (typically set by bin values though)
        """
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        leftedge = self.bins[:-1]
        width = np.diff(self.bins)
        if filled:
            plt.bar(leftedge, metricValues, width, label=label,
                    linewidth=0, alpha=alpha, log=ylog, color=color)
        else:
            x = np.ravel(zip(leftedge, leftedge+width))
            y = np.ravel(zip(metricValues, metricValues))
            if ylog:
                plt.semilogy(x, y, label=label, color=color, alpha=alpha)
            else:
                plt.plot(x, y, label=label, color=color, alpha=alpha)
        if ylabel is None:
            ylabel = 'Count'
        plt.ylabel(ylabel)
        if xlabel is None:
            xlabel=self.sliceColName
            if units != None:
                xlabel += ' (' + units + ')'
        plt.xlabel(xlabel)
        if (yMin is not None) or (yMax is not None):
            plt.ylim(yMin, yMax)
        if (histMin is not None) or (histMax is not None):
            plt.xlim(histMin, histMax)
        if (addLegend):
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if (title!=None):
            plt.title(title)
        return fig.number

