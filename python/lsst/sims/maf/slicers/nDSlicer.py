# nd Slicer slices data on N columns in simData

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
from functools import wraps

from .baseSlicer import BaseSlicer

    
class NDSlicer(BaseSlicer):
    """Nd slicer (N dimensions)"""
    def __init__(self, sliceColList=None, verbose=True, binsList=None, nbinsList=100):  
        """Instantiate object.
        binsList can be a list of numpy arrays with the respective slicepoints for sliceDataColList,
            (default 'None' uses nbinsList together with data min/max values to set bins).
        nbinsList can be a list of values (one per column in sliceDataColList) or a single value
            (repeated for all columns, default=100)."""
        super(NDSlicer, self).__init__(verbose=verbose)
        self.bins = None 
        self.nslice = None
        self.sliceColList = sliceColList
        self.columnsNeeded = self.sliceColList
        if self.sliceColList != None:
            self.nD = len(self.sliceColList)
        else:
            self.nD = None
        self.slicer_init={'sliceColList':sliceColList, 'binList':binsList, 'nbinsList':nbinsList}

    def setupSlicer(self, simData):
        """Set up bins. """
        # For save-file
        self.slicer_setup={'binsList':binsList, 'nbinsList':nbinsList}
        # Parse input bins choices.
        if binsList != None:
            if len(binsList) != self.nD:
                raise Exception('BinsList must be same length as sliceDataColNames')
            self.bins = binsList
            for b in self.bins:
                b = np.sort(b)
        else:
            if isinstance(nbinsList, list):
                if len(nbinsList) != self.nD:
                        raise Exception('nbinsList must be same length as sliceDataColList')
            else:  # we have an nbins but it's a single number to apply to all cols
                nbinsList = [nbinsList for i in range(self.nD)]
            # Set the bins. Bins for NDslicer is a list of bins (one for each dimension).
            self.bins = []
            for sliceColName, nbins in zip(self.sliceColList, nbinsList):
                sliceCol = simData[sliceColName]
                binsize = (sliceCol.max() - sliceCol.min()) / float(nbins)
                bins = np.arange(sliceCol.min(), sliceCol.max() + binsize/2.0,
                                 binsize, 'float')
                self.bins.append(bins)
        # Count how many bins we have total (not counting last 'RHS' bin values, as in oneDSlicer).
        self.nslice = (np.array(map(len, self.bins))-1).prod()
        self.slicePoints['sid'] = np.arange(self.nslice)
        binsForIteration = []
        for b in self.bins:
            binsForIteration.append(b[:-1])
        biniterator = itertools.product(*binsForIteration)
        self.slicePoints['bins'] = []
        for b in biniterator:
            self.slicePoints['bins'].append(b)
        # Set up data slicing.
        self.simIdxs = []
        self.lefts = []
        for sliceColName, bins in zip(self.sliceColList, self.bins):
            simIdxs = np.argsort(simData[sliceColName])
            simFieldsSorted = np.sort(simData[sliceColName])
            # "left" values are location where simdata == bin value
            left = np.searchsorted(simFieldsSorted, bins[:-1], 'left')
            left = np.concatenate((left, np.array([len(simIdxs),])))
            # Add these calculated values into the class lists of simIdxs and lefts.
            self.simIdxs.append(simIdxs)
            self.lefts.append(left)
        @wraps (self._sliceSimData)
        def _sliceSimData(islice):
            """Slice simData to return relevant indexes for slicepoint."""
            # Identify relevant pointings in each dimension.
            simIdxsList = []
            for d, i in zip(range(self.nD), islice):
                simIdxsList.append(set(self.simIdxs[d][self.lefts[d][i]:self.lefts[d][i+1]]))
            idxs = list(set.intersection(*simIdxsList))
            binLeft = []
            for b, i in zip(self.bins, islice):
                binLeft.append(b[i])
            return {'idxs':idxs,
                    'slicePoint':{'sid':islice, 'binLeft':binLeft}}
        setattr(self, '_sliceSimData', _sliceSimData)                

    def __iter__(self):
        """Iterate over the slicepoints."""
        # Order of iteration over bins: go through bins in each sliceCol in the sliceColList in order.
        self.islice = 0
        sliceIdsForIteration = []
        for b in self.bins:
            sliceIdsForiteration.append(np.arange(len(b[:-1])))
        self.sliceIdIterator = itertools.product(*sliceIdsForIteration)
        # Note that this iterates from 'right' to 'left'
        #  (i.e. bins[0] moves slowest, bins[N] moves fastest)
        return self

    def next(self):
        """
        Returns results of self._sliceSimData when iterating over slicer.
        Results of self._sliceSimData should be dictionary of
           {'idxs' - the data indexes relevant for this slice of the slicer,
           'slicePoint' - the metadata for the slicePoint .. always includes ['sid'] key for ID of slicePoint.}
        """
        if self.islice >= self.nslice:
            raise StopIteration
        islice = self.sliceIdIterator.next()
        self.islice += 1        
        return self._sliceSimData(islice)

    def __getitem__(self, islice):
        """Return slicepoint at index ipix."""
        # Set up slice iterator again.
        sliceIdsForIteration = []
        for b in self.bins:
            sliceIdsForiteration.append(np.arange(len(b[:-1])))
        sliceIdIterator = itertools.product(*sliceIdsForIteration)
        # And then run through slice iterator until we hit this slice number.
        #  .. i counts the number of slices.
        sid = sliceIdIterator.next()
        for i, si in zip(range(1, islice+1), sliceIdIterator):
            sid = si
        return self._sliceSimData(sid)
    
    def __eq__(self, otherSlicer):
        """Evaluate if grids are equivalent."""
        if isinstance(otherSlicer, NDSlicer):
            if otherSlicer.nD != self.nD:
                return False
            for i in range(self.nD):
                if np.all(otherSlicer.bins[i] != self.bins[i]):
                    return False                
            return True
        else:
            return False

    def plotBinnedData2D(self, metricValues,
                        xaxis, yaxis, xlabel=None, ylabel=None,
                        title=None, fignum=None, ylog=False, units='',
                        clims=None, cmap=None, cbarFormat=None):
        """Plot 2 axes from the sliceColList, identified by xaxis/yaxis, given the metricValues at all
        slicepoints [sums over non-visible axes]. 

        metricValues = the metric data (as calculated when iterating through slicer)
        xaxis, yaxis = the x and y dimensions to plot (i.e. 0/1 would plot binsList[0] and
            binsList[1] data values, with other axis )
        title = title for the plot (default None)
        xlabel/ylabel = labels for the x and y axis (default None, uses sliceColList names). 
        fignum = the figure number to use (default None - will generate new figure)
        ylog = make the colorscale log.
        """
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        newshape = []
        for b in self.bins:
            newshape.append(len(b)-1)
        newshape.reverse()
        md = metricValues.reshape(newshape)
        # Sum over other dimensions. Note that masked values are not included in sum.
        sumaxes = range(self.nD)
        sumaxes.remove(xaxis)
        sumaxes.remove(yaxis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        # Plot data.
        x, y = np.meshgrid(self.bins[xaxis][:-1], self.bins[yaxis][:-1])
        if ylog:
            norm = colors.LogNorm()
        else:
            norm = None
        if clims is None:
            im = plt.contourf(x, y, md, 250, norm=norm, extend='both', cmap=cmap)
        else:
            im = plt.contourf(x, y, md, 250, norm=norm, extend='both', cmap=cmap,
                              vmin=clims[0], vmax=clims[1])
        if xlabel is None:
            xlabel = self.sliceDataColList[xaxis]
        plt.xlabel(xlabel)
        if ylabel is None:
            ylabel= self.sliceDataColList[yaxis]
        plt.ylabel(ylabel)
        cb = plt.colorbar(im, aspect=25, extend='both', orientation='horizontal', format=cbarFormat)
        cb.set_label(units)
        if title!=None:
            plt.title(title)
        return fig.number

    def plotBinnedData1D(self, metricValues, axis, xlabel=None, ylabel=None,
                         title=None, fignum=None, 
                         histRange=None, units=None,
                         label=None, addLegend=False, legendloc='upper left',
                         filled=False, alpha=0.5, logScale=False):
        """Plot a single axes from the sliceColList, identified by axis, given the metricValues at all
        slicepoints [sums over non-visible axes]. 

        metricValues = the values to be plotted at each bin
        axis = the dimension to plot (i.e. 0 would plot binsList[0])
        title = title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel =  y axis label (default None)
        histRange = x axis min/max values (default None, use plot defaults)
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        filled = flag to plot histogram as filled bars or lines (default False = lines)
        alpha = alpha value for plot bins if filled (default 0.5).
        logScale = make the y-axis log (default False)
        """
        # Reshape the metric data so we can isolate the values to plot
        # (just new view of data, not copy).
        newshape = []
        for b in self.bins:
            newshape.append(len(b)-1)
        newshape.reverse()
        md = metricValues.reshape(newshape) 
        # Sum over other dimensions. Note that masked values are not included in sum.
        sumaxes = range(self.nD)
        sumaxes.remove(axis)
        sumaxes = tuple(sumaxes)
        md = md.sum(sumaxes)
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        # Plot data.
        leftedge = self.bins[axis][:-1]
        width = np.diff(self.bins[axis])
        if filled:
            plt.bar(leftedge, md, width, label=label,
                    linewidth=0, alpha=alpha, log=ylog)
        else:
            x = np.ravel(zip(leftedge, leftedge+width))
            y = np.ravel(zip(md, md))
            if logScale:
                plt.semilogy(x, y, label=label)
            else:
                plt.plot(x, y, label=label)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if xlabel is None:
            xlabel=self.sliceColName[axis]
            if units != None:
                xlabel += ' (' + units + ')'
        plt.xlabel(xlabel)
        if (histRange != None):
            plt.xlim(histRange)
        if (addLegend):
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if (title!=None):
            plt.title(title)
        return fig.number    
    



