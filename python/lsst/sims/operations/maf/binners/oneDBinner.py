# oneDBinner - slices based on values in one data column in simData.

import numpy as np
import matplotlib.pyplot as plt
import warnings
try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf

from .baseBinner import BaseBinner

class OneDBinner(BaseBinner):
    """oneD Binner."""
    def __init__(self, sliceDataColName=None, verbose=True):
        """Instantiate. """
        super(OneDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ONED'
        self.bins = None
        self.nbins = None
        self.sliceDataColName = sliceDataColName
        self.columnsNeeded = [sliceDataColName]

    def setupBinner(self, simData, bins=None, nbins=100, binMin=None, binMax=None): 
        """Set up bins in binner.        

        'bins' can be a numpy array with the binpoints for sliceDataCol 
        or can be left 'None' in which case nbins will be used together with data min/max values
        to slice data in 'sliceDataCol'.
        Bins work like numpy histogram bins: the last 'bin' value is end value of last bin;
          all bins except for last bin are half-open ([a, b>), the last one is ([a, b])."""
        if self.sliceDataColName is None:
            raise Exception('sliceDataColName was not defined when binner instantiated.')
        sliceDataCol = simData[self.sliceDataColName]
        if binMin is None:
            binMin = sliceDataCol.min()
        if binMax is None:
            binMax = sliceDataCol.max()
       
        if bins is None:
            binsize = (binMax - binMin) / float(nbins)
            bins = np.arange(binMin, binMax+binsize/2.0, binsize, 'float')
            self.bins = bins
        else:
            self.bins = np.sort(bins)
        # Set nbins to be one less than # of bins because last binvalue is RH edge only
        self.nbins = len(self.bins) - 1
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.sliceDataColName])
        simFieldsSorted = np.sort(simData[self.sliceDataColName])
        # "left" values are location where simdata == bin value
        self.left = np.searchsorted(simFieldsSorted, self.bins[:-1], 'left')
        self.left = np.concatenate((self.left, np.array([len(self.simIdxs),])))
        
    def __iter__(self):
        self.ipix = 0
        return self

    def next(self):
        """Return the binvalues for this binpoint."""
        if self.ipix >= self.nbins:
            raise StopIteration
        binlo = self.bins[self.ipix]
        self.ipix += 1
        return binlo

    def __getitem__(self, ipix):
        binlo = self.bins[ipix]
        return binlo
    
    def __eq__(self, otherBinner):
        """Evaluate if binners are equivalent."""
        if isinstance(otherBinner, OneDBinner):
            return np.all(otherBinner.bins == self.bins)
        else:
            return False
            
    def sliceSimData(self, binpoint):
        """Slice simData on oneD sliceDataCol, to return relevant indexes for binpoint."""
        # Find the index of this binpoint in the bins array, then use this to identify
        #  the relevant 'left' values, then return values of indexes in original data array
        i = (np.where(binpoint == self.bins))[0]
        return self.simIdxs[self.left[i]:self.left[i+1]]

    def plotBinnedData(self, metricValues, title=None,
                       fignum=None, units=None,
                       legendLabel=None, addLegend=False,
                       legendloc='upper left', 
                       filled=False, alpha=0.5, ylog=False,
                       ylabel=None, xlabel=None, plotMin=None, plotMax=None, histRange=None):
        """Plot a set of oneD binned metric data.

        metricValues = the values to be plotted at each bin
        title = title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel =  y axis label (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        filled = flag to plot histogram as filled bars or lines (default False = lines)
        alpha = alpha value for plot bins if filled (default 0.5).
        ylog = make the y-axis log (default False)
        plotMin = min for y-axis
        plotMax = max for y-axis
        histRange = ignored
        """
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        leftedge = self.bins[:-1]
        width = np.diff(self.bins)
        if filled:
            plt.bar(leftedge, metricValues, width, label=legendLabel,
                    linewidth=0, alpha=alpha, log=ylog)
        else:
            x = np.ravel(zip(leftedge, leftedge+width))
            y = np.ravel(zip(metricValues, metricValues))
            if ylog:
                plt.semilogy(x, y, label=legendLabel)
            else:
                plt.plot(x, y, label=legendLabel)
        if ylabel is None:
            ylabel = 'Count'
        plt.ylabel(ylabel)
        if xlabel is None:
            xlabel=self.sliceDataColName
            if units != None:
                xlabel += ' (' + units + ')'
        plt.xlabel(xlabel)
        if plotMin:
            plt.ylim([plotMin,plotMax])
        if (addLegend != None):
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if (title!=None):
            plt.title(title)
        return fig.number

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """

        header_dict = dict(comment=comment, metricName=metricName, simDataName=simDataName,
                           metadata=metadata, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval)
        base=BaseBinner()
        base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        #update the header
        hdulist = pyf.open(outfilename, mode='update')
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key in header_dict.keys():
                    hdulist[0].header[key] = header_dict[key]
        hdulist.close()

        #append the bins
        hdulist = pyf.open(outfilename, mode='append')
        binHDU = pyf.PrimaryHDU(data=self.bins)
        hdulist.append(binHDU)
        hdulist.flush()
        hdulist.close()
        
        return outfilename

    def readMetricData(self, infilename, verbose=False):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        
        bins = hdulist[1].data.copy()
        
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)
        
        self.bins = bins
        self.nbins = len(self.bins)
        self.badval = header['badval'.upper()]
        self.int_badval = header['int_badval']
        if verbose:
            print '(Re)set bins with data from %s' %(infilename)
                
        return metricValues, self, header
