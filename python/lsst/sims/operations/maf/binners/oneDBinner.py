# oneDBinner - slices based on values in one data column in simData.

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pyf

from .baseBinner import BaseBinner

class OneDBinner(BaseBinner):
    """oneD Binner."""
    def __init__(self,  verbose=True):
        """Instantiate. """
        super(OneDBinner, self).__init__(verbose=verbose)
        self.binnertype = 'ONED'

    def setupBinner(self, simData, sliceDataColName, bins=None, nbins=100):
        """Set up bins in binner.        

        'bins' can be a numpy array with the binpoints for sliceDataCol 
        or can be left 'None' in which case nbins will be used together with data min/max values
        to slice data in 'sliceDataCol'. """
        self.sliceDataColName = sliceDataColName
        self.sliceDataCol = simData[sliceDataColName]
        if bins == None:
            binsize = (self.sliceDataCol.max() - self.sliceDataCol.min()) / float(nbins)
            bins = np.arange(self.sliceDataCol.min(), self.sliceDataCol.max() + binsize, binsize, 'float') 
        self.bins = bins
        self.nbins = len(self.bins)

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
        plt.ylabel(metricLabel)
        plt.xlabel(self.sliceDataColName)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc, numpoints=1)
        if title!=None:
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

    def readMetricData(self, infilename):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        
        bins = hdulist[1].data.copy()
        
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)
        
        binner = OneDBinner()
        binner.bins = bins
        binner.nbins = len(self.bins)
        binner.badval = header['badval'.upper()]
        binner.int_badval = header['int_badval']
                
        return metricValues, binner, header
