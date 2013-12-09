# Global grid class for metrics.
# This kind of grid considers all visits / sequences / etc. regardless of RA/Dec.
# Metrics are calculable on a global grid - the data they will receive will be
#  all of the relevant data, which in this case is all of the data values in a 
#  particular simData column. 
# Subclasses of the global metric could slice on other aspects of the simData (but not
#  RA/Dec as that is done more efficiently in the spatial classes).

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pyf

from .baseGrid import BaseGrid

class GlobalGrid(BaseGrid):
    """Global grid"""
    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate object and call set up global grid method."""
        super(GlobalGrid, self).__init__(verbose=verbose)
        self._setupGrid(*args, **kwargs)
        self.gridtype = 'GLOBAL'
        return
    
    def _setupGrid(self, *args, **kwargs):
        """Set up global grid.

        For base GlobalGrid class, this does nothing. For subclasses this could
        (for example) split the grid by time. """
        # Set number of 'pixels' in the grid. 
        # Corresponds to the number of metric values returned from the metric.
        self.npix = 1
        return

    def __iter__(self):
        """Iterate over the grid."""
        self.ipix = 0
        return self

    def next(self):
        """Set the gridvalues to return when iterating over grid."""
        if self.ipix >= self.npix:
            raise StopIteration
        ipix = self.ipix
        self.ipix += 1
        return ipix

    def __getitem__(self, ipix):
        """Make global grid indexable."""  
        return ipix
    
    def __eq__(self, otherGrid):
        """Evaluate if grids are equivalent."""
        if isinstance(otherGrid, GlobalGrid):
            return True
        else:
            return False
            
    def sliceSimData(self, gridpoint, simDataCol):
        """Return relevant indices in simData for 'gridpoint'. 

        For base GlobalGrid, this is all data."""
        indices = np.where(simDataCol)
        return indices

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        metricHistValues=None, metricHistBins=None,
                        simDataName='', metadata='',
                        gridfile='', int_badval=-666, badval=-666,dt='float'):
        head = pyf.Header()
        head.update(comment=comment, metricName=metricName,metricValue=metricValue,
                    simDataName=simDataName, metadata=metadata, gridfile=gridfile,
                    gridtype=self.gridtype, int_badval=int_badval,
                    badval=badval, hist='False')
        if metricHistValues != None:
            c1 = pyf.Column(name='HistValues', format='K()', array=metricHistValues)
            c2 = pyf.Column(name='HistBins', format='K()', array=metricHistBins)
            hdu = pyf.newtable([c1,c2])
            for i in range(len(head)):  hdu.header[head.keys()[i]]=head[i]
            hdu.header['hist'] = 'True'
            hdu.writeto(outfilename+'.fits')
        else:
            #just write the header and have the metric value as the data
            pyf.writeto(outfilename+'.fits', metricValues.astype(dt), head)
        return

    def readMetricData(self, infilename):
        f = pyf.open(infilename)
        head = f[1].header
        if head['hist'] == 'True':
            metricHistValues = f[1].data['HistValues']
            metricHistBins =f[1].data['HistBins']
        else:
            metricHistValues = None
            metricHistBins = None
            
        return head['metricValues'], metricHistValues,metricHistBins, head['metricName'], \
            head['simDataName'],head['metadata'], head['comment'], head['gridfile'], head['gridtype']
        
        pass
    

    def plotHistogram(self, simDataCol, simDataColLabel, title=None, fignum=None, 
                      legendLabel=None, addLegend=False, bins=100, cumulative=False,
                      histRange=None, flipXaxis=False, scale=1.0):
        """Plot a histogram of simDataCol values, labelled by simDataColLabel.

        simDataCol = the data values for generating the histogram
        simDataColLabel = the units for the simDataCol ('m5', 'airmass', etc.)
        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default 100)
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

    def plotBinnedData(self, histbins, histvalues, xlabel, title=None, fignum=None,
                       legendLabel=None, addLegend=False, alpha=0.5):
        """Plot a set of pre-binned histogrammed data. 

        histbins = the bins for the histogram (as returned by numpy histogram function, for example)
        histvalues = the values of the histogram
        xlabel = histogram label (label for x axis)
        title = title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        alpha = alpha value for plot bins (default 0.5). """
        # Plot the histogrammed data.
        fig = plt.figure(fignum)
        left = histbins[:-1]
        width = np.diff(histbins)
        plt.bar(left, histvalues, width, linewidth=0, alpha=0.5)
        plt.xlabel(xlabel)
        if addLegend:
            plt.legend(fancybox=True, fontsize='smaller', loc='upper left')
        if title!=None:
            plt.title(title)
        return fig.number
