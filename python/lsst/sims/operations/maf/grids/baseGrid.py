# Base class for all grid objects. 
# 
# Grid objects must know slice the complete set of incoming opsim data for the metrics:
#  * for spatial metrics, this means slicing in RA/Dec space and iterating over the sky)
#  * for global metrics this means handing over the entire (or part of the) data column to the metric)
# To facilitate metric calculation, the grid should be iterable and indexable:
#  (for spatial metrics, this means iterating over the RA/Dec points)
#  (for global metrics, this means iterating over the visits based on divisions 
#   in a user-defined 'simDataSliceCol': for the base global grid, there is no split.)
# Grid metrics must also know how to set themselves up ('set up the grid'),
# read and write metric data, and generate plot representations of the metric data. 
# In order to compare metrics calculated on various grids, they must also be
#  able to at least check if two grids are equal. 

# TODO add read/write sql constraint & metric name

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pyfits as pyf

class BaseGrid(object):
    """Base class for all grid objects: sets required methods and implements common functionality."""

    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate the base grid object."""
        self.verbose = verbose
        self.badval = -666 #np.nan
        self.gridtype = None
        return

    def __len__(self):
        """Return npix, the number of pixels in the grid."""
        return self.npix

    def __iter__(self):
        """Iterate over the grid."""
        raise NotImplementedError()

    def next(self):
        """Set the gridvalues to return when iterating over grid."""
        raise NotImplementedError()

    def __getitem__(self):
        """Make grid indexable."""
        raise NotImplementedError()
    
    def __eq__(self, othergrid):
        """Evaluate if two grids are equivalent."""
        raise NotImplementedError()
    
    def sliceSimData(self, gridpoint, simDataCol, **kwargs):
        """Slice the simulation data appropriately for the grid.

        This slice of data should be the indices of the numpy rec array (the simData)
        which are appropriate for the metric to be working on, at that gridpoint."""
        raise NotImplementedError()

    def writeMetricData(self, outfilename, metricValues,
                    comment='', metricName='',
                    simDataName='', metadata='', gridfile='', badval=-666,dt='float'):
        head = pyf.Header()
        head.update(comment=comment, metricName=metricName,
                    simDataName=simDataName, metadata=metadata, gridfile=gridfile, gridtype=self.gridtype, badval=badval)
        if dt == 'object':            
            mask = []
            for val in metricValues:
                if np.size(val)==1:
                    mask.append(val == badval)
                else:
                    mask.append(False)
            mask = np.array(mask)
            ind = np.arange(len(metricValues))
            a1 = ind[mask]
            a2=ind[np.invert(mask)]
            #a1 = np.where(metricValues == badval)[0]
            #a2 = np.where(metricValues != badval)[0]
            try:
                metricValues[a2][0].shape #if this is just a single numpy array
            except:
                ncols = len(metricValues[a2][0]) #if it is a tuple or list 
            else:
                ncols = 1
            #import pdb; pdb.set_trace()
            cols = []
            column = np.empty(len(metricValues), dtype=object)                              
            if ncols == 1:
                for j in a1:  column[j] = np.array([badval])
                for j in a2:  column[j] = metricValues[j]
                column = pyf.Column(name='c'+str(0), format='PD()', array=column)
                cols.append(column)
            else:
                for i in np.arange(ncols):
                    column = np.empty(len(metricValues), dtype=object)    
                    #import pdb; pdb.set_trace()
                    for j in a1:  column[j] = np.array([badval]) #there has to be a better way to do this!
                    #column[a1] = badval
                    for j in a2:  column[j] = metricValues[j][i]
                    column = pyf.Column(name='c'+str(i), format='PD()', array=column) #need to update formatting.  Make a function to convert between dtype and fits format codes.
                    cols.append(column)
            tbhdu = pyf.new_table(cols)
            #append the info from head
            for i in range(len(head)):  tbhdu.header[head.keys()[i]]=head[i]
            tbhdu.writeto(outfilename+'.fits')
        else:              
            pyf.writeto(outfilename+'.fits', metricValues.astype(dt), head) 
        return
    
    def readMetricData(self,infilename):
        f = pyf.open(infilename)
        if f[0].header['NAXIS'] == 0:
            f = pyf.open(infilename)
            head = f[1].header
            badval = head['badval']
            metricValues = np.empty(len(f[1].data), dtype=object)
            #import pdb; pdb.set_trace()
            for i in range(len(f[1].data)): #stupid loop to unpack the variable length array table
                if len(np.unique(np.ravel((f[1].data[i])))) == 0:
                    metricValues[i] = f[1].data[i]
                else:                    
                    if np.max(np.unique(np.ravel((f[1].data[i])))) == badval:
                        metricValues[i] = badval
                    else:
                        metricValues[i] = f[1].data[i]
        else:
            metricValues, head = pyf.getdata(infilename, header=True)
        return metricValues, head['metricName'], \
            head['simDataName'],head['metadata'], head['comment'], head['gridfile'], head['gridtype']
        


    
    def plotHistogram(self, metricValue, metricLabel, title=None, 
                      fignum=None, legendLabel=None, addLegend=False, 
                      bins=100, cumulative=False, histRange=None, flipXaxis=False,
                      scale=1.0):
        """Plot a histogram of metricValue, labelled by metricLabel.

        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default 100)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)"""
        # Histogram metricValues. 
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        # Need to only use 'good' values in histogram.
        good = np.where(metricValue != self.badval)
        n, b, p = plt.hist(metricValue[good], bins=bins, histtype='step', 
                             cumulative=cumulative, range=histRange, label=legendLabel)
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(x,  pos):        
            return "%.3f" % (x * scale)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        plt.ylabel('Area (1000s of square degrees)')
        plt.xlabel(metricLabel)
        if flipXaxis:
            # Might be useful for magnitude scales.
            x0, x1 = plt.xlim()
            plt.xlim(x1, x0)
        if addLegend:
            plt.legend(fancybox=True, fontsize='smaller'), loc='upper left')
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse this if desired).         
        return fig.number
            
