# Base class for all 'Slicer' objects. 
# 

import inspect
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings
from lsst.sims.maf.utils import getDateVersion

class BaseSlicer(object):
    """
    Base class for all slicers: sets required methods and implements common functionality.
    """
    def __init__(self, verbose=True, badval=-666, *args, **kwargs):
        """Instantiate the base slicer object."""
        # After init: everything necessary for using slicer for plotting or saving/restoring metric
        #   data should be present (although slicer does not need to be able to slice data again).
        #   Variables in this init need to be set for slicer to work as such.
        # 
        # Args will include sliceDataCols and other data names that must be fetched from DB
        self.verbose = verbose
        self.badval = badval
        # The sliceMetric has a 'memo-ize' functionality that can save previous indexes & return
        #  metric data value calculated for same set of previous indexes, if desired.
        #  CacheSize = 0 effectively turns this off, otherwise cacheSize should be set by the slicer.
        #  (Most useful for healpix slicer, where many healpixels may have same set of LSST visits). 
        self.cacheSize = 0
        # Set length of Slicer.
        self.nslice = None
        self.slicePoints = {}
        self.slicerName = self.__class__.__name__
        self.columnsNeeded = []
        # Add dictionary of plotting methods for each slicer.
        self.plotFuncs = {}
        for p in inspect.getmembers(self, predicate=inspect.ismethod):
            if p[0].startswith('plot'):
                if p[0] == 'plotData':
                    pass
                else:
                    self.plotFuncs[p[0]] = p[1]
        # Create a dict that saves how to re-init the slicer.
        #  This may not be the whole set of args/kwargs, but those which carry useful metadata or
        #   are absolutely necesary for init.
        # Will often be overwritten by individual slicer slicer_init dictionaries.
        self.slicer_init = {'badval':badval}
        
    def setupSlicer(self, *args):
        """
        Set up internal parameters necessary for slicer to slice data and generates indexes on simData.
        Also sets _sliceSimData for a particular slicer.
        """
        # Typically args will be simData, but opsimFieldSlicer also uses fieldData.
        raise NotImplementedError()


    def getSlicePoints(self):
        """
        Return the slicePoint metadata, for all slice points.
        """
        return self.slicePoints
    
    def __len__(self):
        """
        Return nbins, the number of bins in the slicer.
        """
        return self.nslice

    def __iter__(self):
        """Iterate over the slices.
        """
        self.islice = 0
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
        islice = self.islice
        self.islice += 1
        return self._sliceSimData(islice)

    def __getitem__(self, islice):
        return self._sliceSimData(islice)
    
    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        raise NotImplementedError()

    def _sliceSimData(self, slicePoint):
        """
        Slice the simulation data appropriately for the slicer.

        Given the identifying slicePoint metadata
        The slice of data returned will be the indices of the numpy rec array (the simData)
        which are appropriate for the metric to be working on, for that bin.
        """
        raise NotImplementedError('This method is set up by "setupSlicer" - run that first.')

    def writeData(self, outfilename, metricValues, metricName='',
                  simDataName ='', sqlconstraint='', metadata=''):
        """
        Save metric values along with the information required to re-build the slicer.
        """
        header = {}
        header['metricName']=metricName
        header['sqlconstraint'] = sqlconstraint
        header['metadata'] = metadata
        header['simDataName'] = simDataName
        date, versionInfo = getDateVersion()
        header['dateRan'] = date
        for key in versionInfo.keys():
            header[key] = versionInfo[key]
        if hasattr(metricValues, 'mask'): # If it is a masked array
            data = metricValues.data
            mask = metricValues.mask
            fill = metricValues.fill_value
        else:
            data = metricValues
            mask = None
            fill = None
        # npz file acts like dictionary: each keyword/value pair below acts as a dictionary in loaded NPZ file.
        np.savez(outfilename,
                 header = header, # header saved as dictionary
                 metricValues = data, # metric data values
                 mask = mask, # metric mask values
                 fill = fill, # metric badval/fill val
                 slicer_init = self.slicer_init, # dictionary of instantiation parameters
                 slicerName = self.slicerName, # class name
                 slicePoints = self.getSlicePoints(), # slicePoint metadata saved (is a dictionary)
                 slicerNSlice = self.nslice)
                                 
    def readData(self, infilename):
        """
        Read metric data from disk, along with the info to rebuild the slicer (minus new slicing capability). 
        """
        import lsst.sims.maf.slicers as slicers
        restored = np.load(infilename)
        # Get metric data set
        if restored['mask'][()] is None:
            metricValues = ma.MaskedArray(data=restored['metricValues'])
        else:
            metricValues = ma.MaskedArray(data=restored['metricValues'],
                                          mask=restored['mask'],
                                          fill_value=restored['fill'])
        # Get Metadata & other simData info
        header = restored['header'][()]  # extra brackets restore dictionary to dictionary status
        # Get slicer instantiated.
        slicer_init = restored['slicer_init'][()]
        slicer = getattr(slicers, str(restored['slicerName']))(**slicer_init)
        # Restore slicePoint metadata.
        slicer.nslice = restored['slicerNSlice']
        slicer.slicePoints = restored['slicePoints'][()]
        return metricValues, slicer, header
    
    def plotData(self, metricValues, figformat='pdf', dpi=600, filename='fig', savefig=True, **kwargs):
        """
        Call all available plotting methods.
        """
        # If passed metric data which is not a simple data type, return without plotting.
        # (thus - override this method if your slicer requires plotting complex 'object' data.
        filenames=[]
        filetypes=[]
        figs={}
        if not (metricValues.dtype == 'float') or (metricValues.dtype == 'int'):
            warnings.warn('Metric data type not float or int. No plots generated.')
            return {'figs':figs, 'filenames':filenames, 'filetypes':filetypes}
        # Otherwise, plot.
        for p in self.plotFuncs:
            plottype = p.replace('plot', '')
            figs[plottype] = self.plotFuncs[p](metricValues, **kwargs)
            if savefig:
                outfile = filename + '_' + plottype + '.' + figformat
                plt.savefig(outfile, figformat=figformat, dpi=dpi)
                filenames.append(outfile)
                filetypes.append(plottype)
        return {'figs':figs, 'filenames':filenames, 'filetypes':filetypes}

        
