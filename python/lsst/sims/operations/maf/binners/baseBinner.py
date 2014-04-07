# Base class for all 'Binner' objects. 
# 

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings
try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf

import lsst.sims.operations.maf.binners as binners
    
class BaseBinner(object):
    """Base class for all binners: sets required methods and implements common functionality."""

    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate the base binner object."""
        self.verbose = verbose
        self.badval = -666 
        self.binnertype = 'BASE'
        self.columnsNeeded=[]
        # Create a dict that saves how to re-init the binner
        self.binner_init = {'badval': self.badval}

    def setupBinner(self, *args, **kwargs):
        """Set up internal parameters and bins for binner. """
        # often args will be simData / sliceDataColName(s) / kwargs
        raise NotImplementedError()
    
    def __len__(self):
        """Return nbins, the number of bins in the binner. ."""
        return self.nbins

    def __iter__(self):
        """Iterate over the bins."""
        raise NotImplementedError()

    def next(self):
        """Define the bin values (interval or RA/Dec, etc.) to return when iterating over binner."""
        raise NotImplementedError()

    def __getitem__(self):
        """Make binner indexable."""
        raise NotImplementedError()
    
    def __eq__(self, otherBinner):
        """Evaluate if two binners are equivalent."""
        raise NotImplementedError()

    def _py2fitsFormat(self,pydtype):
        """Utility function to translate between python and pyfits data types."""
        convert_dict={'float64': 'D', 'int64': 'K', 'int32': 'J', 'float32': 'E'}
        result = 'P'+convert_dict[pydtype.name]+'()'
        return result
        
    def sliceSimData(self, binpoint):
        """Slice the simulation data appropriately for the binner.

        This slice of data should be the indices of the numpy rec array (the simData)
        which are appropriate for the metric to be working on, for that bin."""
        raise NotImplementedError()

    def writeData(self, outfilename, metricValues, metricName='', simDataName ='', comment='', metadata=''):
        """Save a set of metric values along with the information required to re-build the binner."""
        header = {}
        header['metricName']=metricName
        header['comment'] = comment
        header['metadata'] = metadata
        header['simDataName'] = simDataName
        
        binnerName=self.binnerName
        
        binner_init = self.binner_init
        binner_setup = self.binner_setup

        binner_bins = self.bins
        
        np.savez(outfilename, header=header, metricValues=metricValues, binner_init=binner_init, binner_setup=binner_setup, binnerName=binnerName, binner_bins=binner_bins)
      
    def readData(self, infilename):
        restored = np.load(infilename)
        metricValues, header, binner_init, binner_setup, binner_bins, binnerName = restored['metricValues'], restored['header'][()], restored['binner_init'][()], restored['binner_setup'][()], restored[binner_bins], restored['binnerName'][()]
        binner = getattr(binners,binnerName)(**binner_init)
        binner.bins = binner_bins
        return metricValues, binner, header


    def writeMetricDataGeneric(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666, dt=np.dtype('float64'), clobber=True):
        """Write metric data values to outfilename, preserving metadata. """
        head = pyf.Header()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            head.update(comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata,
                        binnertype=self.binnertype, int_badval=int_badval, badval=badval)
        if dt == 'object':
            try:
                ncols = len(metricValues.compressed()[0])            
            except:
                ncols = 1
            cols = []
            if ncols == 1:
                dt = metricValues.compressed()[0].dtype
                if dt.name[0:3] == 'int':
                    use_badval = int_badval
                else:
                    use_badval = badval
                #column = ma.filled(metricValues, use_badval)                
                column = pyf.Column(name='c'+str(0), format=self._py2fitsFormat(dt), array=metricValues)
                cols.append(column)
            else:
                good =np.where(metricValues.mask != True)[0][0]
                if type(metricValues[good][0]) != np.array: #I think this is right?  If not all data is getting saved, this line is probably the problem!  Maybe we should just numpy-save the arrays...could make the headers into simple dictionaries...
                    ncols = 1
                        
                for i in np.arange(ncols):
                    dt = np.array(metricValues.compressed()[0][i]).dtype
                    if dt.name[0:3] == 'int':
                        use_badval = int_badval
                    else:
                        use_badval = badval
                    column = np.empty(len(metricValues), 'object')
                    idx = np.where(metricValues.mask)
                    for j in idx[0]:
                        column[j] = np.array([use_badval,])
                    idx = np.where(~metricValues.mask)
                    if np.array(metricValues.mask).size == 1:  idx = np.where(metricValues.data != use_badval)
                    for j in idx[0]:
                        if type(metricValues.data[j][i]) != np.ndarray:
                            column[j] =np.array([metricValues.data[j][i],])
                        else:
                            column[j] =np.array(metricValues.data[j][i])
                    column = pyf.Column(name='c'+str(i), 
                                        format=self._py2fitsFormat(dt), array=column) 
                    cols.append(column)
            tbhdu = pyf.new_table(pyf.ColDefs(cols), header=head)
            # Append the info from head.
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            #    for i in range(len(head)):  tbhdu.header[head.keys()[i]]=head[i]
            tbhdu.writeto(outfilename, clobber=clobber)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                head.update(dtype = dt.name)
            if dt.name[0:3] == 'int':
                use_badval = int_badval
            else:
                use_badval = badval
            tt = ma.filled(metricValues, use_badval)
            pyf.writeto(outfilename, tt.astype(dt), head, clobber=clobber) 
    
    def readMetricDataGeneric(self, infilename):
        """Read metric data values from file 'infilename'.
        
        Return the metric values and the header, not the binner.  
        Note that this method does NOT automatically reconstruct the binner by itself, the
        readMetricData methods in each class does that using information read by this method."""
        f = pyf.open(infilename)
        if f[0].header['NAXIS'] == 0:
            f = pyf.open(infilename)
            head = f[1].header
            badval = head['badval']
            int_badval = head['int_badval']
            metricValues = np.empty(len(f[1].data), dtype=object)
            mask = []
            #import pdb ; pdb.set_trace()
            for arr in f[1].data:
                if np.size(arr) == 0:
                    mask.append(False)
                else:
                    mask.append(np.ravel(arr == np.array([badval]))[0] or 
                                np.ravel(arr == np.array([int_badval]))[0] )
            mask=np.array(mask)
            metricValues[np.where(mask == True)] = badval
            ind = np.where(mask == False)[0]
            for i in ind:  
                metricValues[i] = f[1].data[i] 
                #  This is still a stupid loop.  
                #  But, for some reason, the fits data thinks it's an int, so 
                #  I can't just unpack with metricValues[ind] = f[1].data[ind]
        else:
            metricValues, head = pyf.getdata(infilename, header=True)
            metricValues = metricValues.astype(head['dtype'])
            mask = np.where(metricValues == head['badval'], True, False)
            mask = np.where(metricValues == head['int_badval'], True, mask)
        # Convert metric values to masked array.  
        metricValues = ma.MaskedArray(data = metricValues,
                                      mask = mask,
                                      fill_value = self.badval)
        return metricValues, head
