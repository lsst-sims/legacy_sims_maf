# Base class for all 'Binner' objects. 
# 

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pyfits as pyf
import warnings

class BaseBinner(object):
    """Base class for all binners: sets required methods and implements common functionality."""

    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate the base binner object."""
        self.verbose = verbose
        self.badval = -666 
        self.binnertype = None
    
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
        
    def sliceSimData(self, binpoint, **kwargs):
        """Slice the simulation data appropriately for the binner.

        This slice of data should be the indices of the numpy rec array (the simData)
        which are appropriate for the metric to be working on, for that bin."""
        raise NotImplementedError()

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', binnerfile='', 
                        int_badval=-666, badval=-666, dt=np.dtype('float64')):
        """Write metric data values to outfilename, preserving metadata. """
        head = pyf.Header()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            head.update(comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, binnerfile=binnerfile,
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
                column = ma.filled(metricValues, use_badval)                
                column = pyf.Column(name='c'+str(0), format=self._py2fitsFormat(dt), array=column)
                cols.append(column)
            else:
                for i in np.arange(ncols):
                    dt = metricValues.compressed()[0][i].dtype
                    if dt.name[0:3] == 'int':
                        use_badval = int_badval
                    else:
                        use_badval = badval
                    column = np.empty(len(metricValues), 'object')
                    idx = np.where(metricValues.mask)
                    for j in idx[0]:
                        column[j] = np.array([use_badval,])
                    idx = np.where(~metricValues.mask)
                    for j in idx[0]:
                        column[j] = metricValues.data[j][i]
                    column = pyf.Column(name='c'+str(i), 
                                        format=self._py2fitsFormat(dt), array=column) 
                    cols.append(column)
            tbhdu = pyf.new_table(cols)
            # Append the info from head.
            for i in range(len(head)):  tbhdu.header[head.keys()[i]]=head[i]
            tbhdu.writeto(outfilename+'.fits')
        else:
            head.update(dtype = dt.name)
            if dt.name[0:3] == 'int':
                use_badval = int_badval
            else:
                use_badval = badval
            tt = ma.filled(metricValues, use_badval)
            pyf.writeto(outfilename+'.fits', tt.astype(dt), head) 
    
    def readMetricData(self, infilename):
        """Read metric data values from file 'infilename'."""
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
        # Convert metric values to masked array.  
        metricmasked = ma.MaskedArray(data = metricValues,
                                      mask = mask,
                                      fill_value = self.badval)
        return metricValues, head['metricName'], \
            head['simDataName'],head['metadata'], head['comment'], \
            head['binnerfile'], head['binnertype']
