# Base class for metrics - defines methods which must be implemented.
# If a metric calculates a vector or list at each gridpoint, then there
#  should be additional 'reduce_*' functions defined, to convert the vector
#  into scalar (and thus plottable) values at each gridpoint.
# The philosophy behind keeping the vector instead of the scalar at each gridpoint
#  is that these vectors may be expensive to compute; by keeping/writing the full
#  vector we permit multiple 'reduce' functions to be executed on the same data.

import numpy as np

# ClassRegistry adds some extras to a normal dictionary.
class ClassRegistry(dict):
    @staticmethod
    def makeColArr(cols):
        #Promote scalar to array.  Solution from:
        #http://stackoverflow.com/questions/12653120/how-can-i-make-a-numpy-function-that-accepts-a-numpy-array-an-iterable-or-a-sc
        return np.array(cols, copy=False, ndmin=1)
    # Contents of the classRegistry dictionary look like {metricClassName: 'set' of [simData columns]}
    def __str__(self):
        # Print the contents of the registry nicely.
        retstr = "----------------------------\n"
        retstr += "Registry Contents\n"
        for k in self:
            retstr += "%s: %s\n"%(k, ",".join([str(el) for el in self[k]]))
        retstr += "-----------------------------\n"
        return retstr
    def __setitem__(self, i, y):
        if not hasattr(y, '__iter__'):
            raise TypeError("Can only contain iterable types")
        super(ClassRegistry, self).__setitem__(i,y)
    def uniqueCols(self):
        colset = set()
        for k in self:
            for col in self[k]:
                colset.add(col)
        return colset    
    

class BaseMetric(object):
    """Base class for the metrics."""
    # Add ClassRegistry to keep track of columns needed for metrics. 
    classRegistry = ClassRegistry()
    
    def __init__(self, cols, metricName=None, units=None,
                 *args, **kwargs):
        """Instantiate metric. """
        # Turn cols into numpy array (so we know it's iterable).
        self.colNameList = ClassRegistry.makeColArr(cols)
        # Register the columns in the classRegistry.
        self.registerCols(self.colNameList)
        # Identify type of metric return value. Default 'object'.
        #  Individual metrics should override with more specific value.
        self.metricDtype = 'object'
        # Value to return if the metric can't be computed
        self.badval = -666
        # Save a name for the metric + the data it's working on, so we
        #  can identify this later.
        if metricName:
            self.name = metricName
        else:
            # Else construct our own name from the class name and the data columns.
            allcols = ''
            for i in self.colNameList:
                allcols += '_' + i
            self.name = self.__class__.__name__.replace('Metric', '', 1) + allcols
        # Set physical units, mostly for plotting purposes.
        if units:
            self.units = units
        else:
            # Else look up units in dictionary lookup table. 
            self.units = 'TODO'
        return

    def registerCols(self, cols):
        """Add cols to the column registry. """
        # Set myName to be name of the metric class.
        myName = self.__class__.__name__
        if myName not in self.classRegistry:
            #Add a set to the registry if the key doesn't exist.
            self.classRegistry[myName] = set()
        # Add the columns to the registry.
        for col in cols:
            self.classRegistry[myName].add(col)

    def validateData(self, simData):
        """Check that simData has necessary columns for this particular metric."""
        ## Note that we can also use the class registry to find the list of all columns.
        for col in self.colNameList:
            try:
                simData[col]
            except KeyError:
                raise KeyError('Could not find data column for metric: %s' %(col))
    
    def run(self, dataSlice):
        raise NotImplementedError('Please implement your metric calculation.')
