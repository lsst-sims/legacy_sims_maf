# Base class for metrics - defines methods which must be implemented.
# If a metric calculates a vector or list at each gridpoint, then there
#  should be additional 'reduce_*' functions defined, to convert the vector
#  into scalar (and thus plottable) values at each gridpoint.
# The philosophy behind keeping the vector instead of the scalar at each gridpoint
#  is that these vectors may be expensive to compute; by keeping/writing the full
#  vector we permit multiple 'reduce' functions to be executed on the same data.


# ClassRegistry adds some extras to a normal dictionary.
class ClassRegistry(dict):
    # Contents of the dictionary look like {metricClassName: 'set' of [simData columns]}
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
        for k in self.keys():
            for col in self[k]:
                colset.add(col)
        return colset
        

class BaseMetric(object):
    """Base class for the metrics."""
    # Add ClassRegistry to keep track of columns needed for metrics. 
    classRegistry = ClassRegistry()
    
    def __init__(self, cols, metricName, *args, **kwargs):
        """Instantiate metric. """
        # Register the columns in the classRegistry.
        self.registerCols(cols)
        # Save a name for the metric + the data it's working on, so we
        #  can identify this later.
        if metricName:
            self.name = metricName
        else: # Construct our own name.
            if hasattr(cols, '__iter__'):
                self.name = self.__class__.__name__.rstrip('Metric') + '_' + cols[0]
            else:
                self.name = self.__class__.__name__.rstrip('Metric') + '_' + cols
        # Set size of metric return value (scalar = 1, vector = X, variableList= None)
        self.metricLen = 1
        return

    def registerCols(self, cols):
        """Add cols to the column registry. """
        # Set up the method to add the columns to the registry.
        def addCols(cols, cRegistrySet):
            # Check if cols is a list or a string.
            if hasattr(cols, '__iter__'):  # list
                for col in cols:
                    cRegistrySet.add(col)
            else: #string
                cRegistrySet.add(cols)
        # Set myName to be name of the metric class.
        myName = self.__class__.__name__
        # Add the columns to the registry, if the class is already registered.
        if myName in self.classRegistry:
            addCols(cols, self.classRegistry[myName])
        # Add the columns to the registry, when the class is not there yet.
        else:
            self.classRegistry[myName] = set()
            addCols(cols, self.classRegistry[myName])
        return

    def validateData(self, simData):
        """Check that simData has necessary columns for this particular metric."""
        ## Note that we can also use the class registry to find the list of all columns.
        # If colnames is an iterator (i.e. is a list or set, etc), iterate over it.
        if has_attr(self.colname, '__iter__'):
            for col in self.colname:
                testCol(col)
        # Otherwise colnames is a single  value.
        testCol(self.colname)
        # Here's the test on the simData. 
        def testCol(col): 
            try:
                simData[col]
            except KeyError:
                raise KeyError('Could not find data column for metric: %s' %(col))
        return
    
    def run(self, dataSlice):
        raise NotImplementedError('Please implement your metric calculation.')

    def reduce(self, reduce_function):
        raise NotImplementedError()
    
