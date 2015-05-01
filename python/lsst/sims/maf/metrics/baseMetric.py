# Base class for metrics - defines methods which must be implemented.
# If a metric calculates a vector or list at each gridpoint, then there
#  should be additional 'reduce_*' functions defined, to convert the vector
#  into scalar (and thus plottable) values at each gridpoint.
# The philosophy behind keeping the vector instead of the scalar at each gridpoint
#  is that these vectors may be expensive to compute; by keeping/writing the full
#  vector we permit multiple 'reduce' functions to be executed on the same data.

import numpy as np
import warnings
import inspect
from lsst.sims.maf.utils.getColInfo import ColInfo

__all__ = ['MetricRegistry', 'BaseMetric']

class MetricRegistry(type):
    """
    Meta class for metrics, to build a registry of metric classes.
    """
    def __init__(cls, name, bases, dict):
        super(MetricRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__
        if modname.startswith('lsst.sims.maf.metrics'):
            modname = ''
        else:
            if len(modname.split('.')) > 1:
                modname = '.'.join(modname.split('.')[:-1]) + '.'
            else:
                modname = modname + '.'
        metricname = modname + name
        if metricname in cls.registry:
            raise Exception('Redefining metric %s! (there are >1 metrics with the same name)' %(metricname))
        if metricname not in ['BaseMetric', 'SimpleScalarMetric']:
            cls.registry[metricname] = cls
    def getClass(cls, metricname):
        return cls.registry[metricname]
    def list(cls, doc=False):
        for metricname in sorted(cls.registry):
            if not doc:
                print metricname
            if doc:
                print '---- ', metricname, ' ----'
                print inspect.getdoc(cls.registry[metricname])
    def help(cls, metricname):
        print metricname
        print inspect.getdoc(cls.registry[metricname])
        args, varargs, kwargs, defaults = inspect.getargspec(cls.registry[metricname].__init__)
        args_with_defaults = args[-len(defaults):]
        print ' Metric __init__ keyword args and defaults: '
        for a, d in zip(args_with_defaults, defaults):
            print '     ', a, d


class ColRegistry(object):
    """
    ColRegistry tracks the columns needed for all metric objects (kept internally in a set).

    ColRegistry.colSet returns a list of all unique columns required for metrics;
    ColRegistry.dbCols returns the subset of these which come from the database.
    ColRegistry.stackerCols returns the dictionary of [columns: stacker class].
    """
    colInfo = ColInfo()
    def __init__(self):
        self.colSet = set()
        self.dbSet = set()
        self.stackerDict = {}
    def addCols(self, colArray):
        """
        Add the columns in colArray into the ColRegistry set and identifies their source, using utils.ColInfo.
        """
        for col in colArray:
            self.colSet.add(col)
            source = self.colInfo.getDataSource(col)
            if source == self.colInfo.defaultDataSource:
                self.dbSet.add(col)
            else:
                if col not in self.stackerDict:
                    self.stackerDict[col] = source


class BaseMetric(object):
    """Base class for the metrics."""
    __metaclass__ = MetricRegistry
    colRegistry = ColRegistry()
    colInfo = ColInfo()

    def __init__(self, col=None, metricName=None, maps=None, units=None,
                 metricDtype=None, badval=-666):
        """Instantiate metric.

        'col' is a kwarg for purposes of the MAF driver; when actually using a metric, it must be set to
        the names of the data columns that the metric will operate on. This can be a single string or a list.

        'maps' is a list of any maps that the metric will need, accessed via slicePoint that is passed from the slicer.

        After inheriting from this base metric :
          * every metric object will have metricDtype (the type of data it calculates) set according to:
               -- kwarg (metricDtype='float', 'int', etc)
               -- 'float' (assumes float if not specified in kwarg)
               -- 'object' (if reduce functions are present and value not set in kwarg)
          * every metric object will have the data columns it requires added to the column registry
            (so the driver can know which columns to pull from the database)
          * every metric object will contain a plotDict dictionary, which may contain only the units.

        plotDict is a dictionary containing instructions for plotting (setting plot titles, limits,
        x and y labels, etc.).
        displayGroup is a string defining where the output of the metric should be displayed in the visualization
        layer (metrics with the same displayGroup parameter are grouped together).
        """
        # Turn cols into numpy array so we know we can iterate over the columns.
        self.colNameArr = np.array(col, copy=False, ndmin=1)
        # To support simple metrics operating on a single column, set self.colname
        if len(self.colNameArr) == 1:
            self.colname = self.colNameArr[0]
        # Add the columns to the colRegistry.
        self.colRegistry.addCols(self.colNameArr)
        # Set the maps that are needed:
        if maps is None:
            maps = []
        self.maps = maps
        # Value to return if the metric can't be computed
        self.badval = badval
        # Save a unique name for the metric.
        self.name = metricName
        if self.name is None:
            # If none provided, construct our own from the class name and the data columns.
            self.name = self.__class__.__name__.replace('Metric', '', 1) + ' ' + \
              ', '.join(map(str, self.colNameArr))
        # Set up dictionary of reduce functions (may be empty).
        self.reduceFuncs = {}
        self.reduceOrder = {}
        for r in inspect.getmembers(self, predicate=inspect.ismethod):
            if r[0].startswith('reduce'):
                reducename = r[0].replace('reduce', '', 1)
                self.reduceFuncs[reducename] = r[1]
                self.reduceOrder[reducename] = 0
        # Identify type of metric return value.
        if metricDtype is not None:
            self.metricDtype = metricDtype
        elif len(self.reduceFuncs.keys()) > 0:
            self.metricDtype = 'object'
        else:
            self.metricDtype = 'float'
        # Set physical units, for plotting purposes.
        if units is None:
            units = ' '.join([self.colInfo.getUnits(colName) for colName in self.colNameArr])
            if len(units.replace(' ', '')) == 0:
                units = ''
        self.units = units

    def run(self, dataSlice, slicePoint=None):
        raise NotImplementedError('Please implement your metric calculation.')
