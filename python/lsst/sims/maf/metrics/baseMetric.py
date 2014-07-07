# Base class for metrics - defines methods which must be implemented.
# If a metric calculates a vector or list at each gridpoint, then there
#  should be additional 'reduce_*' functions defined, to convert the vector
#  into scalar (and thus plottable) values at each gridpoint.
# The philosophy behind keeping the vector instead of the scalar at each gridpoint
#  is that these vectors may be expensive to compute; by keeping/writing the full
#  vector we permit multiple 'reduce' functions to be executed on the same data.

import numpy as np
import inspect
from lsst.sims.maf.utils.getColInfo import ColInfo

class MetricRegistry(type):
    """
    Meta class for metrics, to build a registry of metric classes.
    """
    def __init__(cls, name, bases, dict):
        super(MetricRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        if cls in cls.registry:
            warnings.warn('Redefining metric %s! (there are >1 metrics with the same name)' %(name))
        if name not in ['BaseMetric', 'SimpleScalarMetric']:
            cls.registry[name] = cls            
    def metricClass(cls, name):
        return cls.registry[name]
    def listMetrics(cls, docs=False):
        for metricName in sorted(cls.registry):
            if not docs:
                print metricName
            if docs:
                print '---- ', metricName, ' ----'
                print cls.registry[metricName].__doc__
            
            
class ColRegistry(object):
    """
    ColRegistry tracks the columns needed for all metric objects (kept internally in a set). 

    ColRegistry.uniqueCols returns a list of all unique columns required for metrics;
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
                    #tmpStacker = getattr(addCols, stacker)()
                    #for c in tmpStacker.colsReq:
                    #    self.dbSet.add(c)            
    def uniqueCols(self):
        """
        Returns a list of the unique columns used for all metrics.
        """
        return list(self.colSet)    
    def dbCols(self):
        """
        Returns the list of unique columns needed from the database (including columns required by default stackers).
        """
        return list(self.dbSet)
    def stackerCols(self):
        return self.stackerDict
            
                       
class BaseMetric(object):
    """Base class for the metrics."""
    __metaclass__ = MetricRegistry
    colRegistry = ColRegistry()
    colInfo = ColInfo()
    
    def __init__(self, cols, metricName=None, units=None, plotParams=None, metricDtype='object',
                 *args, **kwargs):
        """Instantiate metric.
        After inheriting from this base metric (and using, perhaps via 'super' this __init__):
          * every metric object will have the data columns it requires added to the column registry
            (so the driver can know which columns to pull from the database)
          * every metric object will contain a plotParams dictionary, which may contain only the units.
        """
        # Turn cols into numpy array so we know we can iterate over the columns.
        self.colNameArr = np.array(cols, copy=False, ndmin=1)
        # Add the columns to the colRegistry.
        self.colRegistry.addCols(self.colNameArr)
        # Identify type of metric return value. Default 'object'.
        #  Individual metrics may override with more specific value.
        self.metricDtype = metricDtype
        # Value to return if the metric can't be computed
        self.badval = -666
        # Save a unique name for the metric.
        self.name = metricName
        if self.name is None:
            # If none provided, construct our own from the class name and the data columns.
            self.name = self.__class__.__name__.replace('Metric', '', 1) + ' ' + \
              ' '.join(map(str, self.colNameArr))
        # Set up dictionary of reduce functions (may be empty).
        self.reduceFuncs = {}
        for r in inspect.getmembers(self, predicate=inspect.ismethod):
            if r[0].startswith('reduce'):
                reducename = r[0].replace('reduce', '', 1)
                self.reduceFuncs[reducename] = r[1]
        # Set physical units, for plotting purposes.
        # (If plotParams has 'units' this will be ignored). 
        if units is None:
            units = ' '.join([self.colInfo.getUnits(col) for col in self.colNameArr])
            if len(units.replace(' ', '')) == 0:
                units = ''
        self.units = units
        # Set more plotting preferences (at the very least, the units).
        self.plotParams = plotParams
        if self.plotParams is None:
            self.plotParams = {}
        if 'units' not in self.plotParams:
            self.plotParams['units'] = self.units
        # Example options for plotting parameters: plotTitle, plotMin, plotMax,
        #  plotPercentiles (overriden by plotMin/Max). 
        #  These plotParams are used by the binMetric, passed to the binner plotting utilities.

    def run(self, dataSlice, *args):
        raise NotImplementedError('Please implement your metric calculation.')
