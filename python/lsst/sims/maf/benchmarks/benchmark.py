import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.maps as maps
from lsst.sims.maf.utils import ColInfo

class Benchmark(object):
    """
    Benchmark holds a combination of a (single) metric, slicer and sqlconstraint, which determines
    a unique combination of an opsim evaluation.
    After the metric is evaluated over the slicer, it will hold the benchmark value (metric values) as well.
    It also holds a list of summary statistics to be calculated on those metric values, as well as the resulting
    summary statistic values.
    In addition, it holds plotting parameters (in plotDict) and display parameters for showMaf (in displayDict), as
    well as additional metadata such as the opsim run name.
    Benchmark can autogenerate some metadata, plotting labels, as well as generate plots, save output to disk, and calculate
    'reduce' methods on metrics.
    """
    def __init__(self, metric, slicer, stackerList=None,
                 sqlconstraint='', runName='opsim', metadata=None,
                 plotDict=None, displayDict=None,
                 summaryStats=None, mapsList=None,
                 fileRoot=None):
        # Set the metric.
        if not isinstance(metric, metrics.BaseMetric):
            raise ValueError('Metric must be an lsst.sims.maf.metrics object')
        self.metric = metric
        # Set the slicer.
        if not isinstance(slicer, slicers.BaseSlicer):
            raise ValueError('Slicer must be an lsst.sims.maf.slicers object')
        self.slicer = slicer
        # Set the 'maps' to apply to the slicer, if applicable.
        if mapsList is not None:
            if isinstance(mapsList, maps.BaseMaps):
                self.mapsList = [mapsList,]
            else:
                self.mapsList = []
                for m in mapsList:
                    if not isinstance(m, maps.BaseMap):
                        raise ValueError('MapsList must only contain lsst.sims.maf.maps objects')
                    self.mapsList.append(m)
        else:
            self.mapsList = None
        # Set the stackerlist if applicable.
        if stackerList is not None:
            if isinstance(stackerList, stackers.BaseStacker):
                self.stackerList = [stackerList,]
            else:
                self.stackerList = []
                for s in stackerList:
                    if not isinstance(s, stackers.BaseStacker):
                        raise ValueError('StackerList must only contain lsst.sims.maf.stackers objects')
                    self.stackerList.append(s)
        else:
            self.stackerList = None
        # Add the summary stats, if applicable.
        if summaryStats is not None:
            if isinstance(summaryStats, metrics.BaseMetric):
                self.summaryStats = [summaryStats]
            else:
                self.summaryStats = []
                for s in summaryStats:
                    if not isinstance(s, metrics.BaseMetric):
                        raise ValueError('SummaryStats must only contain lsst.sims.maf.metrics objects')
                    self.summaryStats.append(s)
        else:
            self.summaryStats = None
        # Set the sqlconstraint and metadata.
        self.sqlconstraint = sqlconstraint
        self.runName = runName
        self._buildMetadata(metadata)
        # Build the output filename root if not provided.
        if fileRoot is not None:
            self.fileRoot = fileRoot
        else:
            self._buildFileRoot()
        # Determine the columns needed from the database.
        self._findReqCols()
        
        # This is where we store the metric values and summary stats.
        self.metricValue = None
        self.summaryValues = None

        if plotDict is None:
            self.plotDict = {}
        else:
            self.plotDict = plotDict
        if displayDict is None:
            self.displayDict = {}
        else:
            self.displayDict = displayDict

    def _buildMetadata(self, metadata):
        """
        If no metadata is provided, process the sqlconstraint
        (by removing extra spaces, quotes, the word 'filter' and equal signs) to make a metadata version.
        e.g. 'filter = "r"' becomes 'r'
        """
        if metadata is None:
            self.metadata = self.sqlconstraint.replace('=','').replace('filter','').replace("'",'')
            self.metadata = self.metadata.replace('"', '').replace('  ',' ')
        else:
            self.metadata = metadata

    def _buildFileRoot(self):
        """
        Build an auto-generated output filename root (i.e. minus the plot type or .npz ending).
        """
        # Build basic version.
        self.fileRoot = '_'.join(self.runName, self.metric.metricName, self.metadata, self.slicer.slicerName[:4].upper())
        # Sanitize output name if needed.
        # Replace <, > and = signs.
        self.fileRoot = self.fileRoot.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
        # Strip white spaces (replace with underscores), strip '.'s and ','s
        self.fileRoot = self.fileRoot.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        # and strip quotes and double __'s
        self.fileRoot = self.fileRoot.replace('"','').replace("'",'').replace('__', '_')
        # and remove / and \
        self.fileRoot = self.fileRoot.replace('/', '_').replace('\\', '_')
        # and remove parentheses
        self.fileRoot = self.fileRoot.replace('(', '').replace(')', '')

    def _findReqCols(self):
        """
        Find the columns needed by the metrics, slicers, and stackers.
        If there are any additional stackers required, instatiate them and add them to the self.stackers list.
        (default stackers have to be instantiated to determine what additional columns are needed from database).
        """
        # Find the columns required  by the metrics and slicers (including if they come from stackers).
        colInfo = ColInfo()
        dbcolnames = set()
        defaultstackers = set()
        # Look for the source for the columns for the slicer.
        for col in self.slicer.columnsNeeded:
            colsource = colInfo.getDataSource(col)
            if colsource != colInfo.defaultDataSource:
                defaultstackers.add(colsource)
            else:
                dbcolnames.add(col)
        # Look for the source of columns in the metrics.
        for col in self.metric.colRegistry.colSet:
            colsource = colInfo.getDataSource(col)
            if colsource != colInfo.defaultDataSource:
                defaultstackers.add(colsource)
            else:
                dbcolnames.add(col)
        # Remove explicity instantiated stackers from defaultstacker set.
        for s in self.stackerList:
            if s.__class__ in defaultstackers:
                defaultstackers.remove(s.__class__)
        # Instantiate and add the remaining default stackers.
        for s in defaultstackers:
            self.stackerList.append(s())
        # Add the columns needed by all stackers to the list to grab from the database.
        for s in self.stackerList:
            for col in s.colsReq:
                dbcolnames.add(col)
        self.dbColNames = dbcolnames

