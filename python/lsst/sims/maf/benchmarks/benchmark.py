


class Benchmark(object):
    """
    This is an object to hold instatiated metrics, slicers, stackers etc.
    """
    def __init__(self, metric=None, slicer=None, stackers=None,
                 summaryStats=None, maps=None, sqlWhere='', dbAddress=None,
                 filenameBase=None, metadata=None, plotDict=None,
                 displayDict=None, outDir='Out'):

        self.metricValue = None
        self.summaryValues = None

        self.setMetric(metric)
        self.setSlicer(slicer)
        self.setStackers(stackers)
        self.setSummaryStats(summaryStats)
        self._buildMetadata(metadata)

        self.plotDict = plotDict
        self.displayDict = displayDict
        self.outDir = outDir
        self.maps = maps

    def setMetric(self, metric):
        self.metric = metric

    def setSlicer(self, slicer):
        self.slicer = slicer

    def setStackers(self, stackers):
        if stackers is None:
            stackers = []
        self.stackers = stackers

    def setSummaryStats(self, stats):
        self.summaryStats = stats

    def _buildMetadata(self, metadata):
        if metadata is None:
            pass
        else:
            self.metadata = metadata

    def findReqCols(self):
        """
        Find the columns needed by the metrics, slicers, and stackers.
        If there are any additional stackers required, instatiate them and add them to the self.stackers list.
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
        for col in self.metrics.colRegistry.colSet:
            colsource = colInfo.getDataSource(col)
            if colsource != colInfo.defaultDataSource:
                defaultstackers.add(colsource)
            else:
                dbcolnames.add(col)
        # Remove explicity instantiated stackers from defaultstacker set.
        for s in self.stackes:
            if s.__class__ in defaultstackers:
                defaultstackers.remove(s.__class__)
        # Instantiate and add the remaining default stackers.
        for s in defaultstackers:
            self.stackerObjs.add(s())
        # Add the columns needed by all stackers to the list to grab from the database.
        for s in self.stackerObjs:
            for col in s.colsReq:
                dbcolnames.add(col)
        return dbcolnames
