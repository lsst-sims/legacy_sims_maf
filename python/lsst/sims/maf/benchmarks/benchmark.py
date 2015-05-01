import numpy as np
import numpy.ma as ma

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
    Benchmark can autogenerate some metadata, plotting labels, as well as generate plots,
    save output to disk, and calculate 'reduce' methods on metrics.
    """
    def __init__(self, metric, slicer, sqlconstraint,
                 stackerList=None, runName='opsim', metadata=None,
                 plotDict=None, displayDict=None,
                 summaryStats=None, mapsList=None,
                 fileRoot=None):
        # Set the metric.
        if not isinstance(metric, metrics.BaseMetric):
            raise ValueError('metric must be an lsst.sims.maf.metrics object')
        self.metric = metric
        # Set the slicer.
        if not isinstance(slicer, slicers.BaseSlicer):
            raise ValueError('slicer must be an lsst.sims.maf.slicers object')
        self.slicer = slicer
        # Set the sqlconstraint.
        self.sqlconstraint = sqlconstraint
        # Set the stackerlist if applicable.
        if stackerList is not None:
            if isinstance(stackerList, stackers.BaseStacker):
                self.stackerList = [stackerList,]
            else:
                self.stackerList = []
                for s in stackerList:
                    if not isinstance(s, stackers.BaseStacker):
                        raise ValueError('stackerList must only contain lsst.sims.maf.stackers objs')
                    self.stackerList.append(s)
        else:
            self.stackerList = []
        # Set the 'maps' to apply to the slicer, if applicable.
        if mapsList is not None:
            if isinstance(mapsList, maps.BaseMaps):
                self.mapsList = [mapsList,]
            else:
                self.mapsList = []
                for m in mapsList:
                    if not isinstance(m, maps.BaseMap):
                        raise ValueError('mapsList must only contain lsst.sims.maf.maps objects')
                    self.mapsList.append(m)
        else:
            self.mapsList = None
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
            # Add identity metric to unislicer metric values (to get them into resultsDB).
            if self.slicer.slicerName == 'UniSlicer':
                self.summaryStats = [metrics.IdentityMetric('metricdata')]
            else:
                self.summaryStats = []
        # Set the provenance/metadata.
        self.runName = runName
        self._buildMetadata(metadata)
        # Build the output filename root if not provided.
        if fileRoot is not None:
            self.fileRoot = fileRoot
        else:
            self._buildFileRoot()
        # Determine the columns needed from the database.
        self._findReqCols()
        # Set the plotDict and displayDicts.
        self.plotDict = {}
        self.setPlotDict(plotDict)
        # Update/set displayDict.
        self.displayDict = {}
        self.setDisplayDict(displayDict)
        # This is where we store the metric values and summary stats.
        self.metricValues = None
        self.summaryValues = None

    def _resetBenchmark(self):
        """
        Reset all properties of benchmark.
        """
        self.metric = None
        self.slicer = None
        self.sqlconstraint = ''
        self.stackerList = []
        self.summaryStats = []
        self.mapsList = None
        self.runName = 'opsim'
        self.metadata = ''
        self.dbCols = None
        self.fileRoot = None
        self.plotDict = {}
        self.displayDict = {}
        self.metricValues = None
        self.summaryValues = None

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
        self.fileRoot = '_'.join([self.runName, self.metric.name, self.metadata,
                                  self.slicer.slicerName[:4].upper()])
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
        # Look for the source of columns for this metric (only).
        # We can't use the colRegistry here because we want the columns for this metric only.
        for col in self.metric.colNameArr:
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
        # Remove 'metricdata' from dbcols if it ended here by default.
        if 'metricdata' in dbcolnames:
            dbcolnames.remove('metricdata')
        self.dbCols = dbcolnames

    def setPlotDict(self, plotDict=None):
        """
        Set or update any property of plotDict.
        Will set default values.
        """
        # Set up a temporary dictionary with the default values.
        tmpPlotDict = {}
        tmpPlotDict['units'] = self.metric.units
        title = self.runName + ' ' + self.metadata + ': ' + self.metric.name
        tmpPlotDict['title'] = title
        if self.slicer.slicerName == 'OneDSlicer':
            ylabel = self.metric.name + ' (' + self.metric.units + ')'
            xlabel = self.slicer.sliceColName  + ' (' + self.slicers.sliceColUnits + ')'
            tmpPlotDict['ylabel'] = ylabel
            tmpPlotDict['xlabel'] = xlabel
        else:
            xlabel = self.metric.name  + ' (' + self.metric.units + ')'
            tmpPlotDict['xlabel'] = xlabel
        # Update from self.plotDict (to use existing values, if present).
        tmpPlotDict.update(self.plotDict)
        # And then update from any values being passed now.
        if plotDict is not None:
            tmpPlotDict.update(plotDict)
        # Check for bad zp or normVal values.
        if 'zp' in tmpPlotDict:
            if not np.isfinite(self.plotDict['zp']):
                warnings.warn('Warning! Plot zp for %s was infinite: removing zp from plotDict' %(self.fileRoot))
                del tmpPlotDict['zp']
        if 'normVal' in tmpPlotDict:
            if tmpPlotDict['normVal'] == 0:
                warnings.warn('Warning! Plot normalization value for %s was 0: removing normVal from plotDict'
                              % (self.fileRoot))
                del tmpPlotDict['normVal']
        # Reset self.displayDict to this updated dictionary.
        self.plotDict = tmpPlotDict

    def setDisplayDict(self, displayDict=None, resultsDb=None):
        """
        Set or update any property of displayDict.
        Will set default values.
        """
        # Set up a temporary dictionary with the default values.
        tmpDisplayDict = {'group':None, 'subgroup':None, 'order':0, 'caption':None}
        # Update from self.displayDict (to use existing values, if present).
        tmpDisplayDict.update(self.displayDict)
        # And then update from any values being passed now.
        if displayDict is not None:
            tmpDisplayDict.update(displayDict)
        # Reset self.displayDict to this updated dictionary.
        self.displayDict = tmpDisplayDict
        # If we still need to auto-generate a caption, do it.
        if self.displayDict['caption'] is None:
            caption = self.metric.name + ' calculated on a %s' %(self.slicer.slicerName)
            caption += ' basis, using a subset of data selected via %s.' %(self.sqlconstraint)
            if 'zp' in self.plotDict:
              caption += ' Values plotted with a zeropoint of %.2f.' %(self.plotDict['zp'])
            if 'normVal' in self.plotDict:
              caption += ' Values plotted with a normalization value of %.2f.' %(self.plotDict['normVal'])
            self.displayDict['caption'] = caption
        if resultsDb is not None:
            # Update the display values in the resultsDb.
            metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                              self.runName, self.sqlconstraint,
                                              self.metadata, outfile)
            resultsDb.updateDisplay(metricId, self.displayDict)

    def writeBenchmark(self, comment='', outDir='.', outfileSuffix=None, resultsDb=None):
        """
        Write metricValues (and associated metadata) to disk.

        comment = any additional comments to add to output file (beyond
                   metric name, simDataName, and metadata).
        outfileSuffix = additional suffix to add to output files (numerical suffix for movies).
        """
        if outfileSuffix is not None:
            outfile = self.fileRoot + outfileSuffix + '.npz'
        else:
            outfile = self.fileRoot + '.npz'
        self.slicer.writeData(os.path.join(outDir, outfile),
                                self.metricValues,
                                metricName = self.metric.name,
                                simDataName = self.runName,
                                sqlconstraint = self.sqlconstraint,
                                metadata = self.metadata + comment,
                                displayDict = self.displayDict,
                                plotDict = self.plotDict)
        if resultsDb:
            metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                              self.runName, self.sqlconstraint,
                                              self.metadata, outfile)
            resultsDb.updateDisplay(metricId, self.displayDict)

    def outputJSON(self):
        """
        Set up and call the baseSlicer outputJSON method, to output to IO string.
        """
        io = self.slicer.outputJSON(self.metricValues,
                                    metricName = self.metric.name,
                                    simDataName = self.runName,
                                    metadata = self.metadata,
                                    plotDict = self.plotDict)
        return io


    def readBenchmark(self, filename):
        """
        Read metricValues and associated metadata from disk.
        Overwrites any data currently in benchmark.
        """
        self._resetBenchmark()
        # Set up a base slicer to read data (we don't know type yet).
        baseslicer = slicers.BaseSlicer()
        # Use baseslicer to read file.
        metricValues, slicer, header = baseslicer.readData(filename)
        self.slicer = slicer
        self.metricValues = metricValues
        self.metricValues.fill_value = slicer.badval
        # It's difficult to reinstantiate the metric object, as we don't
        # know what it is necessarily -- the metricName can be changed.
        self.metric = metrics.BaseMetric(col='metricdata')
        # But, for plot label building, we do need to try to recreate the
        #  metric name and units.
        self.metric.name = header['metricName']
        if 'plotDict' in header:
            if 'units' in header['plotDict']:
                self.metric.units = header['plotDict']['units']
        else:
            self.metric.units = ''
        self.runName = header['simDataName']
        self.sqlconstraint = header['sqlconstraint']
        self.metadata = header['metadata']
        if self.metadata is None:
            self._buildMetadata()
        if 'plotDict' in header:
            self.setPlotDict(header['plotDict'])
        if 'displayDict' in header:
            self.setDisplayDict(header['displayDict'])

    def computeSummaryStatistics(self, resultsDb=None):
        """
        Compute summary statistics on benchmark metricValues, using summaryStats (benchmark list).
        """
        if self.summaryStats is None:
            self.summaryValues = None
        else:
            self.summaryValues = []
            for m in self.summaryStats:
                mName = m.name.replace(' metricdata', '')
                if hasattr(m, 'maskVal'):
                    # summary metric requests to use the mask value, as specified by itself, rather than skipping masked vals.
                    rarr = np.array(zip(self.metricValues.filled(summaryMetric.maskVal)),
                                    dtype=[('metricdata', self.metricValues.dtype)])
                else:
                    rarr = np.array(zip(self.metricValues.compressed()),
                                dtype=[('metricdata', self.metricValues.dtype)])
                # The summary metric colname should already be set to 'metricdata', but in case it's not:
                m.colname = 'metricdata'
                if np.size(rarr) == 0:
                    summaryVal = self.slicer.badval
                else:
                    summaryVal = m.run(rarr)
                self.summaryValues.append([mName, summaryVal])
                # Add summary metric info to results database, if applicable.
                if self.resultsDb:
                    metricId = resultsDb.updateMetric(self.metric.metricName, self.slicer.slicerName,
                                                      self.runName, self.sqlconstraint, self.metadata, None)
                    resultsDb.updateSummaryStat(metricId, summaryName=mName, summaryValue=summaryVal)

    def reduceMetric(self, reduceFunc, reducePlotDict=None, reduceDisplayDict=None):
        """
        Run 'reduceFunc' (any function that operates on self.metricValues).
        Typically reduceFunc will be the metric reduce functions, as they are tailored to expect the
          metricValues format.
        reduceDisplayDict and reducePlotDicts are displayDicts and plotDicts to be applied to the new benchmark.
        """
        # Generate a name for the metric values processed by the reduceFunc.
        reduceName = self.metric.name + '_' + reduceFunc.__name__.replace('reduce', '')
        # Set up benchmark to store new metric values, and add plotDict/displayDict.
        newbenchmark = Benchmark(metric=metrics.BaseMetric('metricdata'), slicer=self.slicer, stackerList=self.stackerList,
                                 sqlconstraint=self.sqlconstraint, metadata=self.metadata, runName=self.runName,
                                 plotDict=self.plotDict, displayDict=self.displayDict,
                                 summaryStats=self.summaryStats, mapsList=self.mapsList, fileRoot=self.fileRoot)
        newbenchmark.metric.name = reduceName
        if 'units' in reducePlotDict:
            newbenchmark.metric.units = reducePlotDict['units']
        newbenchmark.setPlotDict(reducePlotDict)
        newbenchmark.setDisplayDict(reduceDisplayDict)
        # Set up new benchmark's metricValues masked arrays, copying metricValue's mask.
        newbenchmark.metricValues = ma.MaskedArray(data = np.empty(len(self.slicer), 'float'),
                                                    mask = self.metricValues.mask,
                                                    fill_value = self.slicer.badval)
        for i, (mVal, mMask) in enumerate(zip(self.metricValues.data, self.metricValues.mask)):
            if not mMask:
                newbenchmark.metricValues.data[i] = reduceFunc(mVal)
        return newbenchmark

    def plotBenchmark(self, outDir='.', outfileSuffix=None, resultsDb=None, savefig=True,
                      figformat='pdf', dpi=600, thumbnail=True):
        """
        Create all plots available from the slicer.
        """
        # plotData for each slicer returns a dictionary with the filenames, filetypes, and fig nums.
        if outfileSuffix is not None:
            outfile = self.fileRoot + outfileSuffix
        else:
            outfile = self.fileRoot
        # Make plots.
        plotResults = self.slicer.plotData(self.metricValues, savefig=savefig,
                                           figformat=figformat, dpi=dpi,
                                           filename=os.path.join(self.outDir, outfile),
                                           thumbnail = thumbnail, **self.plotDict)
        # Save information about the plotted files.
        if resultsDb:
            metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                              self.runName, self.sqlconstraint, self.metadata, None)
            for filename, filetype in zip(plotResults['filenames'], plotResults['filetypes']):
                froot, fname = os.path.split(filename)
                resultsDb.updatePlot(metricId=metricId, plotType=filetype, plotFile=fname)
        return plotResults['figs']
