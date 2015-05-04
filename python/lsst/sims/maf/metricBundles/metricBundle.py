import os
from copy import deepcopy
import numpy as np
import numpy.ma as ma

import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.maps as maps
from lsst.sims.maf.utils import ColInfo


class MetricBundle(object):
    """
    MetricBundle holds a combination of a (single) metric, slicer and sqlconstraint, which determines
    a unique combination of an opsim evaluation.
    After the metric is evaluated over the slicer, it will hold the MetricBundle value (metric values) as well.
    It also holds a list of metrics (in summaryMetrics) to be used to generate summary statistics on the metric values,
    as well as the resulting summary statistic values.
    In addition, it holds plotting parameters (in plotDict) and display parameters for showMaf (in displayDict), as
    well as additional metadata such as the opsim run name.
    MetricBundle can autogenerate some metadata, plotting labels, as well as generate plots,
    save output to disk, and calculate 'reduce' methods on metrics.
    """
    def __init__(self, metric, slicer, sqlconstraint,
                 stackerList=None, runName='opsim', metadata=None,
                 plotDict=None, displayDict=None,
                 summaryMetrics=None, mapsList=None,
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
            if isinstance(mapsList, maps.BaseMap):
                self.mapsList = [mapsList,]
            else:
                self.mapsList = []
                for m in mapsList:
                    if not isinstance(m, maps.BaseMap):
                        raise ValueError('mapsList must only contain lsst.sims.maf.maps objects')
                    self.mapsList.append(m)
        else:
            self.mapsList = []
        # Add the summary stats, if applicable.
        self.setSummaryMetrics(summaryMetrics)
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

    def _resetMetricBundle(self):
        """
        Reset all properties of MetricBundle.
        """
        self.metric = None
        self.slicer = None
        self.sqlconstraint = ''
        self.stackerList = []
        self.summaryMetrics = []
        self.mapsList = None
        self.runName = 'opsim'
        self.metadata = ''
        self.dbCols = None
        self.fileRoot = None
        self.plotDict = {}
        self.displayDict = {}
        self.metricValues = None
        self.summaryValues = None

    def _setupMetricValues(self):
        """
        Set up the numpy masked array to store the metric value data.
        """
        dtype = self.metric.metricDtype
        # Can't store healpix slicer mask values in an int array. 
        if dtype == 'int':
            dtype = 'float'
        self.metricValues = ma.MaskedArray(data = np.empty(len(self.slicer), dtype),
                                            mask = np.zeros(len(self.slicer), 'bool'),
                                            fill_value= self.slicer.badval)

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

    def setSummaryMetrics(self, summaryMetrics):
        """
        Set (or reset) the summary metrics for the metricbundle.
        """
        if summaryMetrics is not None:
            if isinstance(summaryMetrics, metrics.BaseMetric):
                self.summaryMetrics = [summaryMetrics]
            else:
                self.summaryMetrics = []
                for s in summaryMetrics:
                    if not isinstance(s, metrics.BaseMetric):
                        raise ValueError('SummaryStats must only contain lsst.sims.maf.metrics objects')
                    self.summaryMetrics.append(s)
        else:
            # Add identity metric to unislicer metric values (to get them into resultsDB).
            if self.slicer.slicerName == 'UniSlicer':
                self.summaryMetrics = [metrics.IdentityMetric('metricdata')]
            else:
                self.summaryMetrics = []

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
            xlabel = self.slicer.sliceColName  + ' (' + self.slicer.sliceColUnits + ')'
            tmpPlotDict['ylabel'] = ylabel
            tmpPlotDict['xlabel'] = xlabel
        else:
            xlabel = self.metric.name  + ' (' + self.metric.units + ')'
            tmpPlotDict['xlabel'] = xlabel
        if self.metric.metricDtype == 'int':
            tmpPlotDict['cbarFormat'] = '%d'
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
            # And if the user didn't specify cbarFormat (but we did, thinking it was an integer) - remove int format.
            elif 'cbarFormat' not in plotDict:
                del tmpPlotDict['cbarFormat']
        if 'normVal' in tmpPlotDict:
            if tmpPlotDict['normVal'] == 0:
                warnings.warn('Warning! Plot normalization value for %s was 0: removing normVal from plotDict'
                              % (self.fileRoot))
                del tmpPlotDict['normVal']
            # And if the user didn't specify cbarFormat (but we did, thinking it was an integer) - remove int format.
            elif 'cbarFormat' not in plotDict:
                del tmpPlotDict['cbarFormat']
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
            if self.metric.comment is None:
                caption = self.metric.name + ' calculated on a %s' %(self.slicer.slicerName)
                caption += ' basis, using a subset of data selected via %s.' %(self.sqlconstraint)
            else:
                caption = self.metric.comment
            if 'zp' in self.plotDict:
                caption += ' Values plotted with a zeropoint of %.2f.' %(self.plotDict['zp'])
            if 'normVal' in self.plotDict:
                caption += ' Values plotted with a normalization value of %.2f.' %(self.plotDict['normVal'])
            self.displayDict['caption'] = caption
        if resultsDb:
            # Update the display values in the resultsDb.
            metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                              self.runName, self.sqlconstraint,
                                              self.metadata, outfile)
            resultsDb.updateDisplay(metricId, self.displayDict)

    def write(self, comment='', outDir='.', outfileSuffix=None, resultsDb=None):
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


    def read(self, filename):
        """
        Read metricValues and associated metadata from disk.
        Overwrites any data currently in metricbundle.
        """
        self._resetMetricBundle()
        # Set up a base slicer to read data (we don't know type yet).
        baseslicer = slicers.BaseSlicer()
        # Use baseslicer to read file.
        metricValues, slicer, header = baseslicer.readData(filename)
        self.slicer = slicer
        self.metricValues = metricValues
        self.metricValues.fill_value = slicer.badval
        # It's difficult to reinstantiate the metric object, as we don't
        # know what it is necessarily -- the metricName can be changed.
        self.metric = metrics.BaseMetric()
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
        path, head = os.path.split(filename)
        self.fileRoot = head.replace('.npz', '')

    def computeSummaryStats(self, resultsDb=None):
        """
        Compute summary statistics on metricValues, using summaryMetrics (metricbundle list).
        """
        if self.summaryMetrics is None:
            self.summaryValues = None
        else:
            # Build array of metric values, to use for (most) summary statistics.
            rarr_std = np.array(zip(self.metricValues.compressed()),
                                dtype=[('metricdata', self.metricValues.dtype)])
            self.summaryValues = []
            for m in self.summaryMetrics:
                # The summary metric colname should already be set to 'metricdata', but in case it's not:
                m.colname = 'metricdata'
                summaryName = m.name.replace(' metricdata', '').replace(' None', '')
                if hasattr(m, 'maskVal'):
                    # summary metric requests to use the mask value, as specified by itself, rather than skipping masked vals.
                    rarr = np.array(zip(self.metricValues.filled(m.maskVal)),
                                    dtype=[('metricdata', self.metricValues.dtype)])
                else:
                    rarr = rarr_std
                if np.size(rarr) == 0:
                    summaryVal = self.slicer.badval
                else:
                    summaryVal = m.run(rarr)
                self.summaryValues.append([summaryName, summaryVal])
                # Add summary metric info to results database, if applicable.
                if resultsDb:
                    metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                                      self.runName, self.sqlconstraint, self.metadata, None)
                    resultsDb.updateSummaryStat(metricId, summaryName=summaryName, summaryValue=summaryVal)

    def reduceMetric(self, reduceFunc, reducePlotDict=None, reduceDisplayDict=None):
        """
        Run 'reduceFunc' (any function that operates on self.metricValues), return a new MetricBundle.
        Typically reduceFunc will be the metric reduce functions, as they are tailored to expect the
          metricValues format.
        reduceDisplayDict and reducePlotDicts are displayDicts and plotDicts to be applied to the new metricBundle.
        """
        # Generate a name for the metric values processed by the reduceFunc.
        reduceName = self.metric.name + '_' + reduceFunc.__name__.replace('reduce', '')
        # Set up metricBundle to store new metric values, and add plotDict/displayDict.
        newmetricBundle = MetricBundle(metric=deepcopy(self.metric), slicer=self.slicer, stackerList=self.stackerList,
                                 sqlconstraint=self.sqlconstraint, metadata=self.metadata, runName=self.runName,
                                 plotDict=None, displayDict=self.displayDict,
                                 summaryMetrics=self.summaryMetrics, mapsList=self.mapsList, fileRoot='')
        newmetricBundle.metric.name = reduceName
        if reducePlotDict is not None:
            if 'units' in reducePlotDict:
                newmetricBundle.metric.units = reducePlotDict['units']
        # Build a new output file root name.
        newmetricBundle._buildFileRoot()
        # Use existing (self) plotDict, without the title/x or y labels (as these get updated with reduceName)
        cpPlotDict = {}
        for k, v in self.plotDict.iteritems():
            if k not in newmetricBundle.plotDict:
                cpPlotDict[k] = v
        # Then update newmetricBundle's plot dictionary with these values (copied from self).
        newmetricBundle.setPlotDict(cpPlotDict)
        # And update newmetricBundle's plot dictionary with any set explicitly by reducePlotDict.
        newmetricBundle.setPlotDict(reducePlotDict)
        # Update the newmetricBundle's display dictionary with any set explicitly by reduceDisplayDict.
        newmetricBundle.setDisplayDict(reduceDisplayDict)
        # Set up new metricBundle's metricValues masked arrays, copying metricValue's mask.
        newmetricBundle.metricValues = ma.MaskedArray(data = np.empty(len(self.slicer), 'float'),
                                                    mask = self.metricValues.mask,
                                                    fill_value = self.slicer.badval)
        # Fill the reduced metric data using the reduce function.
        for i, (mVal, mMask) in enumerate(zip(self.metricValues.data, self.metricValues.mask)):
            if not mMask:
                newmetricbundle.metricValues.data[i] = reduceFunc(mVal)
        return newmetricbundle

    def plot(self, outDir='.', outfileSuffix=None, resultsDb=None, savefig=True,
             figformat='pdf', dpi=600, thumbnail=True, plotFunc=None):
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
                                            filename=os.path.join(outDir, outfile),
                                            thumbnail = thumbnail, plotFunc=plotFunc,
                                            **self.plotDict)
        # Save information about the plotted files.
        if resultsDb:
            metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                              self.runName, self.sqlconstraint, self.metadata, None)
            for filename, filetype in zip(plotResults['filenames'], plotResults['filetypes']):
                froot, fname = os.path.split(filename)
                resultsDb.updatePlot(metricId=metricId, plotType=filetype, plotFile=fname)
        return plotResults['figs']
