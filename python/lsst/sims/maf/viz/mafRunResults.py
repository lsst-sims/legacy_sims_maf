import os
from collections import OrderedDict
#import numpy as np
import pandas as pd
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles

__all__ = ['MafRunResults']

class MafRunResults(object):
    """
    Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages.

    Deals with a single MAF run (one output directory, one resultsDb) only. """
    def __init__(self, outDir, runName=None, resultsDb=None):
        """
        Instantiate the (individual run) layout visualization class.

        This class provides methods used by our jinja2 templates to help interact
        with the outputs of MAF.
        """
        self.outDir = os.path.relpath(outDir, '.')
        self.runName = runName

        self.configSummary = os.path.join(self.outDir, 'configSummary.txt')
        if not os.path.isfile(self.configSummary):
            self.configSummary = 'Config Summary Not Available'
        else:
            with open (self.configSummary, "r") as myfile:
                config=myfile.read()
            spot = config.find('RunName')
            if spot == -1:
                runName = None
            else:
                runName = config[spot:].split('\n')[0][8:]
        if self.runName is None:
            if runName is None:
                self.runName == 'RunName not available'
            else:
                self.runName = runName

        self.configDetails = os.path.join(self.outDir,'configDetails.txt')
        if not os.path.isfile(self.configDetails):
            self.configDetails = 'Config Details Not Available.'

        # Read in the results database.
        if resultsDb is None:
            resultsDb = os.path.join(self.outDir, 'resultsDb_sqlite.db')
        database = db.ResultsDb(database=resultsDb)

        # Get the metric and display info (1-1 match)
        self.metrics = database.getMetricDisplayInfo()
        self.metrics = self.sortMetrics(self.metrics)

        # Get the plot and stats info (many-1 metric match)
        self.stats = database.getSummaryStats()
        self.plots = database.getPlotFiles()

        # Pull up the names of the groups and subgroups.
        groups = sorted(self.metrics.displayGroup.unique())
        self.groups = OrderedDict()
        for g in groups:
            groupMetrics = self.metrics.query('displayGroup == @g')
            self.groups[g] = sorted(groupMetrics.displaySubgroup.unique())

        self.summaryStatOrder = ['Id', 'Identity', 'Median', 'Mean', 'Rms', 'RobustRms',
                                 'N(-3Sigma)', 'N(+3Sigma)', 'Count',
                                 '25th%ile', '75th%ile', 'Min', 'Max']
        # Add in the table fraction sorting to summary stat ordering.
        tableFractions = list(self.stats[self.stats['summaryName'].str.contains('TableFraction')].summaryName.unique())
        if len(tableFractions) > 0:
            tableFractions.remove('TableFraction 0 == P')
            tableFractions.remove('TableFraction 1 == P')
            tableFractions.remove('TableFraction 1 < P')
            tableFractions = sorted(tableFractions)
            self.summaryStatOrder.append('TableFraction 0 == P')
            for tableFrac in tableFractions:
                self.summaryStatOrder.append(tableFrac)
            self.summaryStatOrder.append('TableFraction 1 == P')
            self.summaryStatOrder.append('TableFraction 1 < P')

        self.plotOrder = ['SkyMap', 'Histogram', 'PowerSpectrum']

    ## Methods to deal with metricIds

    def convertSelectToMetrics(self, groupList, metricIdList):
        """
        Convert the lists of values returned by 'select metrics' template page
        into an appropriate dataframe of metrics (in sorted order).
        """
        metricIds = set()
        for group_subgroup in groupList:
            group = group_subgroup.split('_')[0]
            subgroup = group_subgroup.split('_')[-1].replace('+', ' ')
            mIds = self.metricIdsInSubgroup(group, subgroup)
            for mId in mIds:
                metricIds.add(mId)
        for mId in metricIdList:
            mId = int(mId)
            metricIds.add(mId)
        metrics = self.metricIdsToMetrics(metricIds)
        metrics = self.sortMetrics(metrics)
        return metrics

    def getJson(self, metric):
        """
        Return the JSON string containing the data for a particular metric.
        """
        filename = metric.metricDataFile
        if filename.upper() == 'NULL':
            return None
        datafile = os.path.join(self.outDir, filename)
        # Read data back into a  bundle.
        mB = metricBundles.emptyMetricBundle()
        mB.read(datafile)
        io = mB.outputJSON()
        if io is None:
            return None
        return io.getvalue()

    def getNpz(self, metric):
        """
        Return the npz data.
        """
        filename = metric.metricDataFile
        if filename.upper() == 'NULL':
            return None
        else:
            datafile = os.path.join(self.outDir, filename)
            return datafile

    def getResultsDb(self):
        """
        Return the summary results sqlite filename.
        """
        return os.path.join(self.outDir, 'resultsDb_sqlite.db')

    def metricIdsInSubgroup(self, group, subgroup):
        """
        Return the metricIds within a given group/subgroup.
        """
        metrics = self.metricsInSubgroup(group, subgroup)
        metricIds = list(metrics.metricId)
        return metricIds

    def metricIdsToMetrics(self, metricIds):
        """
        Return an ordered dataframe of metrics matching metricIds.
        """
        metrics = self.metrics.query('metricId in @metricIds')
        return metrics

    def metricsToMetricIds(self, metrics):
        """
        Return a list of the metric Ids corresponding to a subset of metrics.
        """
        return list(metrics.metricId)


    ## Methods to deal with metrics in numpy recarray.

    def sortMetrics(self, metrics, order=['displayGroup', 'displaySubgroup',
                                          'baseMetricNames', 'slicerName', 'displayOrder',
                                          'metricMetadata']):
        """
        Sort the metrics by order specified by 'order'.

        Default is to sort by group, subgroup, metric name, slicer, display order, then metadata.
        Returns sorted dataframe.
        """
        if len(metrics) > 0:
            metrics = metrics.sort(order)
        return metrics

    def metricsInGroup(self, group, metrics=None):
        """
        Given a group, return the metrics belonging to this group, in display order.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics.query('displayGroup == @group')
        metrics = self.sortMetrics(metrics)
        return metrics

    def metricsInSubgroup(self, group, subgroup, metrics=None):
        """
        Given a group and subgroup, return a dataframe of the metrics belonging to these group/subgroups, in display order.

        If 'metrics' is provided, then only consider this subset of metrics.
        """
        metrics = self.metricsInGroup(group, metrics)
        if len(metrics) > 0:
            metrics = metrics.query('displaySubgroup == @subgroup')
            metrics = self.sortMetrics(metrics)
        return metrics

    def metricsToSubgroups(self, metrics):
        """
        Given a dataframe of metrics, return an ordered dict of their group/subgroups.
        """
        groupList = sorted(metrics.displayGroup.unique())
        groups = OrderedDict()
        for g in groupList:
            groups[g] = sorted(metrics.query('displayGroup == @g').displaySubgroup.unique())
        return groups

    def metricsWithPlotType(self, plotType='SkyMap', metrics=None):
        """
        Return dataframe of metrics with plot=plotType (optional, metric subset).
        """
        # Allow some variation in plotType names for backward compatibility.
        plotTypes = [plotType]
        if plotType.endswith('lot'):
            plotTypes.append(plotType[:-4])
        else:
            plotTypes.append(plotType.lower() + 'Plot')
        if metrics is None:
            metrics = self.metrics
        # Identify the plots which would match (we'll then use their metricIds to re-select the metrics)
        plots = self.plots.query('(plotType in @plotTypes) and (metricId in @metrics.metricId)')
        metrics = metrics.query('metricId in @plots.metricId')
        return metrics

    def uniqueMetricNames(self, metrics=None, baseonly=True):
        """
        Return an array (list?) of the unique metric names, preserving the order of 'metrics'.
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            sortName = 'baseMetricNames'
        else:
            sortName = 'metricName'
        metricNames = list(metrics[sortName].unique())
        return metricNames

    def metricsWithSummaryStat(self, summaryStatName='Id', metrics=None):
        """
        Return metrics with summary stat matching 'summaryStatName' (optional, metric subset).
        """
        if metrics is None:
            metrics = self.metrics
        # Identify the matching stats.
        stats = self.stats.query('(summaryName == @summaryStatName) and (metricId in @metrics.metricId)')
        # Identify the subset of relevant metrics.
        metrics = metrics.query('metricId in @stats.metricId')
        # Re-sort metrics because at this point, probably want displayOrder + metadata before metric name.
        metrics = self.sortMetrics(metrics, order=['displayGroup', 'displaySubgroup', 'slicerName',
                                                   'displayOrder', 'metricMetadata', 'baseMetricNames'])
        return metrics

    def metricsWithStats(self, metrics=None):
        """
        Return metrics that have any summary stat.
        """
        if metrics is None:
            metrics = self.metrics
        # Identify metricIds which are also in stats.
        metrics = metrics.query('metricId in @self.stats.metricId')
        metrics = self.sortMetrics(metrics, order = ['displayGroup', 'displaySubgroup', 'slicerName',
                                                     'displayOrder', 'metricMetadata', 'baseMetricNames'])
        return metrics

    def uniqueSlicerNames(self, metrics=None):
        """
        For a dataframe metrics, return the unique slicer names.
        """
        if metrics is None:
            metrics = self.metrics
        return list(metrics.slicerName.unique())

    def metricsWithSlicer(self, slicer, metrics=None):
        """
        For a dataframe of metrics, return the subset which match a particular 'slicername' value.
        """
        if metrics is None:
            metrics = self.metrics
        return metrics.query('slicerName == @slicer')

    def uniqueMetricNameAndMetadata(self, metrics=None):
        """
        For a dataframe of metrics, return the unique metric names + metadata combo in same order.
        """
        if metrics is None:
            metrics = self.metrics
        metricmetadata = []
        for metricName, metadata in zip(metrics.metricName, metrics.metricMetadata):
            metricmeta = ' '.join([metricName, metadata])
            if metricmeta not in metricmetadata:
                metricmetadata.append(metricmeta)
        return metricmetadata

    def uniqueMetricMetadata(self, metrics=None):
        """
        For a dataframe of metrics, return a list of the unique metadata.
        """
        if metrics is None:
            metrics = self.metrics
        return list(metrics.metricMetadata.unique())

    def metricsWithMetadata(self, metadata, metrics=None):
        """
        For a dataframe of metrics, return the subset which match a particular 'metadata' value.
        """
        if metrics is None:
            metrics = self.metrics
        metrics = metrics.query('metricMetadata == @metadata')
        return metrics

    def metricsWithMetricName(self, metricName, metrics=None, baseonly=True):
        """
        Return all metrics which match metricName (default, only the 'base' metric name).
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            metrics = metrics.query('baseMetricNames == @metricName')
        else:
            metrics = metrics.query('metricName == @metricName')
        return metrics

    def metricInfo(self, metric=None, withDataLink=True, withSlicerName=True):
        """
        Return a dict with the metric info we want to show on the webpages.

        Currently : MetricName / Slicer/ Metadata / datafile (for download)
        Used to build a lot of tables in showMaf.
        """
        metricInfo = OrderedDict()
        if metric is None:
            metricInfo['MetricName'] = ''
            if withSlicerName:
                metricInfo['Slicer'] = ''
            metricInfo['Metadata'] = ''
            if withDataLink:
                metricInfo['Data'] = []
                metricInfo['Data'].append([None, None])
            return metricInfo
        # Otherwise, do this for real (not a blank).
        metricInfo['MetricName'] = metric.metricName
        if withSlicerName:
            metricInfo['Slicer'] = metric.slicerName
        metricInfo['Metadata'] = metric.metricMetadata
        if withDataLink:
            metricInfo['Data'] = []
            metricInfo['Data'].append(metric.metricDatafile)
            metricInfo['Data'].append(os.path.join(self.outDir, metric.metricDatafile))
        return metricInfo

    def captionForMetric(self, metric):
        """
        Return the caption for a given metric.
        """
        caption = metric['displayCaption']
        if caption == 'NULL':
            return ''
        else:
            return caption

    ## Methods for plots.

    def plotsForMetric(self, metric):
        """
        Return a dataframe of the plot which match a given metric.
        """
        return self.plots.query('metricId == @metric.metricId')

    def plotDict(self, plots=None):
        """
        Given a dataframe of plots (for a single metric usually).
        Returns an ordered dict with 'plotType' for interfacing with jinja2 templates.
          plotDict == {'SkyMap': {'plotFile': [], 'thumbFile', []}, 'Histogram': {}..}
          If no plot of a particular type, the plotFile and thumbFile are empty lists.
        Calling with plots=None returns a blank plotDict.
        """
        plotDict = OrderedDict()
        # Go through plots in 'plotOrder'.
        if plots is None:
            for p in self.plotOrder:
                plotDict[p] = {}
                plotDict[p]['plotFile'] = ''
                plotDict[p]['thumbFile'] = ''
        else:
            if isinstance(plots, pd.Series):
                p = plots.plotType
                plotDict[p] = {}
                plotDict[p]['plotFile'] = [self.getPlotfile(plots)]
                plotDict[p]['thumbFile'] = [self.getThumbfile(plots)]
            elif isinstance(plots, pd.DataFrame):
                plotTypes = list(plots.plotType.unique())
                for p in self.plotOrder:
                    if p in plotTypes:
                        plotDict[p] = {}
                        plotmatch = plots.query('plotType == @p')
                        plotDict[p]['plotFile'] = list(plotmatch.apply(self.getPlotfile, axis=1))
                        plotDict[p]['thumbFile'] = list(plotmatch.apply(self.getThumbfile, axis=1))
                        plotTypes.remove(p)

                # Round up remaining plots.
                for p in plotTypes:
                    plotDict[p] = {}
                    plotmatch = plots.query('plotType == @p')
                    plotDict[p]['plotFile'] = list(plotmatch.apply(self.getPlotfile, axis=1))
                    plotDict[p]['thumbFile'] = list(plotmatch.apply(self.getThumbfile, axis=1))
        return plotDict

    def getThumbfile(self, plot):
        """
        Return the thumbnail file name for a given plot.
        """
        thumbfile = os.path.join(self.outDir, plot.thumbFile)
        return thumbfile

    def getPlotfile(self, plot):
        """
        Return the filename for a given plot.
        """
        plotFile = os.path.join(self.outDir, plot.plotFile)
        return plotFile

    def orderPlots(self, skyPlots):
        """
        skyPlots = dataframe of skymap plots.

        Returns an ordered list of plotDicts.

        The goal is to lay out the skymaps in a 3x2 grid on the MultiColor page, in ugrizy order.
        If a plot for a filter is missing, add a gap. (i.e. if there is no u, keep a blank spot).
        If there are other plots, with multiple filters or no filter info, they are added to the end.

        If skyPlots includes multiple plots in the same filter, just goes back to displayOrder.
        """
        orderedSkyPlots = []
        if len(skyPlots) == 0:
            return orderedSkyPlots

        orderList = ['u','g','r','i','z','y']
        blankPlotDict = self.plotDict(None)

        # Look for filter names in the plot filenames.
        tooManyPlots = False
        for f in orderList:
            pattern = '_'+f+'_'
            matchSkyPlot = skyPlots[skyPlots.plotFile.str.contains(pattern)]
            if len(matchSkyPlot) == 1:
                orderedSkyPlots.append(self.plotDict(matchSkyPlot))
            elif len(matchSkyPlot) == 0:
                orderedSkyPlots.append(blankPlotDict)
            else:
                # If we found more than one plot in the same filter, we just go back to displayOrder.
                tooManyPlots = True
                break

        if not (tooManyPlots):
            # Add on any additional non-filter plots (e.g. joint completeness)
            pattern = '_[ugrizy]_' # for regex
            nonmatchSkyPlots = skyPlots[skyPlots.plotFile.str.contains(pattern, regex=True) == False]
            if len(nonmatchSkyPlots) > 0:
                for i, skyPlot in nonmatchSkyPlots.iterrows():
                    orderedSkyPlots.append(self.plotDict(skyPlot))

        elif tooManyPlots:
            metrics = self.metrics.query('metricId in @skyPlots.metricId')
            metrics = self.sortMetrics(metrics, order=['displayOrder'])
            orderedSkyPlots = []
            for i,m in metrics.iterrows():
                skyPlot = skyPlots.query('metricId == @m.metricId')
                orderedSkyPlots.append(self.plotDict(skyPlot))

        # Pad out to make sure there are rows of 3
        while len(orderedSkyPlots) % 3 != 0:
            orderedSkyPlots.append(blankPlotDict)

        return orderedSkyPlots

    def getSkyMaps(self, metrics=None):
        """
        Return a dataframe of the skymaps, optionally for subset of metrics.
        """
        if metrics is None:
            metrics = self.metrics
        skymatchPlots = self.plots.query('(metricId in @metrics.metricId) and (plotType=="SkyMap")')
        return skymatchPlots

    ## Set of methods to deal with summary stats.

    def statsForMetric(self, metric, statName=None):
        """
        Return a dataframe of summary statistics which match a given metric(s).

        Optionally specify a particular statName that you want to match.
        """
        metricIds = metric.metricId
        # If you call statsForMetric from an metrics.iterrows loop, metricId will be an int.
        if isinstance(metricIds, int):
            stats = self.stats.query('metricId == @metricIds')
        # Otherwise, if you call with normal dataframe, metricId will be a Series.
        else:
            stats = self.stats.query('metricId in @metricIds')
        if statName is not None:
            stats = stats.query('summaryName == @statName')
        return stats


    def statDict(self, stats):
        """
        Returns an ordered dictionary with statName:statValue for dataframe of stats.

        Note that if you pass 'stats' from multiple metrics with the same summary names, they
         will be overwritten in the resulting dictionary!
         So just use stats from one metric, with unique summaryNames.
        """
        # Result = dict with key == summary stat name, value = summary stat value.
        sdict = OrderedDict()
        statnames = self.orderStatNames(stats)
        for n in statnames:
            match = stats.query('summaryName == @n')
            # We're only going to look at the first value; and this should be a float.
            sdict[n] = match.summaryValue.iloc[0]
        return sdict

    def orderStatNames(self, stats):
        """
        Given a dataframe of stats, return a list containing all the unique 'summaryNames'
        in a default ordering (identity-count-mean-median-rms..).
        """
        names = list(stats.summaryName.unique())
        # Add some default sorting:
        namelist = []
        for nord in self.summaryStatOrder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist

    def allStatNames(self, metrics):
        """
        Given a dataframe of metrics, return a list containing all the unique 'summaryNames'
        in a default ordering.
        """
        names = list(self.stats.query('metricId in @metrics.metricId').summaryName.unique())
        # Add some default sorting.
        namelist = []
        for nord in self.summaryStatOrder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist
