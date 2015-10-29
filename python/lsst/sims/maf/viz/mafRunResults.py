import os
from collections import OrderedDict
import numpy as np
from numpy.lib.recfunctions import rec_join, merge_arrays
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
            resultsDb = os.path.join(outDir, 'resultsDb_sqlite.db')
        database = db.Database(resultsDb, longstrings=True,
                               dbTables={'metrics':['metrics','metricID'] ,
                                         'displays':['displays', 'displayId'],
                                         'plots':['plots','plotId'],
                                         'stats':['summarystats','statId']})
        # Just pull all three tables.
        # Below, we provide some methods to interface between the numpy rec arrays returned
        #  by these queries and what the templates need.
        # The idea being that this should make the template code & presentation layer more
        #  easily maintainable in the future.
        self.metrics = database.queryDatabase('metrics', 'select * from metrics')
        self.displays = database.queryDatabase('displays', 'select * from displays')
        # Combine metrics and displays arrays (these are one-to-one).
        self.metrics = rec_join('metricId', self.metrics, self.displays)
        # Add base metric names (to keep order for reduce methods).
        baseNames = np.empty(len(self.metrics), dtype=[('baseMetricNames', '|S50')])
        for i, m in enumerate(self.metrics):
            baseNames['baseMetricNames'][i] = m['metricName'].split('_')[0]
        self.metrics = merge_arrays([self.metrics, baseNames], flatten=True, usemask=False)
        self.metrics = self.sortMetrics(self.metrics)
        del self.displays
        # Get plot and summary stat info.
        self.plots = database.queryDatabase('plots', 'select * from plots')
        self.stats = database.queryDatabase('stats', 'select * from summarystats')

        # Make empty arrays if there was nothing in the database
        if len(self.plots) == 0:
            self.plots = np.zeros(0, dtype=[('metricId',int), ('plotFile', '|S10')])
        if len(self.stats) == 0:
            self.stats = np.zeros(0, dtype=[('metricId',int), ('summaryName', '|S10'), ('summaryValue', float)])

        # Pull up the names of the groups and subgroups.
        groups = sorted(list(np.unique(self.metrics['displayGroup'])))
        self.groups = OrderedDict()
        for g in groups:
            self.groups[g] = set()
        for metric in self.metrics:
            self.groups[metric['displayGroup']].add(metric['displaySubgroup'])
        for g in self.groups:
            self.groups[g] = sorted(list(self.groups[g]))

        self.summaryStatOrder = ['Id', 'Identity', 'Median', 'Mean', 'Rms', 'RobustRms',
                                 'N(-3Sigma)', 'N(+3Sigma)', 'Count', '%ile']
        # Add in the table fraction sorting to summary stat ordering.
        tableFractions = list(set([name for name in self.stats['summaryName'] if 'TableFraction' in name]))
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
        into an appropriate numpy recarray of metrics (in sorted order).
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
        return self.sortMetrics(metrics)

    def getJson(self, metric):
        """
        Return the JSON string containing the data for a particular metric.
        """
        filename = metric['metricDataFile'][0]
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
        filename = metric['metricDataFile'][0]
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
        metricIds = metrics['metricId']
        return list(metricIds)

    def metricIdsToMetrics(self, metricIds):
        """
        Return an ordered numpy recarray of metrics matching metricIds.
        """
        metrics = np.empty(len(metricIds), dtype=self.metrics.dtype)
        for i, mId in enumerate(metricIds):
            match = np.where(self.metrics['metricId'] == mId)
            metrics[i] = self.metrics[match]
        return metrics

    def metricsToMetricIds(self, metrics):
        """
        Return a list of the metric Ids corresponding to a subset of metrics.
        """
        metricIds = []
        for m in metrics:
            metricIds.append(m['metricId'])
        return metricIds


    ## Methods to deal with metrics in numpy recarray.

    def sortMetrics(self, metrics, order=['displayGroup', 'displaySubgroup',
                                          'baseMetricNames', 'slicerName', 'displayOrder',
                                          'metricMetadata']):
        """
        Sort the metrics by order specified by 'order'.

        Default is to sort by group, subgroup, metric name, slicer, display order, then metadata.
        """
        return np.sort(metrics, order=order)

    def metricsInGroup(self, group, metrics=None):
        """
        Given a group, return the metrics belonging to this group, in display order.
        """
        if metrics is None:
            metrics = self.metrics
        match = np.where(metrics['displayGroup'] == group)
        return self.sortMetrics(metrics[match])

    def metricsInSubgroup(self, group, subgroup, metrics=None):
        """
        Given a group and subgroup, return the metrics belonging to these group/subgroups, in display order.

        If 'metrics' is provided, then only consider this subset of metrics.
        """
        metrics = self.metricsInGroup(group, metrics)
        match = np.where(metrics['displaySubgroup'] == subgroup)
        metrics = metrics[match]
        return self.sortMetrics(metrics)

    def metricsToSubgroups(self, metrics):
        """
        Given a recarray of metrics, return an ordered dict of their group/subgroups.
        """
        metrics = self.sortMetrics(metrics)
        grouplist= sorted(list(np.unique(metrics['displayGroup'])))
        groups = OrderedDict()
        for g in grouplist:
            groups[g] = set()
        for metric in metrics:
            groups[metric['displayGroup']].add(metric['displaySubgroup'])
        for g in groups:
            groups[g] = sorted(list(groups[g]))
        return groups

    def metricsWithPlotType(self, plotType='SkyMap', metrics=None):
        """
        Return recarray of metrics with plot=plotType (optional, metric subset).
        """
        if metrics is None:
            metrics = self.metrics
        hasplot = np.zeros(len(metrics))
        for i, m in enumerate(metrics):
            match = np.where(self.plots['metricId'] == m['metricId'])
            matchType = np.where(self.plots['plotType'][match] == plotType)
            if len(self.plots[matchType]) > 0:
                hasplot[i] = 1
        metrics = metrics[np.where(hasplot > 0)]
        return metrics

    def uniqueMetricNames(self, metrics=None, baseonly=True):
        """
        Return a list of the unique metric names, preserving the order of 'metrics'.
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            sortName = 'baseMetricNames'
        else:
            sortName = 'metricName'
        metricNames = []
        for m in metrics:
            if m[sortName] not in metricNames:
                metricNames.append(m[sortName])
        return metricNames

    def metricsWithSummaryStat(self, summaryStatName='Id', metrics=None):
        """
        Return metrics with summary stat matching 'summaryStatName' (optional, metric subset).
        """
        if metrics is None:
            metrics = self.metrics
        hasstat = np.zeros(len(metrics))
        for i, m in enumerate(metrics):
            match = np.where(self.stats['metricId'] == m['metricId'])
            matchStat = np.where(self.stats['summaryName'][match] == summaryStatName)
            if len(self.stats[matchStat]) > 0:
                hasstat[i] = 1
        metrics = metrics[np.where(hasstat > 0)]
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
        hasstat = np.zeros(len(metrics))
        for i, m in enumerate(metrics):
            match = np.where(self.stats['metricId'] == m['metricId'])
            if len(self.stats[match]) > 0:
                hasstat[i] = 1
        metrics = metrics[np.where(hasstat > 0)]
        metrics = self.sortMetrics(metrics, order = ['displayGroup', 'displaySubgroup', 'slicerName',
                                                    'displayOrder', 'metricMetadata', 'baseMetricNames'])
        return metrics

    def uniqueSlicerNames(self, metrics=None):
        """
        For a recarray of metrics, return the unique slicer names.
        """
        if metrics is None:
            metrics = self.metrics
        slicernames = []
        for m in metrics:
            if m['slicerName'] not in slicernames:
                slicernames.append(m['slicerName'])
        return slicernames

    def metricsWithSlicer(self, slicer, metrics=None):
        """
        For a recarray of metrics, return the subset which match a particular 'slicername' value.
        """
        if metrics is None:
            metrics = self.metrics
        match = np.where(metrics['slicerName'] == slicer)
        return metrics[match]

    def uniqueMetricNameAndMetadata(self, metrics=None):
        """
        For a recarray of metrics, return the unique metric names + metadata combo.
        """
        if metrics is None:
            metrics = self.metrics
        metricmetadata = []
        for m in metrics:
            metricmeta = ' '.join([m['metricName'], m['metricMetadata']])
            if metricmeta not in metricmetadata:
                metricmetadata.append(metricmeta)
        return metricmetadata

    def uniqueMetricMetadata(self, metrics=None):
        """
        For a recarray of metrics, return the unique metadata.
        """
        if metrics is None:
            metrics = self.metrics
        metadata = []
        for m in metrics:
            if m['metricMetadata'] not in metadata:
                metadata.append(m['metricMetadata'])
        return metadata

    def metricsWithMetadata(self, metadata, metrics=None):
        """
        For a recarray of metrics, return the subset which match a particular 'metadata' value.
        """
        if metrics is None:
            metrics = self.metrics
        match = np.where(metrics['metricMetadata'] == metadata)
        return metrics[match]

    def metricsWithMetricName(self, metricName, metrics=None, baseonly=True):
        """
        Return all metrics which match metricName (default, only the 'base' metric name).
        """
        if metrics is None:
            metrics = self.metrics
        if baseonly:
            match = np.where(metrics['baseMetricNames'] == metricName)
        else:
            match = np.where(metrics['metricName'] == metricName)
        return metrics[match]

    def metricInfo(self, metric, withDataLink=True, withSlicerName=True):
        """
        Return a dict with the metric info we want to show on the webpages.

        Currently : MetricName / Slicer/ Metadata / datafile (for download)
        Used to build a lot of tables in showMaf.
        """
        metricInfo = OrderedDict()
        metricInfo['MetricName'] = metric['metricName']
        if withSlicerName:
            metricInfo['Slicer'] = metric['slicerName']
        metricInfo['Metadata'] = metric['metricMetadata']
        if withDataLink:
            metricInfo['Data'] = []
            metricInfo['Data'].append(metric['metricDataFile'])
            metricInfo['Data'].append(os.path.join(self.outDir, metric['metricDataFile']))
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
        Return a recarray of the plot which match a given metric.
        """
        match = np.where(self.plots['metricId'] == metric['metricId'])
        return self.plots[match]

    def plotDict(self, plots):
        """
        Returns an ordered dicts with 'plotType':{dict of 'plotFile': [], 'thumbFile', []}, given recarray of plots.
        """
        plotDict = OrderedDict()
        plotTypes = list(plots['plotType'])
        # Go through plots in 'plotOrder'.
        for p in self.plotOrder:
            if p in plotTypes:
                plotDict[p] = {}
                plotDict[p]['plotFile'] = []
                plotDict[p]['thumbFile'] = []
                plotmatch = plots[np.where(plots['plotType'] == p)]
                for pl in plotmatch:
                    plotDict[p]['plotFile'].append(self.getPlotfile(pl))
                    plotDict[p]['thumbFile'].append(self.getThumbfile(pl))
                plotTypes.remove(p)
        # Round up remaining plots.
        for p in plotTypes:
            plotDict[p] = {}
            plotDict[p]['plotFile'] = []
            plotDict[p]['thumbFile'] = []
            plotmatch = plots[np.where(plots['plotType'] == p)]
            for pl in plotmatch:
                plotDict[p]['plotFile'].append(self.getPlotfile(pl))
                plotDict[p]['thumbFile'].append(self.getThumbfile(pl))
        return plotDict

    def getThumbfile(self, plot):
        """
        Return the thumbnail file name for a given plot.
        """
        thumbname =  'thumb.' + ''.join(plot['plotFile'].split('.')[:-1]) + '.png'
        thumbfile = os.path.join(self.outDir, thumbname)
        return thumbfile

    def getPlotfile(self, plot):
        """
        Return the filename for a given plot.
        """
        return os.path.join(self.outDir, plot['plotFile'])

    def orderPlots(self, skyPlots):
        """
        If the plots are of single filters, add gaps so that they will be layed out
        in a 3x2 grid on the Multi Color page.  If there are other plots that are not of
        a single filter, they will be appended to the end.

        If the plots include multiple plots in the same single filter no gaps are added.
        """
        orderList = ['u','g','r','i','z','y']
        orderedSkymatchPlots = []

        # Make a copy of the original, which should already be in order
        skyPlotsOrig = list(skyPlots)

        if len(skyPlots) > 0:
            blankRecord = skyPlots[0].copy()
            blankRecord['plotId'] = -1
            blankRecord['metricId'] = -1
            blankRecord['plotFile'] = None

        for f in orderList:
            found = False
            for i, rec in enumerate(skyPlots):
                plot = rec['plotFile']
                if '_'+f+'_' in plot:
                    orderedSkymatchPlots.append(rec)
                    skyPlots.remove(rec)
                    found = True
            # If there isn't a filter, just put in a blank dummy placeholder
            if not found:
                orderedSkymatchPlots.append(blankRecord)

        # If there are multiple plots for a filter, revert to the original
        filtHist = np.zeros(len(orderList))
        for plot in orderedSkymatchPlots:
            for i,filt in enumerate(orderList):
                if '_'+filt+'_' in plot['plotFile']:
                    filtHist[i] += 1
        if np.max(filtHist) > 1:
            orderedSkymatchPlots = skyPlotsOrig
        else:
            # Tack on any left over plots (e.g., joint completeness)
            for plot in skyPlots:
                orderedSkymatchPlots.append(plot)

        # Pad out to make sure there are rows of 3
        while len(orderedSkymatchPlots) % 3 != 0:
            orderedSkymatchPlots.append(blankRecord)

        return orderedSkymatchPlots

    def getSkyMaps(self, metrics=None):
        """
        Return a list of the skymaps, optionally for subset of metrics.
        """
        orderList = ['u','g','r','i','z','y']
        if metrics is None:
            metrics = self.metrics
        skymatchPlots = []
        for m in metrics:
            match = np.where(self.plots['metricId'] == m['metricId'])
            matchPlots = self.plots[match]
            if len(matchPlots) > 0 :
                match = np.where(matchPlots['plotType'] == 'SkyMap')
                for skymatch in matchPlots[match]:
                    skymatchPlots.append(skymatch)

        return skymatchPlots

    ## Set of methods to deal with summary stats.

    def statsForMetric(self, metric, statName=None):
        """
        Return the summary statistics which match a given metric.

        Optionally specify a particular statName that you want to match.
        """
        match = np.where(self.stats['metricId'] == metric['metricId'])
        stats = self.stats[match]
        if statName is not None:
            match = np.where(stats['summaryName'] == statName)
            stats = stats[match]
        return stats

    def statDict(self, stats):
        """
        Returns an ordered dictionary with statName:statValue for numpy recarray of stats.

        Note that if you pass 'stats' from multiple metrics with the same summary names, they
         will be overwritten in the resulting dictionary! So just use stats from one metric.
        """
        # Result = dict with key == summary stat name, value = summary stat value.
        sdict = OrderedDict()
        statnames = self.orderStatNames(stats)
        for n in statnames:
            match = np.where(stats['summaryName'] == n)
            sdict[stats['summaryName'][match][0]] = stats['summaryValue'][match][0]
        return sdict

    def orderStatNames(self, stats):
        """
        For a recarray of stats, return a list containing all the unique 'summaryNames'
        in a default ordering (identity-count-mean-median-rms..).
        """
        names = set()
        for stat in stats:
            names.add(stat['summaryName'])
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
        For a recarray of metrics, return a list containing all the unique 'summaryNames'
        in a default ordering.
        """
        names = set()
        for metric in metrics:
            stats = self.statsForMetric(metric)
            for stat in stats:
                names.add(stat['summaryName'])
        # Add some default sorting.
        namelist = []
        for nord in self.summaryStatOrder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return namelist
