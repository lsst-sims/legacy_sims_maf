import os
from collections import OrderedDict
import numpy as np
from numpy.lib.recfunctions import rec_join
import lsst.sims.maf.db as db

class layoutResults(object):
    """Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages """
    def __init__(self, outDir):
        """
        Instantiate the (individual run) layout visualization class.

        This class provides methods used by our jinja2 templates to help interact
        with the outputs of MAF. 
        """
        if outDir == '.':
            raise Exception('Cannot run showMaf.py from within the output directory. '
                            'Go up a level and try again.')

        self.outDir = outDir

        self.configSummary = os.path.join(self.outDir, 'configSummary.txt')
        if not os.path.isfile(self.configSummary):
            self.configSummary = 'Config Summary Not Available'
            self.runName = 'RunName Not Available'
        else:
            with open (self.configSummary, "r") as myfile:
                config=myfile.read().replace('\n', '')
            spot = config.find('RunName')
            self.runName = config[spot:spot+300].split(' ')[1]

        self.configDetails = os.path.join(self.outDir, 'configDetails.txt')
        if not os.path.isfile(self.configDetails):
            self.configDetails = 'Config Details Not Available.'


        # Read in the results database.
        database = db.Database('sqlite:///'+outDir+'/resultsDb_sqlite.db',
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
        del self.displays
        # Get plot and summary stat info.
        self.plots = database.queryDatabase('plots', 'select * from plots')
        self.stats = database.queryDatabase('stats', 'select * from summarystats')

        # Make empty arrays if there was nothing in the database
        if len(self.plots) == 0:
            self.plots = np.zeros(0, dtype=[('metricId',int),('plotFile', '|S10')])
        if len(self.stats) == 0:
            self.stats = np.zeros(0, dtype=[('metricId',int), ('summaryName', '|S10'),('summaryValue', float)])

        # Pull up the names of the groups and subgroups. 
        groups = sorted(list(np.unique(self.metrics['displayGroup'])))
        self.groups = OrderedDict()
        for g in groups:
            self.groups[g] = set()
        for metric in self.metrics:
            self.groups[metric['displayGroup']].add(metric['displaySubgroup'])
        for g in self.groups:
            self.groups[g] = sorted(list(self.groups[g]))

        self.summaryStatOrder = ['Identity', 'Count', 'Mean', 'Median', 'Rms', 'RobustRms', 
                                 'm3Sigma', 'p3Sigma']
        self.plotOrder = ['SkyMap', 'Histogram', 'PowerSpectrum']


    def _convertMetricId(self, metricId):
        if not isinstance(metricId, int):
            if isinstance(metricId, list):
                metricId = metricId[0]
            metricId = int(metricId)
        return metricId
                                                   

    def allMetricIds(self):
        """
        Return a dict of key=metricId / value = metricId, for all metricIds.
        """
        metricDict = {}
        for m in self.metrics['metricId']:
            key = '%s' %(m)
            metricDict[key] = [key]
        return metricDict
        
    def orderMetricIds(self, metricIds):
        """
        Given a list of metric Ids, return them in group/subgroup/order/metricName order. 
        """
        metricIdList = []
        for mId in metricIds:
            mId = self._convertMetricId(mId)
            metricIdList.append(mId)
        metricIds = metricIdList
        orderMIds = []
        groupDict = self.groupWithMetricIds(metricIds)
        for g in groupDict:
            for sg in groupDict[g]:
                metrics = self.metricsInSubgroup(g, sg)
                metrics = self.sortMetrics(metrics)
                for mId in metrics['metricId']:
                    if mId in metricIds:
                        orderMIds.append(mId)            
        return orderMIds

    def groupWithMetricIds(self, metricIds):
        """
        Given a list of metric Ids, generate the ordered dict of relevant group/subgroups. 
        """
        groups = []
        subgroups = []
        for mId in metricIds:
            mId = self._convertMetricId(mId)
            match = (self.metrics['metricId'] == mId)
            groups.append(self.metrics['displayGroup'][match])
            subgroups.append(self.metrics['displaySubgroup'][match])
        groupDict = OrderedDict()
        for g in self.groups:
            if g in groups:
                groupDict[g] = []
                for sg in self.groups[g]:
                    if sg in subgroups:
                        groupDict[g].append(sg)
        return groupDict
                

    def metricIdsInSubgroup(self, metricIds, group, subgroup):
        """
        Return a list of the subset of metricIds which are within a group/subgroup.
        """
        metricIdList = []
        for mId in metricIds:
            mId = self._convertMetricId(mId)
            metricIdList.append(mId)
        metricIds = metricIdList
        # Find all metrics in this group/subgroup.
        metrics = self.metricsInSubgroup(group, subgroup)
        orderMIds = []
        for m in metrics:
            if m['metricId'] in metricIds:
                orderMIds.append(m['metricId'])
        return orderMIds
        
    def sortMetrics(self, metrics, order=['displayGroup','displaySubgroup','displayOrder', 
                                          'slicerName', 'metricName', 'metricMetadata']):
        """
        Sort the metrics by group, subgroup, order, and then finally 'metricName'. 
        """
        return np.sort(metrics, order=order)    

    def metricsInGroup(self, group):
        """
        Given a group, return the metrics belonging to this group.
        """
        match = (self.metrics['displayGroup'] == group)
        return self.metrics[match]
        
    def metricsInSubgroup(self, group, subgroup):
        """
        Given a group and subgroup, return the metrics belonging to these group/subgroups, in display order.
        """
        metrics = self.metricsInGroup(group)
        match = (metrics['displaySubgroup'] == subgroup)
        return self.sortMetrics(metrics[match])

    def metricWithMetricId(self, metricId):
        """
        Given a single metric ID, return the metric which matches.
        """
        metricId = self._convertMetricId(metricId)
        match = (self.metrics['metricId'] == metricId)
        return self.sortMetrics(self.metrics[match])

    def metricInfo(self, metric):
        """
        Return a dict with the metric info we want to show on the webpages.
        """
        # Provides a way to easily modify what we show on all webpages without
        #  significantly altering the templates. Or provides an abstraction layer 
        #  in case the resultsDB column names change.
        metricInfo = OrderedDict()
        metricInfo['MetricName'] = metric['metricName']
        metricInfo['Slicer'] = metric['slicerName']
        metricInfo['Metadata'] = metric['metricMetadata']
        return metricInfo

    def captionForMetric(self, metric):
        """
        Return the caption for a given metric.
        """
        return metric['displayCaption'][0]

    def plotsForMetric(self, metric):
        """
        Return an ordered dict with the plots matching a given metric.
        """
        plotDict = OrderedDict()
        match = (self.plots['metricId'] == metric['metricId'])
        matchPlots = self.plots[match]
        plotTypes = list(matchPlots['plotType'])
        for o in self.plotOrder:
            if o in plotTypes:
                plotDict[o] = matchPlots['plotFile'][np.where(matchPlots['plotType'] == o)][0]
                plotTypes.remove(o)
        for p in plotTypes:
            plotDict[p] = matchPlots['plotFile'][np.where(matchPlots['plotType'] == p)][0]
        return plotDict

    def getThumbfile(self, plotfile):
        """
        Convert a plot filename into the expected thumbnail file name.
        """
        path, file = os.path.split(plotfile)        
        if path == '':
            path = self.outDir
        thumbname =  'thumb.' + ''.join(file.split('.')[:-1]) + '.png'
        thumbfile = os.path.join(path, thumbname)
        return thumbfile

    def getPlotfile(self, plot):
        """
        Return the filename of a particular plot.
        """
        if isinstance(plot, str):
            return os.path.join(self.outDir, plot)
        else:
            return os.path.join(self.outDir, plot['plotFile'])

    def statsForMetric(self, metric):
        """
        Return the summary statistics which match a given metric.
        """
        match = (self.stats['metricId'] == metric['metricId'])        
        return self.stats[match]
    
    def statDict(self, stats):
        """
        Utility to turn a single (or multiple) stat lines from numpy structured array into dict. 

        Note that if you pass 'stats' from multiple metrics with the same summary names, they
         will be overwritten in the resulting dictionary! So just use stats from one metric.
        """
        # Provides way that templates do not have to know exact field names in resultsDB,
        #  plus allows packaging multiple stats into a single dictionary. 
        # Result = dict with key == summary stat name, value = summary stat value. 
        sdict = OrderedDict()
        statnames = self.orderStatNames(stats)
        for n in statnames:
            match = (stats['summaryName'] == n)
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

    def packageMonster(self):
        #XXX--plan on breaking this into several methods for displaying different info on different pages.
        # Maybe take "groups" as input, then only display the selected groups
        """Sort results from database for using with monster.html """
        # Set up lists that will be looped over by template
        blocks =[]
        completenessBlocks = []
        identStats = []
        basicStats = []
        completeStats = []
        etcStats = []
        # Apply the default sorting
        metrics = self.sortMetrics(self.metrics)

        # List what should go in the "basic summary stat" table
        basicStatNames = sorted(['Mean', 'Rms', 'Median', 'p3Sigma', 'm3Sigma', 'Count'])
                                
        for metric in metrics:
            mId = metric['metricId']
            relevant_plots = self.plots[np.where(self.plots['metricId'] == mId)[0]]
            thumb_plots = relevant_plots.copy()
            for i in np.arange(relevant_plots.size):
                thumb_plots['plotFile'][i] = 'thumb.'+relevant_plots['plotFile'][i].replace('.pdf', '.png')
            relevant_stats = self.stats[np.where(self.stats['metricId'] == mId)[0] ]
            relevant_metrics = self.metrics[np.where(self.metrics['metricId'] == mId)[0] ]
        
            stat_list = [(i, '%.4g'%j) for i,j in  zip(relevant_stats['summaryName'],
                                                       relevant_stats['summaryValue']) ]
            statsDict=OrderedDict()
            name = relevant_metrics['metricName'][0]+', '+ \
                                 relevant_metrics['slicerName'][0] \
                                 + ', ' +  relevant_metrics['metricMetadata'][0]
            
            for rel_stat in relevant_stats:
                statsDict[rel_stat['summaryName'].replace('TableFraction', '')] = '%.4g'%rel_stat['summaryValue']

            # Break it down into 4 different summary stat tables,
            # 1) Completeness tables
            # 2) Identity (i.e., unislicer) table
            # 3) "basic" table (mean, RMS, median, p/m 3 sigma...)
            # 4) the etc table for anything left over.

            
            if len(statsDict) != 0 :
                if 'Completeness' in name:
                    completeStats.append({'NameInfo':name, 'stats':statsDict} )
                elif ('Identity' in statsDict.keys()) & (len(statsDict.keys()) == 1):
                    identStats.append({'NameInfo':name, 'stats':statsDict})
                elif sorted(statsDict.keys()) == basicStatNames:
                    basicStats.append({'NameInfo':name, 'stats':statsDict} )
                else:
                    etcStats.append({'NameInfo':name, 'stats':statsDict} )
            block = {'NameInfo': relevant_metrics['metricName'][0]+', '+
                     relevant_metrics['slicerName'][0]
                     + ', ' +  relevant_metrics['metricMetadata'][0],
                     'plots':zip(relevant_plots['plotFile'].tolist(), thumb_plots['plotFile'].tolist()),
                     'stats':stat_list}
            # If it's a completeness metric, pull it out
            if metric['metricName'][0:12] == 'Completeness':
                completenessBlocks.append(block)
            else:
                blocks.append(block)

        return {'blocks':blocks, 'completenessBlocks':completenessBlocks,
                'identStats':identStats, 'basicStats':basicStats,
                'completeStats':completeStats, 'etcStats':etcStats, 'runName':self.runName}

