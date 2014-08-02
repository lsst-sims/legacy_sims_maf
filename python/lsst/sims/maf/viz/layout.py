import lsst.sims.maf.db as db
import numpy as np
from collections import OrderedDict
import os


class layoutResults(object):
    """Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages """
    def __init__(self, outDir):
        """
        Instantiate the (individual run) layout visualization class.

        This class provides methods used by our jinja2 templates to help interact
        with the outputs of MAF. 
        """
        self.outDir = outDir
        # Read in the results database.
        database = db.Database('sqlite:///'+outDir+'/resultsDb_sqlite.db',
                               dbTables={'metrics':['metrics','metricID'] ,
                                         'plots':['plots','plotId'],
                                         'stats':['summarystats','statId']})
        # Just pull all three tables.
        # self.metrics == numpy structured array with : 
        #   metricId|metricName|slicerName|simDataName|sqlConstraint|metricMetadata|
        #   metricDataFile|displayGroup|displaySubgroup|displayOrder|displayCaption 
        # We provide methods below to return filenames, metric info, etc. as dictionaries --
        #   rather than having to know the exact names of the fields from the DB in each template.
        #  The idea being that this should make the template code & presentation layer more
        #    easily maintainable in the future.
        self.metrics = database.queryDatabase('metrics', 'select * from metrics')        
        self.plots = database.queryDatabase('plots', 'select * from plots')
        self.stats = database.queryDatabase('stats', 'select * from summarystats')

        # Grab the runName for the page headers. 

        # Make empty arrays if there was nothing in the database
        if len(self.plots) == 0:
            self.plots = np.zeros(0, dtype=[('metricId',int),('plotFile', '|S10')])
        if len(self.stats) == 0:
            self.stats = np.zeros(0, dtype=[('metricId',int), ('summaryName', '|S10'),('summaryValue', float)])

        configFile = os.path.join(outDir, 'configSummary.txt' )
        if os.path.isfile(configFile):
            with open (configFile, "r") as myfile:
                config=myfile.read().replace('\n', '')
            spot = config.find('RunName')
            self.runName = config[spot:spot+300].split(' ')[1]
        else:
            self.runName = 'Not available'

        # Pull up the names of the groups and subgroups. 
        self.groups = {}
        for metric in self.metrics:
            group = metric['displayGroup']
            subgroup = metric['displaySubgroup']
            # Check if group already a key in self.groups dictionary.
            if group not in self.groups:
                self.groups[group] = set()
            self.groups[group].add(metric['displaySubgroup'])
        for g in self.groups:
            self.groups[g] = list(self.groups[g])
                                                               

    def orderMetricIds(self, metricIds):
        """
        Given a list of metric Ids, return them in group/subgroup/order/metricName order. 
        """
        #  TOOODOOOOO
        return metricIds
        
    def sortMetrics(self, metrics, order=['displayGroup','displaySubgroup','displayOrder', 'metricName']):
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
        if not isinstance(metricId, int):
            if isinstance(metricId, list):
                metricId = metricId[0]
            metricId = int(metricId)
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

    def plotsForMetric(self, metric):
        """
        Return an ordered dict with the plots matching a given metric.
        """
        plotDict = OrderedDict()
        match = (self.plots['metricId'] == metric['metricId'])
        matchPlots = self.plots[match]
        order = ['SkyMap', 'Histogram', 'PowerSpectrum']
        plotTypes = list(matchPlots['plotType'])
        for o in order:
            if o in plotTypes:
                plotDict[o] = matchPlots['plotFile'][np.where(matchPlots['plotType'] == o)][0]
                plotTypes.remove(o)
        for p in plotTypes:
            plotDict[p] = matchPlots['plotFile'][np.where(matchPlots['plotType'] == p)][0]
        return plotDict

    def getThumbname(self, plotfile):
        """
        Convert a plot filename into the expected thumbnail file name.
        """
        thumbname =  'thumb.' + ''.join(plotfile.split('.')[:-1]) + '.png'
        return thumbname

    def getPlotname(self, plot):
        """
        Return the filename of a particular plot.
        """
        return plot['plotFile']

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
        sdict = {}
        for stat in stats:
            sdict[stat['summaryName']] = stat['summaryValue']
        return sdict
        
    def allStatNames(self, stats):
        """
        For a group of stats, return a list containing all the unique 'summaryNames' in that group.

        Add a default ordering to returned list. (identity-mean-median-rms..)
        """
        names = set()
        for stat in stats:
            names.add(stat['summaryName'])
        # Add some default sorting:
        defaultorder = ['Identity', 'Mean', 'Median', 'Rms', 'RobustRms']
        namelist = []
        for nord in defaultorder:
            if nord in names:
                namelist.append(nord)
                names.remove(nord)
        for remaining in names:
            namelist.append(remaining)
        return names


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
        metrics = self._sortMetrics(self.metrics)

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

