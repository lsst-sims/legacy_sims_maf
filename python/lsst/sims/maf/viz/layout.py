import lsst.sims.maf.db as db
import numpy as np
from collections import OrderedDict
import os


class layoutResults(object):
    """Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages """
    def __init__(self, outDir):
        """Read in the results database. """
        database = db.Database('sqlite:///'+outDir+'/resultsDb_sqlite.db',
                               dbTables={'metrics':['metrics','metricID'] ,
                                         'plots':['plots','plotId'],
                                         'stats':['summarystats','statId']})
        # Just pull all three tables.
        # self.metrics == numpy structured array with 'metricID', '... ' (other parameters from
        self.metrics = database.queryDatabase('metrics', 'select * from metrics')
        self.plots = database.queryDatabase('plots', 'select * from plots')
        self.stats = database.queryDatabase('stats', 'select * from summarystats')

        # Grab the runName as well
        configFile = os.path.join(outDir, 'configSummary.txt' )
        if os.path.isfile(configFile):
            with open (configFile, "r") as myfile:
                config=myfile.read().replace('\n', '')
            spot = config.find('RunName')
            self.runName = config[spot:spot+300].split(' ')[1]
        else:
            self.runName = 'No configSummary.txt'

        
    def _sortMetrics(self, metrics, order=['displayGroup','displaySubgroup','displayOrder', 'metricName']):
        # Sort the metrics (numpy structured array sorting, ftw). 
        return np.sort(metrics, order=order)
        
    def _matchPlots(self, metric):
        # Find the plots which match a given metric.
        return self.plots[np.where(self.plots['metricId'] == metric['metricId'])[0]]
                          
    def _matchStats(self, metric):
        # Find the summary statistics which match a given metric.
        return self.stats[np.where(self.stats['metricId'] == metric['metricId'])[0]]
    
    
                          
                          
        
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
            for i in np.arange(relevant_plots.size):
                relevant_plots['plotFile'][i] = relevant_plots['plotFile'][i].replace('.pdf', '.png')
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
                           'plots':relevant_plots['plotFile'].tolist(),
                           'stats':stat_list}
            # If it's a completeness metric, pull it out
            if metric['metricName'][0:12] == 'Completeness':
                completenessBlocks.append(block)
            else:
                blocks.append(block)

        return {'blocks':blocks, 'completenessBlocks':completenessBlocks,
                'identStats':identStats, 'basicStats':basicStats,
                'completeStats':completeStats, 'etcStats':etcStats, 'runName':self.runName}

