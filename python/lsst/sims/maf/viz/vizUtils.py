import lsst.sims.maf.db as db
import numpy as np

def loadResults(sourceDir):
    """Load up the three tables from resultsDb_sqlite.db """
    database = db.Database('sqlite:///'+sourceDir+'/resultsDb_sqlite.db',
                           dbTables={'metrics':['metrics','metricID'] ,
                                     'plots':['plots','plotId'],
                                     'stats':['summarystats','statId']})
    # Hmm, seems like there should be a better way to do this--maybe an outer join or something?
    metrics = database.queryDatabase('metrics', 'select * from metrics')
    plots = database.queryDatabase('plots', 'select * from plots')
    stats = database.queryDatabase('stats', 'select * from summarystats')
    return metrics, plots, stats


def blockAll(metrics, plots, stats):
    """Package up all the MAF results to be displayed"""
    blocks = []
    for mId in metrics['metricId']:
        relevant_plots = plots[np.where(plots['metricId'] == mId)[0]]
        for i in np.arange(relevant_plots.size):
            relevant_plots['plotFile'][i] = relevant_plots['plotFile'][i].replace('.pdf', '.png')
        relevant_stats = stats[np.where(stats['metricId'] == mId)[0] ]
        relevant_metrics = metrics[np.where(metrics['metricId'] == mId)[0] ]
        stat_list = [(i, '%.4g'%j) for i,j in  zip(relevant_stats['summaryName'],
                                                   relevant_stats['summaryValue']) ]  
        blocks.append({'NameInfo': relevant_metrics['metricName'][0]+', '+
                       relevant_metrics['slicerName'][0]
                       + ', ' +  relevant_metrics['sqlConstraint'][0],
                       'plots':relevant_plots['plotFile'].tolist(),
                       'stats':stat_list})

    return blocks


def blockSS(metrics, plots, stats):
    """Group up results to be layed out in SSTAR-like way """
    blocks =[]

    completenessBlocks = []
    allStats = []

    metrics.sort(order=['metricName', 'slicerName', 'sqlConstraint'])


    for metric in metrics:
        mId = metric['metricId']
        relevant_plots = plots[np.where(plots['metricId'] == mId)[0]]
        for i in np.arange(relevant_plots.size):
            relevant_plots['plotFile'][i] = relevant_plots['plotFile'][i].replace('.pdf', '.png')

        relevant_stats = stats[np.where(stats['metricId'] == mId)[0] ]
        relevant_metrics = metrics[np.where(metrics['metricId'] == mId)[0] ]
        stat_list = [(i, '%.4g'%j) for i,j in  zip(relevant_stats['summaryName'],
                                                   relevant_stats['summaryValue']) ]
        block = {'NameInfo': relevant_metrics['metricName'][0]+', '+
                       relevant_metrics['slicerName'][0]
                       + ', ' +  relevant_metrics['sqlConstraint'][0],
                       'plots':relevant_plots['plotFile'].tolist(),
                       'stats':stat_list}
        if metric['metricName'][0:12] == 'Completeness':
            completenessBlocks.append(block)
            
        else:
            blocks.append(block)
    
    return {'blocks':blocks, 'completenessBlocks':completenessBlocks, 'allStats':allStats}

    
    #how do I sort things by ugrizy?

    # So let's say I grab all the seeing metrics--
    #
    #normalBlock = {'skymap_u':somefile,  'skymap_u':somefile, 'skymap_u':somefile, ...
    #               'hist_u': , 'hist_g':
    #}
    

