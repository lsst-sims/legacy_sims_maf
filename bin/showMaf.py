#! /usr/bin/env python
import lsst.sims.maf.db as db
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
import os, argparse

class layoutResults(object):
    """Class to read MAF's resultsDb_sqlite.db and organize the output for display on web pages """
    def __init__(self, outDir):
        """Read in the results database """
        database = db.Database('sqlite:///'+outDir+'/resultsDb_sqlite.db',
                               dbTables={'metrics':['metrics','metricID'] ,
                                         'plots':['plots','plotId'],
                                         'stats':['summarystats','statId']})
        # Just pull all three tables
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

    def SStar(self):
        """Sort results from database for using with ssTemplate.html """
        # Set up lists that will be looped over by template
        blocks =[]
        completenessBlocks = []
        identStats = []
        basicStats = []
        completeStats = []
        etcStats = []

        # Stack on an extra column for the column name so we can sort on it
        # and get "Median Airmass" next to "Rms Airmass", etc.
        metrics = rfn.merge_arrays([self.metrics,
                                    np.empty(self.metrics.size,
                                                      dtype=[('colName','|S256'), ('filt', '|S256')])],
                                   flatten=True, usemask=False)
        # Stack on a column so filters sort in order of "ugrizy"
        for i in np.arange(metrics.size):
            filt = metrics['metricMetadata'][i].replace(' u ', ' a ')
            filt = filt.replace(' g ', ' b ')
            filt = filt.replace(' r ', ' c ')
            filt = filt.replace(' i ', ' d ')
            filt = filt.replace(' z ', ' e ')
            filt = filt.replace(' y ', ' f ')

            metrics['filt'][i] = filt
            name = metrics['metricName'][i].split(' ')
            if len(name) > 1:
                metrics['colName'][i] = name[1]
            else:
                metrics['colName'][i] = metrics['metricName'][i]

            if '_u' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('u','a')
            if '_g' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('g','b')
            if '_r' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('r','c')
            if '_i' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('i','d')
            if '_z' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('z','e')
            if '_y' in metrics['colName'][i] : metrics['colName'][i] = metrics['colName'][i].replace('y','f')

        metrics.sort(order=['colName', 'slicerName', 'filt', 'sqlConstraint'])

        for metric in metrics:
            mId = metric['metricId']
            relevant_plots = self.plots[np.where(self.plots['metricId'] == mId)[0]]
            for i in np.arange(relevant_plots.size):
                relevant_plots['plotFile'][i] = relevant_plots['plotFile'][i].replace('.pdf', '.png')
            relevant_stats = self.stats[np.where(self.stats['metricId'] == mId)[0] ]

            relevant_metrics = metrics[np.where(metrics['metricId'] == mId)[0] ]
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
                # XXX -- need to tighten up this constraint, decide on formatting.
                elif ('Mean' in statsDict.keys()) & ('Rms' in statsDict.keys()):
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


class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("main.html")
        qargs = self.request.query_arguments
        self.write(gridTempl.render(metrics=qargs))

class SelectPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        mainTempl = env.get_template("ssTemplate.html")
        self.write(mainTempl.render(outDir=outDir, **layout.SStar()))

        
def make_app():
    """The tornado global configuration """
    application = web.Application([
            ("/metricResult", MetricGridPageHandler),
            ("/", SelectPageHandler),
            (r"/"+outDir+"/(.*)", web.StaticFileHandler, {'path':outDir}), 
            (r"/(favicon.ico)", web.StaticFileHandler, {'path':faviconPath}),
            ])
    return application

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python script to display MAF output in a web browser."+
                                     "  After launching, point your browser to 'http://localhost:8888/'")
    parser.add_argument("outDir", help="Directory holding the MAF output to be displayed")
    args = parser.parse_args()
    outDir = args.outDir

    layout = layoutResults(outDir)
    
    mafDir = os.getenv('SIMS_MAF_DIR')
    templateDir = os.path.join(mafDir, 'python/lsst/sims/maf/viz/templates/' )
    faviconPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    env = Environment(loader=FileSystemLoader(templateDir))
    
    application = make_app()
    application.listen(8888)
    print 'Tornado Starting: \nPoint your web browser to http://localhost:8888/ \nCtrl-C to stop'

    ioloop.IOLoop.instance().start()
    
