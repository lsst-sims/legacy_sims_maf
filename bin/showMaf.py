#! /usr/bin/env python
import numpy as np
from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
import os, argparse

from lsst.sims.maf.viz import MafRunResults, MafTracking
import lsst.sims.maf.db as db

class RunSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("runselect.html")
        if 'runId' in self.request.query_arguments:
            runId = int(self.request.query_arguments['runId'][0])
        else:
            # Set runID to a negative number, to default to first run.
            runId = startRunId
        self.write(selectTempl.render(runlist=runlist, runId=runId))

class MetricSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("metricselect.html")
        runId = int(self.request.query_arguments['runId'][0])
        self.write(selectTempl.render(runlist=runlist, runId=runId))

class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("grid.html")
        runId = int(self.request.query_arguments['runId'][0])
        if 'metricId' in self.request.query_arguments:
            metricIdList = self.request.query_arguments['metricId']
        else:
            metricIdList = []
        if 'Group_subgroup' in self.request.query_arguments:
            groupList = self.request.query_arguments['Group_subgroup']
        else:
            groupList = []
        self.write(gridTempl.render(metricIdList=metricIdList, groupList=groupList,
                                    runId=runId, runlist=runlist))

class ConfigPageHandler(web.RequestHandler):
    def get(self):
        configTempl = env.get_template("configs.html")
        runId = int(self.request.query_arguments['runId'][0])
        self.write(configTempl.render(runlist=runlist, runId=runId))

class StatPageHandler(web.RequestHandler):
    def get(self):
        statTempl = env.get_template("stats.html")
        runId = int(self.request.query_arguments['runId'][0])
        self.write(statTempl.render(runlist=runlist, runId=runId))

class AllMetricResultsPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        allresultsTempl = env.get_template("allmetricresults.html")
        runId = int(self.request.query_arguments['runId'][0])    
        self.write(allresultsTempl.render(runlist=runlist, runId=runId))
        
def make_app():
    """The tornado global configuration """
    application = web.Application([
            ("/", RunSelectHandler),
            ("/metricSelect", MetricSelectHandler),
            ("/metricResults", MetricGridPageHandler),
            ("/configParams", ConfigPageHandler),
            ("/summaryStats", StatPageHandler), 
            ("/allMetricResults", AllMetricResultsPageHandler),
            (r"/(favicon.ico)", web.StaticFileHandler, {'path':faviconPath}),
            (r"/*/(.*)", web.StaticFileHandler, {'path':staticpath}), 
            ])
    return application

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python script to display MAF output in a web browser."+
                                     "  After launching, point your browser to 'http://localhost:8888/'")
    defaultdb = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')
    defaultdb = 'sqlite:///' + defaultdb
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb, help="Tracking database dbAddress.")
    parser.add_argument("-d", "--mafDir", type=str, default=None, help="Add this directory to the trackingDb and open immediately.")
    parser.add_argument("-c", "--mafComment", type=str, default=None, help="Add a comment to the trackingDB describing the MAF analysis of this directory (paired with mafDir argument).")
    args = parser.parse_args()

    # Check tracking DB is sqlite (and add as convenience if forgotten).
    trackingDbAddress = args.trackingDb
    if not trackingDbAddress.startswith('sqlite:///'):
        trackingDbAddress = 'sqlite:///' + trackingDbAddress
    print 'Using tracking database at %s' %(trackingDbAddress)
    
    global startRunId
    startRunId = -666
    # If given a directory argument:
    if args.mafDir is not None:
        mafDir = os.path.realpath(args.mafDir)
        if not os.path.isdir(mafDir):
            print 'There is no directory containing MAF outputs at %s.' %(mafDir)
            print 'Just opening using tracking db at %s.' %(trackingDbAddress)
        # Open tracking database to add a run.
        trackingDb = db.TrackingDb(trackingDbAddress=trackingDbAddress)    
        # Set opsim comment and name from the config files from the run.
        opsimComment = ''
        opsimRun = ''
        if os.path.isfile(os.path.join(mafDir, 'configSummary.txt')):
            file = open(os.path.join(mafDir, 'configSummary.txt'))
            for line in file:
                tmp = line.split()
                if tmp[0].startswith('RunName'):
                    opsimRun = ' '.join(tmp[1:])
                if tmp[0].startswith('RunComment'):
                    opsimComment = ' '.join(tmp[1:])
        # Give some feedback to the user about what we're doing.
        print 'Adding to tracking database at %s:' %(trackingDbAddress)
        print ' MafDir = %s' %(mafDir)
        print ' MafComment = %s' %(args.mafComment)
        print ' OpsimRun = %s' %(opsimRun)
        print ' OpsimComment = %s' %(opsimComment)
        # Add the run.
        startRunId = trackingDb.addRun(opsimRun, opsimComment, args.mafComment, mafDir)
        print ' Used runID %d' %(startRunId)
        trackingDb.close()
        
    # Open tracking database and start visualization.
    global runlist
    runlist = MafTracking(trackingDbAddress)
    if startRunId < 0:
        startRunId = runlist.runs[0]['mafRunId']
    # Set up path to template and favicon paths, and load templates.
    mafDir = os.getenv('SIMS_MAF_DIR')
    templateDir = os.path.join(mafDir, 'python/lsst/sims/maf/viz/templates/' )
    global faviconPath
    faviconPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    env = Environment(loader=FileSystemLoader(templateDir))

    global staticpath
    staticpath = '.'
    
    # Start up tornado app.
    application = make_app()
    application.listen(8888)
    print 'Tornado Starting: \nPoint your web browser to http://localhost:8888/ \nCtrl-C to stop'

    ioloop.IOLoop.instance().start()
    
