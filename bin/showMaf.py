#! /usr/bin/env python
import numpy as np
from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
import os, argparse

from lsst.sims.maf.viz import MafRunResults, MafTracking

class RunSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("runselect.html")
        self.write(selectTempl.render(runlist=runlist))

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
            (r"/*/(.*)", web.StaticFileHandler, {'path':'.'}), 
            ])
    return application

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python script to display MAF output in a web browser."+
                                     "  After launching, point your browser to 'http://localhost:8888/'")
    defaultdb = os.path.join(os.getenv('SIMS_MAF_DIR'), 'bin', 'trackingDb_sqlite.db')
    defaultdb = 'sqlite:///' + defaultdb
    parser.add_argument("--trackingDb", type=str, default=defaultdb, help="Tracking database dbAddress.")
    args = parser.parse_args()

    runlist = MafTracking(args.trackingDb)

    # Set up path to template and favicon paths, and load templates.
    mafDir = os.getenv('SIMS_MAF_DIR')
    templateDir = os.path.join(mafDir, 'python/lsst/sims/maf/viz/templates/' )
    faviconPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    env = Environment(loader=FileSystemLoader(templateDir))

    # Start up tornado app.
    application = make_app()
    application.listen(8888)
    print 'Tornado Starting: \nPoint your web browser to http://localhost:8888/ \nCtrl-C to stop'

    ioloop.IOLoop.instance().start()
    
