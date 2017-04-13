#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
from tornado import ioloop
from tornado import web
from jinja2 import Environment
from jinja2 import FileSystemLoader
import webbrowser

from lsst.sims.maf.viz import MafTracking
import lsst.sims.maf.db as db


class RunSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("runselect.html")
        if 'runId' in self.request.arguments:
            runId = int(self.request.arguments['runId'][0])
        else:
            # Set runID to a negative number, to default to first run.
            runId = startRunId
        self.write(selectTempl.render(runlist=runlist, runId=runId, jsPath=jsPath))


class MetricSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("metricselect.html")
        runId = int(self.request.arguments['runId'][0])
        self.write(selectTempl.render(runlist=runlist, runId=runId))


class MetricResultsPageHandler(web.RequestHandler):
    def get(self):
        resultsTempl = env.get_template("results.html")
        runId = int(self.request.arguments['runId'][0])
        if 'metricId' in self.request.arguments:
            metricIdList = self.request.arguments['metricId']
        else:
            metricIdList = []
        if 'Group_subgroup' in self.request.arguments:
            groupList = self.request.arguments['Group_subgroup']
        else:
            groupList = []
        self.write(resultsTempl.render(metricIdList=metricIdList, groupList=groupList,
                                       runId=runId, runlist=runlist))


class DataHandler(web.RequestHandler):
    def get(self):
        runId = int(self.request.arguments['runId'][0])
        metricId = int(self.request.arguments['metricId'][0])
        if 'datatype' in self.request.arguments:
            datatype = self.request.arguments['datatype'][0].lower()
        else:
            datatype = 'npz'
        run = runlist.getRun(runId)
        metric = run.metricIdsToMetrics([metricId])
        if datatype == 'npz':
            npz = run.getNpz(metric)
            if npz is None:
                self.write('No npz file available.')
            else:
                self.redirect(npz)
        elif datatype == 'json':
            jsn = run.getJson(metric)
            if jsn is None:
                self.write('No JSON file available.')
            else:
                self.write(jsn)
        else:
            self.write('Data type "%s" not understood.' % (datatype))


class ConfigPageHandler(web.RequestHandler):
    def get(self):
        configTempl = env.get_template("configs.html")
        runId = int(self.request.arguments['runId'][0])
        self.write(configTempl.render(runlist=runlist, runId=runId))


class StatPageHandler(web.RequestHandler):
    def get(self):
        statTempl = env.get_template("stats.html")
        runId = int(self.request.arguments['runId'][0])
        self.write(statTempl.render(runlist=runlist, runId=runId))


class AllMetricResultsPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        allresultsTempl = env.get_template("allmetricresults.html")
        runId = int(self.request.arguments['runId'][0])
        self.write(allresultsTempl.render(runlist=runlist, runId=runId))


class MultiColorPageHandler(web.RequestHandler):
    def get(self):
        """Display sky maps. """
        multiColorTempl = env.get_template("multicolor.html")
        runId = int(self.request.arguments['runId'][0])
        self.write(multiColorTempl.render(runlist=runlist, runId=runId))


def make_app():
    """The tornado global configuration """
    application = web.Application([
        ("/", RunSelectHandler),
        ("/metricSelect", MetricSelectHandler),
        ("/metricResults", MetricResultsPageHandler),
        ("/getData", DataHandler),
        ("/configParams", ConfigPageHandler),
        ("/summaryStats", StatPageHandler),
        ("/allMetricResults", AllMetricResultsPageHandler),
        ("/multiColor", MultiColorPageHandler),
        (r"/(favicon.ico)", web.StaticFileHandler, {'path': faviconPath}),
        (r"/(sorttable.js)", web.StaticFileHandler, {'path': jsPath}),
        (r"/*/(.*)", web.StaticFileHandler, {'path': staticpath})])
    return application

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python script to display MAF output in a web browser." +
                                     "  After launching, point your browser to 'http://localhost:8888/'")
    defaultdb = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb,
                        help="Tracking database filename.")
    parser.add_argument("-d", "--mafDir", type=str, default=None,
                        help="Add this directory to the trackingDb and open immediately.")
    parser.add_argument("-c", "--mafComment", type=str, default=None,
                        help="Add a comment to the trackingDB describing the " +
                        " MAF analysis of this directory (paired with mafDir argument).")
    parser.add_argument("-p", "--port", type=int, default=8888, help="Port for connecting to showMaf.")
    parser.add_argument("--noBrowser", dest='noBrowser', default=False,
                        action='store_true', help="Do not open a new browser tab")

    args = parser.parse_args()

    # Check tracking DB is sqlite (and add as convenience if forgotten).
    trackingDb = args.trackingDb
    print('Using tracking database at %s' % (trackingDb))

    global startRunId
    startRunId = -666
    # If given a directory argument:
    if args.mafDir is not None:
        mafDir = os.path.realpath(args.mafDir)
        if not os.path.isdir(mafDir):
            print('There is no directory containing MAF outputs at %s.' % (mafDir))
            print('Just opening using tracking db at %s.' % (trackingDb))
        # Open tracking database to add a run.
        trackDb = db.TrackingDb(database=trackingDb)
        # Set opsim comment and name from the config files from the run.
        opsimComment = ''
        opsimRun = 'NULL'
        opsimDate = ''
        mafDate = ''
        if os.path.isfile(os.path.join(mafDir, 'configSummary.txt')):
            file = open(os.path.join(mafDir, 'configSummary.txt'))
            for line in file:
                tmp = line.split()
                if tmp[0].startswith('RunName'):
                    opsimRun = ' '.join(tmp[1:])
                if tmp[0].startswith('RunComment'):
                    opsimComment = ' '.join(tmp[1:])
                if tmp[0].startswith('MAFVersion'):
                    mafDate = tmp[-1]
                if tmp[0].startswith('OpsimVersion'):
                    opsimDate = tmp[-2]
                    # Let's go ahead and make the formats match
                    opsimDate = opsimDate.split('-')
                    try:
                        opsimDate = opsimDate[1] + '/' + opsimDate[2] + '/' + opsimDate[0][2:]
                    except:
                        opsimDate = 'XXX'
        # Give some feedback to the user about what we're doing.
        print('Adding to tracking database at %s:' % (trackingDb))
        print(' MafDir = %s' % (mafDir))
        print(' MafComment = %s' % (args.mafComment))
        print(' OpsimRun = %s' % (opsimRun))
        print(' OpsimComment = %s' % (opsimComment))
        print(' OpsimDate = %s' % (opsimDate))
        print(' MafDate = %s' % (mafDate))
        # Add the run.
        startRunId = trackDb.addRun(opsimRun, opsimComment, args.mafComment, mafDir,
                                    opsimDate, mafDate, trackingDb)
        print(' Used runID %d' % (startRunId))
        trackDb.close()

    # Open tracking database and start visualization.
    global runlist
    runlist = MafTracking(trackingDb)
    if startRunId < 0:
        startRunId = runlist.runs[0]['mafRunId']
    # Set up path to template and favicon paths, and load templates.
    mafDir = os.getenv('SIMS_MAF_DIR')
    templateDir = os.path.join(mafDir, 'python/lsst/sims/maf/viz/templates/')
    global faviconPath
    faviconPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    global jsPath
    jsPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    env = Environment(loader=FileSystemLoader(templateDir))
    # Add 'zip' to jinja templates.
    env.globals.update(zip=zip)

    global staticpath
    staticpath = '.'

    # Start up tornado app.
    application = make_app()
    application.listen(args.port)
    print('Tornado Starting: \nPoint your web browser to http://localhost:%d/ \nCtrl-C to stop' % (args.port))
    if not args.noBrowser:
        webbrowser.open_new_tab('http://localhost:%d' % (args.port))
    ioloop.IOLoop.instance().start()
