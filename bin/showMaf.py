#! /usr/bin/env python
import numpy as np
from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
import os, argparse
from lsst.sims.maf.viz import layoutResults


class MasterPageHandler(web.RequestHandler):
    def get(self):
        masterTempl = env.get_template("master.html")
        self.write(masterTempl.render(run=layout))

class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("grid.html")
        metricIds = self.request.query_arguments
        self.write(gridTempl.render(metricIds=metricIds, run=layout))

class MetricSelectHandler(web.RequestHandler):
    def get(self):
        selectTempl = env.get_template("select.html")
        self.write(selectTempl.render(run=layout))

class ConfigPageHandler(web.RequestHandler):
    def get(self):
        configTempl = env.get_template("configs.html")
        self.write(configTempl.render(run=layout))

class MonsterPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        monsterTempl = env.get_template("monster.html")
        self.write(monsterTempl.render(outDir=outDir, **layout.packageMonster()))

        
def make_app():
    """The tornado global configuration """
    application = web.Application([
            ("/master", MasterPageHandler),
            ("/metricSelect", MetricSelectHandler),
            ("/metricResults", MetricGridPageHandler),
            ("/configParams", ConfigPageHandler),
            ("/monster", MonsterPageHandler),
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
    
