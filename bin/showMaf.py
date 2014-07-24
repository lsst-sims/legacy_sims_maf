#! /usr/bin/env python

from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
from collections import OrderedDict
from lsst.sims.maf.viz.vizUtils import  loadResults, blockSS
import os, argparse



class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("main.html")
        qargs = self.request.query_arguments
        self.write(gridTempl.render(metrics=qargs))

class SelectPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        metrics, plots, stats, runName = loadResults(outDir)
        mainTempl = env.get_template("ssTemplate.html")
        blocks = blockSS(metrics, plots, stats)
        self.write(mainTempl.render(outDir=outDir, runName=runName, **blocks))

        
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

    parser = argparse.ArgumentParser(description="Python script to display MAF output in a web browser.  After launching, point your browser to 'http://localhost:8888/'")
    parser.add_argument("outDir", help="Directory holding the MAF output to be displayed")
    args = parser.parse_args()
    outDir = args.outDir
    
    mafDir = os.getenv('SIMS_MAF_DIR')
    templateDir = os.path.join(mafDir, 'python/lsst/sims/maf/viz/templates/' )
    faviconPath = os.path.join(mafDir, 'python/lsst/sims/maf/viz/')
    env = Environment(loader=FileSystemLoader(templateDir))
    
    application = make_app()
    application.listen(8888)
    print 'Tornado Starting: \nPoint your web browser to http://localhost:8888/ \nCtrl-C to stop'

    ioloop.IOLoop.instance().start()
    
