#! /usr/bin/env python
import numpy as np
from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
import os, argparse
from lsst.sims.maf.viz import layoutResults


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
    
