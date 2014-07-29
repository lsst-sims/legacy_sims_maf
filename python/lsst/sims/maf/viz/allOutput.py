from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
from collections import OrderedDict
import glob
from .vizUtils import blockAll, loadResults, blockSS



env = Environment(loader=FileSystemLoader('templates'))
outDir = 'SS_7.22' #'Allslicers'
class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("allOut.html")
        qargs = self.request.query_arguments
        self.write(gridTempl.render(metrics=qargs))

class SelectPageHandler(web.RequestHandler):
    def get(self):
        """Load up the files and display """
        metrics, plots, stats = loadResults(outDir)
        
        mainTempl = env.get_template("allOut.html")
        blocks = blockAll(metrics, plots, stats)
        self.write(mainTempl.render(metrics=blocks, outDir=outDir))

application = web.Application([
    ("/metricResult", MetricGridPageHandler),
    ("/", SelectPageHandler),
    (r"/"+outDir+"/(.*)", web.StaticFileHandler, {'path':outDir}), 
    (r"/(favicon.ico)", web.StaticFileHandler, {'path':outDir}),
])

if __name__ == "__main__":
    application.listen(8888)
    ioloop.IOLoop.instance().start()
