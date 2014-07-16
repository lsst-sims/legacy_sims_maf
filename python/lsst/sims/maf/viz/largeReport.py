from tornado import ioloop
from tornado import web
from jinja2 import Environment, FileSystemLoader
from collections import OrderedDict
import glob

env = Environment(loader=FileSystemLoader('templates'))
class MetricGridPageHandler(web.RequestHandler):
    def get(self):
        gridTempl = env.get_template("grid.html")
        qargs = self.request.query_arguments
        self.write(gridTempl.render(metrics=qargs))

class SelectPageHandler(web.RequestHandler):
    def get(self):
        mainTempl = env.get_template("grid.html")
        # Load up the list of metrics (from a sqlite in the future)
        
        # Sort into metric groups and super-groups (same metric, different filters)
        #files = glob.glob('SS_6.15/*finSeeing*SkyMap*.png')
        filterOrder = ['u','g','r','i','z','y']
        hourglassFile = 'SS_6.15/opsimblitz2_1060_hourglass_HOUR_hr.png'
        mergedHist = 'SS_6.15/opsimblitz2_1060_opsimblitz2_1060_Count_finSeeing_ONED_hist.png'
        skyFiles = ['SS_6.15/opsimblitz2_1060_Median_finSeeing_u_OPSI_SkyMap.png',
                    'SS_6.15/opsimblitz2_1060_Median_finSeeing_g_OPSI_SkyMap.png',
                    'SS_6.15/opsimblitz2_1060_Median_finSeeing_r_OPSI_SkyMap.png',
                    'SS_6.15/opsimblitz2_1060_Median_finSeeing_i_OPSI_SkyMap.png',
                    'SS_6.15/opsimblitz2_1060_Median_finSeeing_z_OPSI_SkyMap.png',
                    'SS_6.15/opsimblitz2_1060_Median_finSeeing_y_OPSI_SkyMap.png']
        summaryTable = {'Mean_u': 0.85297, 'Mean_g':0.8341885,
                        'Mean_r':0.7951665 ,'Mean_i':0.783639 ,'Mean_z':0.764888 ,'Mean_y':0.736021 }
        block = {'skyMaps':skyFiles , 'skyMapsCaption': 'A caption for sky maps', 
                'mergedHist':[mergedHist], 'mergedHistCaption':'A caption for mergedHist',
                 'hourglassPlot':hourglassFile, 'hourglassCaption':'A hourglass caption',
                'summaryTables':summaryTable, 'BlockTitle':'Block Title',
                 'completenessTable':{}, }
        # Send a list of plot and table blobs
        # Could gerenate thumbnails for all the files, then display the thumbs and
        # link to the full files.
        #metricDict = OrderedDict([(files[i], i+1) for i in range(len(files))])
        #self.write(mainTempl.render(metrics=metricDict))
        self.write(mainTempl.render(metrics=block))

outDir = 'SS_6.15'
application = web.Application([
    ("/metricResult", MetricGridPageHandler),
    ("/", SelectPageHandler),
    (r"/"+outDir+"/(.*)", web.StaticFileHandler, {'path':outDir}), 
    (r"/(favicon.ico)", web.StaticFileHandler, {'path':outDir}),
])

if __name__ == "__main__":
    application.listen(8888)
    ioloop.IOLoop.instance().start()
