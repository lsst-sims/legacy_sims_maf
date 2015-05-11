import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.plots import NeoDetectPlotter


# Set up the database connection
opsdb = db.OpsimDatabase('sqlite:///enigma_1189_sqlite.db')
outDir = 'NeoDetect'
resultsDb = db.ResultsDb(outDir=outDir)

slicer = slicers.UniSlicer()
metric = metrics.PassMetric(metricName='NEODistances')
stacker = stackers.NEODistStacker()
filters = ['u','g','r','i','z','y']
counter = 0
bundleDict = {}
for filterName in filters:
    bundle = metricBundles.MetricBundle(metric, slicer,
                                        'night < 365 and filter="%s"'%filterName,
                                        stackerList=[stacker],
                                        plotDict={'title':'%s-band'%filterName},
                                        plotFuncs=[NeoDetectPlotter()])

    bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll()


#plotter = NeoDetectPlotter()
#plotter(bundle.metricValues, bundle.slicer, {})
#plt.show()
