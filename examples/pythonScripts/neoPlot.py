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
bundle = metricBundles.MetricBundle(metric, slicer, 'night < 10', stackerList=[stacker])

bgroup = metricBundles.MetricBundleGroup({0:bundle}, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()

plotter = NeoDetectPlotter()
plotter(bundle.metricValues, bundle.slicer, {})
plt.show()
