import numpy as np
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

# Let's try running all the Slicers and check that things work


# Set up the database connection
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
outDir = 'AllSlicers'
resultsDb = db.ResultsDb(outDir=outDir)
sqlWhere = 'night < 365'

bundleList = []

# Hourglass slicer
slicer = slicers.HourglassSlicer()
metric = metrics.HourglassMetric()
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# UniSlicer
slicer = slicers.UniSlicer()
metric = metrics.MeanMetric(col='airmass')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# HealpixSlicer
slicer = slicers.HealpixSlicer(nside=16)
metric = metrics.MeanMetric(col='airmass', metricName='MeanAirmass_heal')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# OneDSlicer
slicer = slicers.OneDSlicer(sliceColName='night', binsize=10)
metric = metrics.CountMetric(col='expMJD')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# OpsimFieldSlicer
slicer = slicers.OpsimFieldSlicer()
metric = metrics.MeanMetric(col='airmass')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# UserPointsSlicer
ra = np.arange(0,101,1)/100.*np.pi
dec = np.arange(0,101,1)/100.*(-np.pi)
slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
metric = metrics.MeanMetric(col='airmass', metricName='meanAirmass_user')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleList.append(bundle)

# healpixComplexSlicer (healpix slicer + summaryHistogram)
bins = np.arange(0.5, 3.0, 0.1)
slicer = slicers.HealpixSlicer(nside=16)
metric = metrics.TgapsMetric(bins=bins)
plotDict = {'bins':bins}
plotFuncs = [plots.SummaryHistogram()]
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere, plotDict=plotDict, plotFuncs=plotFuncs)
bundleList.append(bundle)

# f0 plot -- this should just go to a healpixslicer with a different plotter.
slicer = slicers.HealpixSlicer(nside=64)
metric = metrics.CountMetric('expMJD', metricName='fO')
plotFuncs = [plots.FOPlot()]
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere, plotFuncs=plotFuncs)
bundleList.append(bundle)


# Run everything above
bundleDict = metricBundles.makeBundleDict(bundleList)
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll()
bgroup.writeAll()


# Make a 6-panel seeing plot
filters = ['u','g','r','i','z','y']
slicer = slicers.HealpixSlicer(nside=64)
metric = metrics.MeanMetric(col='finSeeing')
for f in filters:
    bundle = metricBundles.MetricBundle(metric, slicer, 'filter = "%s" and night < 365'%f)
    bundleDict = metricBundles.makeBundleDict([bundle])
    bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll()
    bgroup.writeAll()


