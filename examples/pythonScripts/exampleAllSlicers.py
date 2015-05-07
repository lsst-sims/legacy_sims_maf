import numpy as np
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots

# Let's try running all the Slicers and check that things work


# Set up the database connection
opsdb = db.OpsimDatabase('sqlite:///enigma_1189_sqlite.db')
outDir = 'AllSlicers'
resultsDb = db.ResultsDb(outDir=outDir)
sqlWhere = 'night < 365'

bundleDict={}
counter=0

# Hourglass slicer
slicer = slicers.HourglassSlicer()
metric = metrics.HourglassMetric()
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# UniSlicer
slicer = slicers.UniSlicer()
metric = metrics.MeanMetric(col='airmass')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# HealpixSlicer
slicer = slicers.HealpixSlicer(nside=16)
metric = metrics.MeanMetric(col='airmass', metricName='MeanAirmass_heal')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# OneDSlicer
slicer = slicers.OneDSlicer(sliceColName='night', binsize=10)
metric = metrics.CountMetric(col='expMJD')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# OpsimFieldSlicer
slicer = slicers.OpsimFieldSlicer()
metric = metrics.MeanMetric(col='airmass')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# UserPointsSlicer
ra = np.arange(0,101,1)/100.*np.pi
dec = np.arange(0,101,1)/100.*(-np.pi)
slicer = slicers.UserPointsSlicer(ra=ra,dec=dec)
metric = metrics.MeanMetric(col='airmass', metricName='meanAirmass_user')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# healpixComplexSlicer

# f0Slicer -- this should just go to a healpixslicer with a different plotter.



# Run everything above
bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll()
bgroup.writeAll()




# healpixSDSSSlicer
# need to run setup pymssql to get this part to run. Do we want to add that to our dependencies?
runSDSS = False
if runSDSS:
    sdssDB = db.SdssDatabase('mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433')
    sqlWhere = "filter='r' and nStars > 0 and nGalaxy > 0"
    slicer = slicers.HealpixSDSSSlicer(nside=64, lonCol='RA1', latCol='Dec1')
    metric = metrics.MeanMetric(col='psfWidth')
    bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
    bgroup = metricBundles.MetricBundleGroup({0:bundle}, sdssDB, outDir=outDir, resultsDb=resultsDb)
    bgroup.runAll()
    bgroup.plotAll()
    bgroup.writeAll()
