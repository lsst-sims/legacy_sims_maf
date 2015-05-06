import numpy as np
import matplotlib.pyplot as plt
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
metric = metrics.MeanMetric(col='airmass')
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
metric = metrics.MeanMetric(col='airmass')
bundle = metricBundles.MetricBundle(metric, slicer, sqlWhere)
bundleDict[counter] = bundle
counter += 1

# healpixSDSSSlicer

# healpixComplexSlicer

# f0Slicer -- this should just go to a healpixslicer with a different plotter.



bgroup = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=outDir, resultsDb=resultsDb)
bgroup.runAll()
bgroup.plotAll()
bgroup.writeAll()
