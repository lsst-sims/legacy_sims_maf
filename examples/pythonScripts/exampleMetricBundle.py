import matplotlib.pyplot as plt
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.db as db



database = db.OpsimDatabase('sqlite:///enigma_1189_sqlite.db')


metric = metrics.MeanMetric(col='HA')
slicer = slicers.HealpixSlicer(nside=4)
stackerList = [stackers.NormAirmassStacker()]

mb = metricBundles.MetricBundle(metric, slicer, stackerList=stackerList, sqlconstraint='filter="r" and night < 100')
metric = metrics.RmsMetric(col='airmass')
mb2 = metricBundles.MetricBundle(metric, slicer, stackerList=stackerList, sqlconstraint='filter="r" and night < 100')


print mb.dbCols

mbD = {0:mb, 1:mb2}

mbg = metricBundles.MetricBundleGroup(mbD, database, outDir='test')
mbg.runAll()
mbg.plotAll(closefigs=False)
plt.show()
