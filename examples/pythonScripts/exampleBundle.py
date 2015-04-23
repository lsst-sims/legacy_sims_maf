import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.driver import runBundle


# Pick a slicer
slicer = slicers.HealpixSlicer(nside=64)

# Configure some metrics
metricList = []
metricList.append(metrics.Coaddm5Metric(summaryStatList=[metrics.MeanMetric(), metrics.RmsMetric()]))
metricList.append(metrics.MeanMetric(col='airmass',
                                     summaryStatList=[metrics.MeanMetric(), metrics.RmsMetric()]))


# Set the database and query
dbAddress = 'sqlite:///enigma_1189_sqlite.db'
sqlWhere = 'filter = "r" and night < 200'


mafBundle = {'slicer':slicer, 'metricList':metricList, 'dbAddress':dbAddress,
             'sqlWhere':sqlWhere, 'outDir':'BundleExampleOut'}

metricResults = runBundle(mafBundle)
