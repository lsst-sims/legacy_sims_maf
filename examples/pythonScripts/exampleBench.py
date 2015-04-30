import lsst.sims.maf.benchmarks as benchmarks
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers


metric = metrics.MeanMetric(col='HA')
slicer = slicers.HealpixSlicer(nside=4)
stackerList = [stackers.NormAirmassStacker()]

bm = benchmarks.Benchmark(metric, slicer, stackerList=stackerList, sqlconstraint='filter="r"')

print bm.dbColNames
