import lsst.sims.maf.benchmarks as benchmarks
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.db as db



database = db.OpsimDatabase('sqlite:///enigma_1189_sqlite.db')


metric = metrics.MeanMetric(col='HA')
slicer = slicers.HealpixSlicer(nside=4)
stackerList = [stackers.NormAirmassStacker()]

bm = benchmarks.Benchmark(metric, slicer, stackerList=stackerList, sqlconstraint='filter="r" and night < 100')
metric = metrics.RmsMetric(col='airmass')
bm2 = benchmarks.Benchmark(metric, slicer, stackerList=stackerList, sqlconstraint='filter="r" and night < 100')


print bm.dbCols

bmD = {0:bm, 1:bm2}

bmg = benchmarks.BenchmarkGroup(bmD, database, outDir='test')
bmg.getData()
bmg.runAll()
