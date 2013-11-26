import numpy
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics

# set up some test data
simdata = tu.makeSimpleTestSet()

print 'simdata shape', numpy.shape(simdata)
print simdata.dtype.names

# Set up grid.
gg = grids.GlobalGrid()

# Set up metrics.
dtmin = 1./60./24.
dtmax = 360./60./24.
print dtmin, dtmax
visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

gm = gridMetrics.BaseGridMetric(gg)
gm.runGrid([visitPairs,], simdata)
gm.reduceMetric(visitPairs)

print gm.metricValues[visitPairs.name]
for k in visitPairs.reduceFuncs.keys():
    print k, gm.reduceValues[visitPairs.name][k]

