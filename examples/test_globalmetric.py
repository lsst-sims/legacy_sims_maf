import numpy
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics
import glob

# set up some test data
simdata = tu.makeSimpleTestSet()

gg = grids.GlobalGrid()

# Set up metrics.
magmetric = metrics.MeanMetric('m5')
seeingmean = metrics.MeanMetric('seeing')
seeingrms = metrics.RmsMetric('seeing')

gm = gridMetrics.GlobalGridMetric()
gm.setGrid(gg)
gm.runGrid([magmetric, seeingmean, seeingrms], simdata)
print gm.metricValues[magmetric.name]
print gm.metricValues[seeingmean.name]
print gm.metricValues[seeingrms.name]

gm.plotAll()

#gm.plotComparisons([magmetric.name, seeingmean.name, seeingrms.name])
plt.show()

exit()
gm.writeAll(outfileRoot='savetest')

filenames = glob.glob('savetest*.fits')

ack = gridMetrics.BaseGridMetric()
ack.readGrid('savetest_grid.obj')
ack.readMetric(filenames)
print gm.metricValues[magmetric.name]
print gm.metricValues[seeingmean.name]
print gm.metricValues[seeingrms.name]
