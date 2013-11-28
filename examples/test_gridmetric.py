import numpy
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics
import glob

# set up some test data
simdata = tu.makeSimpleTestSet()

print 'simdata shape', numpy.shape(simdata)
print simdata.dtype.names

# Set up grid.
#gg = grids.GlobalGrid()
gg = grids.HealpixGrid()
gg.buildTree(simdata['fieldra'],simdata['fielddec'])

# Set up metrics.
magmetric = metrics.MeanMetric('m5')
seeingmean = metrics.MeanMetric('seeing')
seeingrms = metrics.RmsMetric('seeing')

print magmetric.classRegistry

gm = gridMetrics.BaseGridMetric(gg)
#gm.setupRun([magmetric, seeingmean, seeingrms], simdata)
gm.runGrid([magmetric, seeingmean, seeingrms], simdata)
#print gm.metricValues
print gm.metricValues[magmetric.name]
print gm.metricValues[seeingmean.name]
print gm.metricValues[seeingrms.name]

#try to save and restore
#gm.writeMetric([magmetric, seeingmean, seeingrms],outfile_root='savetest')
gm.writeAll(outfile_root='savetest')
#the fits files that were output
filelist = glob.glob('savetest*.fits')

#new object to restore info into
ack = gridMetrics.BaseGridMetric(None) #can instant with None grid since it will be loaded.
ack.readMetric(filelist, gridfile='savetestgrid.obj')
print ack.metricValues[magmetric.name]
print ack.metricValues[seeingmean.name]
print ack.metricValues[seeingrms.name]
