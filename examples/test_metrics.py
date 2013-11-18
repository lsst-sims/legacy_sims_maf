import numpy
from lsst.sims.maf.grids import *
import lsst.sims.maf.metrics as mafmetrics

# set up some test data
simdata = numpy.recarray((1000), dtype=([('seeing', 'float'), ('expmjd', 'float'), ('mag', 'float'), ('fieldra', 'float'), ('fielddec', 'float')]))
simdata['seeing'] = numpy.random.randn(1000)
simdata['expmjd'] = numpy.arange(0, 1000, 1.0)
simdata['mag'] = numpy.random.rand(1000) + 25.0
simdata['fieldra'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)
simdata['fielddec'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)

metric = mafmetrics.MeanMetric('mag')

print metric.classRegistry

grid = GlobalGrid()

testmetricvalues = numpy.zeros(len(grid), 'float')
print len(grid)

for i,g in enumerate(grid):
    idx = grid.sliceSimData(g, simdata['mag'])
    print len(simdata[idx])
    print i, g, metric.run(simdata[idx]), simdata['mag'].mean()
    #print i, g, idx, metric.run(simdata[idx])
    testmetricvalues[i] = metric.run(simdata[idx])

#print testmetricvalues
