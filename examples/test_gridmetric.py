import numpy
import lsst.sims.maf.grids as grids
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.gridMetrics import BaseGridMetric

# set up some test data
simdata = numpy.recarray((1000), dtype=([('seeing', 'float'), ('expmjd', 'float'), ('mag', 'float'), ('fieldra', 'float'), ('fielddec', 'float')]))
simdata['seeing'] = numpy.random.randn(1000)
simdata['expmjd'] = numpy.arange(0, 1000, 1.0)
simdata['mag'] = numpy.random.rand(1000) + 25.0
simdata['fieldra'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)
simdata['fielddec'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)

print 'simdata shape', numpy.shape(simdata)
print simdata.dtype.names
print simdata.dtype.names[0]
print numpy.shape(simdata[simdata.dtype.names[0]])

gr = grids.GlobalGrid()

magmetric = metrics.MeanMetric('mag')
seeingmetric = metrics.MeanMetric('seeing')

print magmetric.name
print seeingmetric.name

print magmetric.classRegistry

gg = BaseGridMetric(gr, [magmetric, seeingmetric], simdata)
gg.runGrid()
print gg.metricValues
#print gg.metricValues['magmetric']
#print gg.metricValues['seeingmetric']
