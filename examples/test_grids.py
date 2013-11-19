import numpy
from lsst.sims.maf.grids.baseGrid import BaseGrid
from lsst.sims.maf.grids.healpixGrid import HealpixGrid
from lsst.sims.maf.grids.globalGrid import GlobalGrid

# set up some test data
simdata = numpy.recarray((1000), dtype=([('seeing', 'float'), ('expmjd', 'float'), ('mag', 'float'), ('fieldra', 'float'), ('fielddec', 'float')]))
simdata['seeing'] = numpy.random.randn(1000)
simdata['expmjd'] = numpy.arange(0, 1000, 1.0)
simdata['mag'] = numpy.random.rand(1000) + 25.0
simdata['fieldra'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)
simdata['fielddec'] = numpy.random.rand(1000)*numpy.radians(10) + numpy.radians(20.0)

print numpy.shape(simdata['fieldra'])
print simdata['fieldra'][0]

bb = GlobalGrid()

print '#GLOBAL GRID'
print 'Verbose?', bb.verbose
print 'Length:', len(bb)
if hasattr(bb, '__iter__'):
    try:
        for i,b in enumerate(bb):
            print i, b
    except NotImplementedError:
        print "Iteration not implemented yet"
else:
    "not iterable"



bb = HealpixGrid(1)
bb.buildTree(simdata['fieldra'], simdata['fielddec'])
print '# HEALPIX GRID'
print 'Verbose?', bb.verbose
print 'Length:', len(bb)
if hasattr(bb, '__iter__'):
    try:
        for i,b in enumerate(bb):
            print i, b
    except NotImplementedError:
        print "Iteration not implemented yet"
else:
    "not iterable"



