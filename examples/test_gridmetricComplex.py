import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

# set up some test data
#simdata = tu.makeSimpleTestSet()
#print 'simdata shape', np.shape(simdata)
#print simdata.dtype.names

# get sim data from DB
bandpass = 'r'
#dbTable = 'output_opsim3_61_forLynne' 
#dbTable = 'output_opsim2_145_forLynne'   
#dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST' 

#dbTable = 'output_opsimblitz2_1007'
dbTable = 'output_opsim3_61'
dbAddress = 'mysql://lsst:lsst@localhost/opsim?unix_socket=/opt/local/var/run/mariadb/mysqld.sock'

t = time.time()

table = db.Table(dbTable, 'obsHistID', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expMJD',  'night',
                                                 'fieldRA', 'fieldDec', 'airmass',
                                                 '5sigma_modified', 'seeing',
                                                 'skybrightness_modified', 'altitude',
                                                 'hexdithra', 'hexdithdec'], 
                                                 groupByCol='expMJD')


dt, t = dtime(t)
print 'Query complete: %f s' %(dt)
print 'Retrieved %d observations' %(len(simdata['expMJD']))

nside = 128*2*2*2

# Set up spatial grid.
gg = grids.HealpixGrid(nside)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['fieldRA'], simdata['fieldDec'], leafsize=100)

dt, t = dtime(t)
print 'Set up grid (and built kdtree if spatial grid) %f s' %(dt)

# Set up metrics.
dtmin = 1./60./24.
dtmax = 360./60./24.
visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

meanseeing = metrics.MeanMetric('seeing')
minseeing = metrics.MinMetric('seeing')
maxseeing = metrics.MaxMetric('seeing')
rmsseeing = metrics.RmsMetric('seeing')
meanairmass = metrics.MeanMetric('airmass')
minairmass = metrics.MinMetric('airmass')
meanm5 = metrics.MeanMetric('5sigma_modified')
maxm5 = metrics.MaxMetric('5sigma_modified')
rmsm5 = metrics.RmsMetric('5sigma_modified')
meanskybright = metrics.MeanMetric('skybrightness_modified')
maxskybright = metrics.MaxMetric('skybrightness_modified')
coaddm5 = metrics.Coaddm5Metric('5sigma_modified')

#metricList = [meanseeing, minseeing, rmsseeing, meanairmass, minairmass, meanm5, minm5, rmsm5, 
#              meanskybright, maxskybright, coaddm5]
              #metricList = [meanseeing, minseeing, maxseeing]

metricList = [coaddm5, minseeing, maxm5]

dt, t = dtime(t)
print 'Set up metrics %f s' %(dt)

gm = gridMetrics.SpatialGridMetric(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)

gm.runGrid(metricList, simdata, simDataName=dbTable, metadata = bandpass)
dt, t = dtime(t)
print 'Ran grid of %d points with %d metrics using gridMetric %f s' %(len(gg), len(metricList), dt)
                    
gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=True)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)

gm.writeAll()


print 'Round 2 (dithered)'

gg = grids.HealpixGrid(nside)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['hexdithra'], simdata['hexdithdec'], leafsize=100)

dt, t = dtime(t)
print 'Set up grid and built kdtree if spatial grid %f s' %(dt)


gm = gridMetrics.SpatialGridMetric(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)

gm.runGrid(metricList, simdata, simDataName=dbTable, 
           metadata = bandpass + ' dithered' )

dt, t = dtime(t)
print 'Ran grid %f s' %(dt)

gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=True)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)

gm.writeAll()


plt.show()

#print gm.metricValues[visitPairs.name]
#for k in visitPairs.reduceFuncs.keys():
#    print k, gm.reduceValues[visitPairs.name][k]

