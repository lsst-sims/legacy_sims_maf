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

# Set up some test data (for testing - no DB access).
#simdata = tu.makeSimpleTestSet()

# Get sim data from DB
# Set up database access info. 

# On Lynne's laptop
#dbTable = 'output_opsim3_61'
dbTable = 'output_opsimblitz2_1007'
dbAddress = 'mysql://lsst:lsst@localhost/opsim?unix_socket=/opt/local/var/run/mariadb/mysqld.sock'

# In department, use Peter's postgres.
#dbTable = 'output_opsim3_61'
#dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'

bandpass = 'r'


t = time.time()

table = db.Table(dbTable, 'obsHistID', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expMJD',  'night',
                                                 'fieldRA', 'fieldDec', 'airmass',
                                                 '5sigma_modified',  'finSeeing',
                                                 'skybrightness_modified', 
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

meanseeing = metrics.MeanMetric('finSeeing')
minseeing = metrics.MinMetric('finSeeing')
maxseeing = metrics.MaxMetric('finSeeing')
rmsseeing = metrics.RmsMetric('finSeeing')
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

metricList = [coaddm5, maxm5, meanm5, minseeing, rmsseeing, meanseeing, minairmass, meanairmass, visitPairs]

dt, t = dtime(t)
print 'Set up metrics %f s' %(dt)

gm = gridMetrics.SpatialGridMetric()
gm.setGrid(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)

gm.runGrid(metricList, simdata, simDataName=dbTable, metadata = bandpass)
dt, t = dtime(t)
print 'Ran grid of %d points with %d metrics using gridMetric %f s' %(len(gg), len(metricList), dt)
                    
gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=True, closefig=True)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)


for m in metricList:
   try:
      mean = gm.computeSummaryStatistics(m.name, np.mean)
      print "Mean of ", m.name, mean
   except ValueError:
      pass

gm.writeAll()


print 'Round 2 (dithered)'

gg = grids.HealpixGrid(nside)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['hexdithra'], simdata['hexdithdec'], leafsize=100)

dt, t = dtime(t)
print 'Set up grid and built kdtree if spatial grid %f s' %(dt)


gm = gridMetrics.SpatialGridMetric()
gm.setGrid(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)

gm.runGrid(metricList, simdata, simDataName=dbTable, 
           metadata = bandpass + ' dithered' )

dt, t = dtime(t)
print 'Ran grid %f s' %(dt)

gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=True, closefig=True)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)

for m in metricList:
   try:
      mean = gm.computeSummaryStatistics(m.name, np.mean)
      print "Mean of ", m.name, mean
   except ValueError:
      pass
   

gm.writeAll()

#plt.show()

