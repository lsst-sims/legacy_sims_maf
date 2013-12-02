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
                                                 'hexdithra', 'hexdithdec'], 
                                                 groupByCol='expMJD')


dt, t = dtime(t)
print 'Query complete: %f s' %(dt)
print 'Retrieved %d observations' %(len(simdata['expMJD']))

# Set up global grid.
#gg = grids.GlobalGrid()

# Set up spatial grid.
gg = grids.HealpixGrid(128)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['fieldRA'], simdata['fieldDec'], leafsize=100)

dt, t = dtime(t)
print 'Set up grid (and built kdtree if spatial grid) %f s' %(dt)

# Set up metrics.
dtmin = 1./60./24.
dtmax = 360./60./24.
visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

meanseeing = metrics.MeanMetric('seeing')
meanairmass = metrics.MeanMetric('airmass')
minairmass = metrics.MinMetric('airmass')
minm5 = metrics.MinMetric('5sigma_modified')
coaddm5 = metrics.Coaddm5Metric('5sigma_modified')

dt, t = dtime(t)
print 'Set up metrics %f s' %(dt)

gm = gridMetrics.BaseGridMetric(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)


## TEST
a = np.zeros(len(gg), 'object')
a2 = np.zeros(len(gg), 'object')
m = np.zeros(len(gg), 'object')
s = np.zeros(len(gg), 'object')
c = np.zeros(len(gg), 'object')
for i, g in enumerate(gg):
    idxs = gg.sliceSimData(g, simdata['seeing'])
    simslice = simdata[idxs]
    if len(idxs)==0:
        s[i] = gg.badval
        c[i] = gg.badval
        m[i] = gg.badval
        a[i] = gg.badval
        a2[i] = gg.badval
    else:
        a[i] = simdata['airmass'][idxs].mean()
        a2[i] = simdata['airmass'][idxs].min()
        m[i] = simdata['5sigma_modified'][idxs].min()
        s[i] = simdata['seeing'][idxs].mean()
        c[i] = 1.25 * np.log10(np.sum(10.**(.8*simdata['5sigma_modified'][idxs])))
dt, t = dtime(t)
print 'Ran grid here direct (and with idxs to individual direct numpy methods) %f s' %(dt)

a = np.zeros(len(gg), 'object')
a2 = np.zeros(len(gg), 'object')
m = np.zeros(len(gg), 'object')
s = np.zeros(len(gg), 'object')
c = np.zeros(len(gg), 'object')
for i, g in enumerate(gg):
    idxs = gg.sliceSimData(g, simdata['seeing'])
    simslice = simdata[idxs]
    if len(idxs)==0:
        s[i] = gg.badval
        c[i] = gg.badval
        m[i] = gg.badval
        a[i] = gg.badval
        a2[i] = gg.badval
    else:
        a[i] = simslice['airmass'].mean() #simdata['airmass'][idxs].mean()
        a2[i] = simslice['airmass'].min() #simdata['airmass'][idxs].min()
        m[i] = simslice['5sigma_modified'].min() #simdata['5sigma_modified'][idxs].min()
        s[i] = simslice['seeing'].mean() #simdata['seeing'][idxs].mean()
        c[i] = 1.25 * np.log10(np.sum(10.**(.8*simslice['5sigma_modified'])))
        #1.25 * np.log10(np.sum(10.**(.8*simdata['5sigma_modified'][idxs])))
dt, t = dtime(t)
print 'Ran grid here direct to numpy (but without idxs) %f s' %(dt)

a = np.zeros(len(gg), 'object')
a2 = np.zeros(len(gg), 'object')
m = np.zeros(len(gg), 'object')
s = np.zeros(len(gg), 'object')
c = np.zeros(len(gg), 'object')
for i, g in enumerate(gg):
    idxs = gg.sliceSimData(g, simdata['seeing'])
    if len(idxs)==0:
        s[i] = gg.badval
        c[i] = gg.badval
        m[i] = gg.badval
        a[i] = gg.badval
        a2[i] = gg.badval
    else:
        s[i] = meanseeing.run(simdata[idxs])
        c[i] = coaddm5.run(simdata[idxs])
        m[i] = minm5.run(simdata[idxs])
        a[i] = meanairmass.run(simdata[idxs])
        a[i] = minairmass.run(simdata[idxs])
dt, t = dtime(t)
print 'Ran grid here class methods, using simdata[idxs] %f s' %(dt)

gm.runGrid([meanseeing, coaddm5, minm5, meanairmass, minairmass], simdata, simDataName=dbTable.rstrip('_forLynne'))

dt, t = dtime(t)
print 'Ran grid using gridMetric %f s' %(dt)

exit()
                
gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=False)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)

exit()

print 'Round 2 (dithered)'

gg = grids.HealpixGrid(128)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['hexdithra'], simdata['hexdithdec'], leafsize=100)

dt, t = dtime(t)
print 'Set up grid and built kdtree if spatial grid %f s' %(dt)

# Set up metrics.
dtmin = 1./60./24.
dtmax = 360./60./24.
visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

meanseeing = metrics.MeanMetric('seeing')
coaddm5 = metrics.Coaddm5Metric('5sigma_modified')

dt, t = dtime(t)
print 'Set up metrics %f s' %(dt)

gm = gridMetrics.BaseGridMetric(gg)

dt, t = dtime(t)
print 'Set up gridMetric %f s' %(dt)

gm.runGrid([meanseeing, coaddm5], simdata, simDataName=dbTable.rstrip('_forLynne'), metadata='Dithered' )

dt, t = dtime(t)
print 'Ran grid %f s' %(dt)

gm.reduceAll()

dt, t = dtime(t)
print 'Ran reduce functions %f s' %(dt)

gm.plotAll(savefig=False)

dt, t = dtime(t)
print 'Made plots %f s' %(dt)



plt.show()

#print gm.metricValues[visitPairs.name]
#for k in visitPairs.reduceFuncs.keys():
#    print k, gm.reduceValues[visitPairs.name][k]

