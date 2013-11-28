import numpy
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics

# set up some test data
#simdata = tu.makeSimpleTestSet()
#print 'simdata shape', numpy.shape(simdata)
#print simdata.dtype.names

# get sim data from DB
bandpass = 'r'
dbTable = 'output_opsim3_61_forLynne' 
#dbTable = 'output_opsim2_145_forLynne'   
dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'  
table = db.Table(dbTable, 'obsHistID', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expMJD',  'night',
                                                 'fieldRA', 'fieldDec',
                                                 '5sigma_modified', 'seeing'], 
                                                 groupByCol='expMJD')


# Set up global grid.
#gg = grids.GlobalGrid()

# Set up spatial grid.
gg = grids.HealpixGrid(1)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['fieldRA'], simdata['fieldDec'])

# Set up metrics.
dtmin = 1./60./24.
dtmax = 360./60./24.
visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

meanseeing = metrics.MeanMetric('seeing')

gm = gridMetrics.BaseGridMetric(gg)
#gm.runGrid([visitPairs,], simdata)
gm.runGrid([meanseeing,], simdata)
#print gm.metricValues[visitPairs.name]
gm.reduceAll()
gm.plotAll(savefig=False)

plt.show()

#print gm.metricValues[visitPairs.name]
#for k in visitPairs.reduceFuncs.keys():
#    print k, gm.reduceValues[visitPairs.name][k]

