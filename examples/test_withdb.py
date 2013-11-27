import numpy
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics

# set up some test data from a database.
#dbTable = 'output_opsim3_61'
#dbTable = 'output_opsim2_145'
#dbAddress = 'mysql://lsst:lsst@localhost/opsimdev'
bandpass = 'r'
dbTable = 'output_opsim3_61_forLynne' 
#dbTable = 'output_opsim2_145_forLynne'   
dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'  
table = db.Table(dbTable, 'obsHistID', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expMJD', 'fieldRA', 'fieldDec',
                                        '5sigma_modified', 'seeing'], groupByCol='expMJD')

print 'Got simdata', numpy.shape(simdata)

# Set up grid.
gg = grids.GlobalGrid()

# Set up metrics.
magmetric = metrics.MeanMetric('5sigma_modified')
seeingmean = metrics.MeanMetric('seeing')
seeingrms = metrics.RmsMetric('seeing')

print magmetric.classRegistry

gm = gridMetrics.BaseGridMetric(gg)
gm.runGrid([magmetric, seeingmean, seeingrms], simdata)

#print gm.metricValues
print gm.metricValues[magmetric.name]
print gm.metricValues[seeingmean.name]
print gm.metricValues[seeingrms.name]
