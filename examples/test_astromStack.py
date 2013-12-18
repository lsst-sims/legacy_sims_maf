import numpy as np
import lsst.sims.operations.maf.utils.astromStack as asstack
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.gridMetrics as gridMetrics


bandpass = 'r'

dbTable = 'output_opsim3_61'
dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'


table = db.Table(dbTable, 'obshistid', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
                                       colnames=['filter', 'expmjd',  'night',
                                                 'fieldra', 'fielddec', 'airmass',
                                                 '5sigma_modified', 'seeing',
                                                 'skybrightness_modified', 'altitude',
                                                 'hexdithra', 'hexdithdec'], 
                                                 groupByCol='expmjd')


# Fixing stupid postgres case-sensitivity issues.
simdata.dtype.names = 'obsHistID', 'filter', 'expMJD', 'night', 'fieldRA', 'fieldDec', 'airmass', '5sigma_modified', 'seeing', 'skybrightness_modified', 'altitude', 'hexdithra', 'hexdithdec'

# Eliminate the observations where hexdithra has failed for some reason
good=np.where((simdata['hexdithra'] < np.pi*2) )
simdata=simdata[good]


# Crop down a bit
good = np.where( (simdata['fieldRA'] > 0) & (simdata['fieldRA'] < np.radians(40)) & (simdata['fieldDec'] > np.radians(-60)) & (simdata['fieldDec'] < np.radians(-20)))
simdata = simdata[good]

# Add on the paralax factor per observation
simdata = asstack.astroStack(simdata)



nside = 128
gg = grids.HealpixGrid(nside)
gg.buildTree(simdata['fieldRA'], simdata['fieldDec'], leafsize=50000)

pmMetric = metrics.ProperMotionMetric(badval = gg.badval)


metricList = [pmMetric]

gm = gridMetrics.SpatialGridMetric()
gm.setGrid(gg)

gm.runGrid(metricList, simdata, simDataName=dbTable, metadata = bandpass)

gm.plotAll(savefig=True)

gm.writeAll()
