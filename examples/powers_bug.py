#trying to replicate bug with generating power spectra

import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.utils.testUtils as tu
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics


# Load data from postgres
dbTable = 'output_opsim3_61'
dbAddress = 'postgres://calibuser:calibuser@ivy.astro.washington.edu:5432/calibDB.05.05.2010'
bandpass = 'r'
table = db.Table(dbTable, 'obshistid', dbAddress)
simdata = table.query_columns_RecArray(constraint="filter = \'%s\'" %(bandpass), 
    colnames=['filter', 'expmjd',  'night',
    'fieldra', 'fielddec', 'airmass','5sigma_modified', 'seeing',
    'skybrightness_modified', 'altitude','hexdithra', 'hexdithdec'], 
    groupByCol='expmjd')
# Fixing stupid postgres case-sensitivity issues.
simdata.dtype.names = 'obsHistID', 'filter', 'expMJD', 'night', 'fieldRA', 'fieldDec', 'airmass', '5sigma_modified', 'seeing', 'skybrightness_modified', 'altitude', 'hexdithra', 'hexdithdec'
# Eliminate the observations where hexdithra has failed for some reason
good=np.where((simdata['hexdithra'] < np.pi*2) )
simdata=simdata[good]
# Crop down to 2 years, limited RA/Dec range
good= np.where((simdata['night'] < 365*2.) & (simdata['fieldRA'] > 0) &
               (simdata['fieldRA'] < np.radians(60.))
               & (simdata['fieldDec'] >  np.radians(-60)) &
               (simdata['fieldDec'] < np.radians(-20))  )
simdata=simdata[good]


nside = 128*2#*2*2
gg = grids.HealpixGrid(nside)
# Build kdtree on ra/dec for spatial grid.
gg.buildTree(simdata['fieldRA'], simdata['fieldDec'], leafsize=500)
minseeing = metrics.MinMetric('seeing')
coaddm5 = metrics.Coaddm5Metric('5sigma_modified')
metricList = [coaddm5, minseeing]
gm = gridMetrics.SpatialGridMetric()
gm.setGrid(gg)
gm.runGrid(metricList, simdata, simDataName=dbTable, metadata = bandpass)
gm.reduceAll()
gm.plotAll(savefig=True)

for m in metricList:
    mean = gm.computeSummaryStatistics(m.name, np.mean)
    print "Mean of ", m.name, mean

gm.writeAll()
# Save for latter comparison
gm_orig=gm

filenames=['output_opsim3_61_Min_seeing_r_sp.fits','output_opsim3_61_coaddm5_r_sp.fits']

#read in the metrics we just calculated.
gm.readMetric(filenames)

#add a little noise so that we don't get pure zero comparison
names = ['coaddm5__0','Min_seeing__0']
for n in names:
    good = np.where(gm.metricValues[n] != gm.grid.badval)
    gm.metricValues[n][good] = gm.metricValues[n][good] + \
    np.random.randn(np.size(good[0]) )*gm.metricValues[n][good].std()


# Are the values similar?
print 'mean diff in min seeing = %f'%np.mean(gm.metricValues['Min_seeing__0'][good] -gm.metricValues['Min_seeing'][good] )


# This throws an error, and the values of Min_seeing__0 are now wacky.
gm.plotComparisons(['Min_seeing','Min_seeing__0'], savefig=True)





