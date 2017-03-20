from __future__ import print_function
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.db as db

# Connect to opsim
dbAddress = 'sqlite:///ops1_1140_sqlite.db'
oo = db.OpsimDatabase(dbAddress)
colnames = ['expMJD', 'fieldRA', 'fieldDec']
sqlconstraint ='filter="r"'
# Get opsim simulation data
simdata = oo.fetchMetricData(colnames, sqlconstraint)
# Init the slicer, set 2 points
slicer = slicers.UserPointsSlicer(ra=[0., .1], dec=[0., -.1])
# Setup slicer (builds kdTree)
slicer.setupSlicer(simdata)
# Slice Point for index zero
ind = slicer._sliceSimData(0)
expMJDs = simdata[ind['idxs']]['expMJD']
print('mjd for the 1st user defined point', expMJDs)
# Find the expMJDs for the 2nd point
ind = slicer._sliceSimData(1)
expMJDs = simdata[ind['idxs']]['expMJD']
print('mjd for the 2nd user defined point', expMJDs)
