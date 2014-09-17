import lsst.sims.maf.sliceMetrics as sliceMetrics

import time
def dtime(time_prev):
    return (time.time()-time_prev, time.time())


# Data file name. 
infilename = 'ops2_1075_sstar/ops2_1075_Nvisits_r_band_WFD_only_HEAL.npz'
outfilename = 'ops2_1075_Nvisits_r_band_WFD_only_HEAL'

# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(infilename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
metricName = sm.metricNames[iid]
simDataName = sm.simDataNames[iid]
metadata = sm.metadatas[iid]

t= time.time()
slicer.writeData(outfilename+'.npz', metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata)
dt, t = dtime(t)
print 'npz file took %s to write' %(dt)

slicer.writeJSON(outfilename, metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata) 
dt, t = dtime(t)
print 'json file took %s to write' %(dt)

# Data file name. 
infilename = 'ops2_1075_sstar/ops2_1075_M5_r_histogram_r_band_WFD_only_ONED.npz'
outfilename = 'ops2_1075_M5_r_band_WFD_only_ONED'

# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(infilename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
metricName = sm.metricNames[iid]
simDataName = sm.simDataNames[iid]
metadata = sm.metadatas[iid]


t= time.time()
slicer.writeData(outfilename+'.npz', metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata)
dt, t = dtime(t)
print 'npz file took %s to write' %(dt)

slicer.writeJSON(outfilename+'.json', metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata) 

dt, t = dtime(t)
print 'json file took %s to write' %(dt)

infilename = 'ops2_1075_cadence/ops2_1075_SupernovaMetric__HEAL.npz'
outfilename = 'ops2_1075_SN_HEAL'


# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(infilename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
metricName = sm.metricNames[iid]
simDataName = sm.simDataNames[iid]
metadata = sm.metadatas[iid]


t= time.time()
slicer.writeData(outfilename+'.npz', metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata)
dt, t = dtime(t)
print 'npz file took %s to write' %(dt)

slicer.writeJSON(outfilename+'.json', metricValues, metricName=metricName,
                 simDataName=simDataName, metadata=metadata) 

dt, t = dtime(t)
print 'json file took %s to write' %(dt)
