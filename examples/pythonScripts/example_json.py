import lsst.sims.maf.sliceMetrics as sliceMetrics

# Data file name. 
infilename = 'ops2_1075_sstar/ops2_1075_Nvisits_r_band_WFD_only_HEAL.npz'
outfilename = 'ops2_1075_Nvisits_r_band_WFD_only_HEAL.json'

# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(infilename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
metricName = sm.metricNames[iid]
simDataName = sm.simDataNames[iid]
metadata = sm.metadatas[iid]

slicer.writeJSON(outfilename, metricValues, metricName=metricName,
                 simDataName =simDataName, metadata=metadata) 

# Data file name. 
infilename = 'ops2_1075_sstar/ops2_1075_M5_r_histogram_r_band_WFD_only_ONED.npz'
outfilename = 'ops2_1075_M5_r_band_WFD_only_ONED.json'

# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(infilename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
metricName = sm.metricNames[iid]
simDataName = sm.simDataNames[iid]
metadata = sm.metadatas[iid]

slicer.writeJSON(outfilename, metricValues, metricName=metricName,
                 simDataName =simDataName, metadata=metadata) 
