import os
import numpy as np
import matplotlib.pyplot as plt, mpld3

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.sliceMetrics as sliceMetrics


# Data file name. 
filename = 'OutSstar2_1075/ops2_1075_Nvisits_g_HEAL.npz'

# Read data file and get dictionary key.
sm = sliceMetrics.BaseSliceMetric()
iid = sm.readMetricData(filename)
iid = iid[0]

slicer = sm.slicers[iid]
metricValues = sm.metricValues[iid]
plotParams = sm.plotParams[iid]
plotParams['title'] = sm.simDataNames[iid] + ' ' + sm.metadatas[iid] + ': ' + sm.metricNames[iid]
plotParams['xlabel'] = sm.metricNames[iid]
plotParams['colorMin'] = 0
plotParams['colorMax'] = 200
slicer.plotSkyMap(metricValues, **plotParams)
# To show using mpld3 : 
#mpld3.show()
# To show result using matplotlib :
plt.show()



