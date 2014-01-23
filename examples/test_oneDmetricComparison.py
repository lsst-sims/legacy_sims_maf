CURRENTLY NEEDS WORKS (COMPARISON PLOTS)

import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics
import glob

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


t= time.time()

# Set up binMetric object. 
gm = binMetrics.BaseBinMetric()

dt, t = dtime(t)
print 'Set up gridmetric %f s' %(dt)

# Read in grid pickle

binfile = 'output_opsim3_61_binner.obj_ON'
gm.readBinner(binfile)

# Read in metric files

filenames = glob.glob('output_opsim3_61*ON.fits')

gm.readMetric(filenames, checkBinner=False)

dt, t = dtime(t)
print 'Read metric files and set up/unpickled grid %f s' %(dt)

print 'Metrics read' , gm.metricValues.keys()
#print 'Metric simnames', gm.simDataName
#print 'Metric metadata', gm.metadata
#print 'Metric comments', gm.comment


print 'Count_seeing_r', len(gm.metricValues['Count_seeing'])
print gm.metricValues['Count_seeing']

print 'Count_seeing_i', len(gm.metricValues['Count_seeing__0'])
print gm.metricValues['Count_seeing__0']

# Generate comparison plots

compare = ['Count_seeing',]
for c in compare:
    comparelist = []
    for k in gm.metricValues.keys():
        if k.startswith(c):
            comparelist.append(k)
    print 'Comparing', comparelist
    if c == 'Count_seeing':
       gm.plotComparisons(comparelist, legendloc = 'upper right')
    else:
       gm.plotComparisons(comparelist)

dt, t = dtime(t)
print 'Generated comparison plots %f s' %(dt)

plt.show()
