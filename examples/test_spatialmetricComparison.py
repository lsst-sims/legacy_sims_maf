import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


t= time.time()

# Set up grid metric object. 
gm = gridMetrics.SpatialGridMetric()

dt, t = dtime(t)
print 'Set up gridmetric %f s' %(dt)

# Read in grid pickle

gridfile = 'output_opsim3_61_grid.obj_sp'
gm.readGrid(gridfile)

# Read in metric files

filenames =  ['output_opsim3_61_coaddm5_r_sp.fits', 'output_opsim3_61_coaddm5_r_dit_sp.fits', 
              'output_opsim3_61_Min_seeing_r_sp.fits', 'output_opsim3_61_Min_seeing_r_dit_sp.fits',
              'output_opsim3_61_Max_5sigma_modified_r_sp.fits', 'output_opsim3_61_Max_5sigma_modified_r_dit_sp.fits']

gm.readMetric(filenames)

dt, t = dtime(t)
print 'Read metric files and set up/unpickled grid %f s' %(dt)

print 'Metrics read' , gm.metricValues.keys()
#print 'Metric simnames', gm.simDataName
#print 'Metric metadata', gm.metadata
#print 'Metric comments', gm.comment

# Generate comparison plots

compare = ['Max_5sigma_modified', 'Min_seeing', 'coaddm5']
for c in compare:
    comparelist = []
    for k in gm.metricValues.keys():
        if k.startswith(c):
            comparelist.append(k)
    print 'Comparing', comparelist
    gm.plotComparisons(comparelist, savefig=True)

dt, t = dtime(t)
print 'Generated comparison plots %f s' %(dt)

plt.show()
