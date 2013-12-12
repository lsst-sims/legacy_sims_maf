import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics
import glob

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


t= time.time()

# Set up grid metric object. 
gm = gridMetrics.GlobalGridMetric()

dt, t = dtime(t)
print 'Set up gridmetric %f s' %(dt)

# Read in grid pickle

gridfile = 'output_opsim3_61_grid.obj_gl'
gm.readGrid(gridfile)

# Read in metric files

filenames = glob.glob('output_opsim3_61*gl.fits')

gm.readMetric(filenames)

dt, t = dtime(t)
print 'Read metric files and set up/unpickled grid %f s' %(dt)

print 'Metrics read' , gm.metricValues.keys()
#print 'Metric simnames', gm.simDataName
#print 'Metric metadata', gm.metadata
#print 'Metric comments', gm.comment


#print len(gm.metricHistBins['Max_5sigma_modified'])
#print len(gm.metricHistValues['Max_5sigma_modified'])
#print len(gm.metricHistBins['Max_5sigma_modified__0'])
#print len(gm.metricHistValues['Max_5sigma_modified__0'])
#print gm.simDataName['Max_5sigma_modified']
#print gm.metadata['Max_5sigma_modified']

# Generate comparison plots

compare = ['Max_5sigma_modified','Min_seeing', 'coaddm5']
for c in compare:
    comparelist = []
    for k in gm.metricValues.keys():
        if k.startswith(c):
            comparelist.append(k)
    print 'Comparing', comparelist
    gm.plotComparisons(comparelist)

dt, t = dtime(t)
print 'Generated comparison plots %f s' %(dt)

plt.show()
