import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.binMetrics as binMetrics
import glob

cbm = binMetrics.ComparisonBinMetric(verbose=True)
dictnums = []
dictnames = []

# Read the metric data
filenames = glob.glob('spie/*.npz')
dictnums = []
filelist = []
for f in filenames:
    print 'working on %s' %(f)
    dictnum = cbm.readMetricData(f)
    dictnums.append(dictnum)
    dictnames.append(f)

####            
for num, name in zip(dictnums, dictnames):
    print num, name
        
print 'Healpix comparisons'
# Find the dict nums with healpix binners
oneDDicts = cbm.findDictNums(binnerName='HealpixBinner')
# Find the metric names associated with those oneD binmetrics 
oneDmetrics = cbm.uniqueMetrics(dictNums=oneDDicts)
# Plot the same metrics on the same plot
for mname in oneDmetrics:
    dicts = cbm.findDictNums(metricNames=mname, binnerName='HealpixBinner')
    metricnames = [mname for d in dicts]
    cbm.plotHistograms(dicts, metricnames)


    
plt.show()
    
