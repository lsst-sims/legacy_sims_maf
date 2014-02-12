import numpy as np
import lsst.sims.operations.maf.binMetrics as binMetrics
import glob

cbm = binMetrics.ComparisonBinMetric(verbose=True)

filenames = glob.glob('*.fits')

cbm.readMetrics(filenames)

uniqueMetrics = list(cbm.uniqueMetrics())
uniqueMetadata = list(cbm.uniqueMetadata())
uniqueSimDataNames = list(cbm.uniqueSimDataNames())
print (uniqueSimDataNames)
print (uniqueMetadata)
print (uniqueMetrics)

print ''

print 'all dicts: ', cbm.identifyDictNums()
print 'dicts with simname', uniqueSimDataNames[0], '=', cbm.identifyDictNums(simDataName = uniqueSimDataNames[0])
print 'dicts with metadata', uniqueMetadata[0], '=',  cbm.identifyDictNums(metadata = uniqueMetadata[0])
print 'dicts with metric', uniqueMetrics[0], '=', cbm.identifyDictNums(metricNames = uniqueMetrics[0])



