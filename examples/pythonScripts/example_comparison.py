import matplotlib.pyplot as plt
import lsst.sims.maf.sliceMetrics as sliceMetrics
import glob


cbm = sliceMetrics.ComparisonSliceMetric(verbose=True)

# Read some metric data in from files.
filenames = glob.glob('*.npz')
filelist = []
for f in filenames:
    if 'VisitPairs' not in f:
        print 'working on %s' %(f)
        cbm.readMetricData(f)

uniqueMetrics = list(cbm.uniqueMetrics())
uniqueMetadata = list(cbm.uniqueMetadata())
uniqueSimDataNames = list(cbm.uniqueSimDataNames())
print (uniqueSimDataNames)
print (uniqueMetadata)
print (uniqueMetrics)

print ''

print 'all iids: ', cbm.findDictNums()
print 'iids with simname', uniqueSimDataNames[0], '=', cbm.findIids(simDataName = uniqueSimDataNames[0])
print 'iids with metadata', uniqueMetadata[0], '=',  cbm.findIids(metadata = uniqueMetadata[0])
print 'iids with metric', uniqueMetrics[0], '=', cbm.findIids(metricNames = uniqueMetrics[0])
print 'iids with oneD slicer =', cbm.findIids(slicerName = 'OneDSlicer')


print ''

print 'oneD comparisons'
# Find the dict nums with oneD slicers
oneDIids = cbm.findIids(slicerName='OneDSlicer')
# Find the metric names associated with those oneD binmetrics
oneDmetrics = cbm.uniqueMetrics(iids=oneDIids)
# Plot the same metrics on the same plot
for mname in oneDmetrics:
    iids = cbm.findIids(metricNames=mname, slicerName='OneDSlicer')
    cbm.plotHistograms(iids)

plt.show()

