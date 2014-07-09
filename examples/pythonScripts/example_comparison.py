import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.db as db
import glob


cbm = sliceMetrics.ComparisonSliceMetric(verbose=True)
dictnums = []
dictnames = []

# Generate some metric data from two sliceMetrics (dithered & nondithered)
#
runSliceMetrics = False
if runSliceMetrics:
    opsimrun = 'opsimblitz1_1131'
    sqlitepath = '../../tests/opsimblitz1_1131_sqlite.db'
    dbAddress = 'sqlite:///' + sqlitepath
    opsimdb = db.OpsimDatabase(dbAddress)

    sqlconstraint = 'filter = "r"'
    seeingcol = 'finSeeing'
    metricList = []
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                        plotParams={'ylog':False, 'title':'Number of visits',
                                                    'plotMin':0, 'plotMax':300, 'cbarFormat': '%d'}))
    metricList.append(metrics.Coaddm5Metric('fivesigma_modified', metricName='Coadd_m5',
                                            plotParams={'title':'Coadded m5'}))
    bb1 = slicers.HealpixSlicer(nside=16, spatialkey1='fieldRA', spatialkey2='fieldDec')
    bb2 = slicers.HealpixSlicer(nside=16, spatialkey1='hexdithra', spatialkey2='hexdithdec')
    
    datacolnames = list(metricList[0].classRegistry.uniqueCols())
    datacolnames += bb1.columnsNeeded
    datacolnames += bb2.columnsNeeded
    datacolnames = list(set(datacolnames))
    
    print datacolnames
    simdata = opsimdb.fetchMetricData(datacolnames, sqlconstraint)
    
    print 'setting up slicers'
    bb1.setupSlicer(simdata)
    bb2.setupSlicer(simdata)
    
    bbm1 = sliceMetrics.BaseSliceMetric()
    bbm1.setSlicer(bb1)
    bbm1.setMetrics(metricList)
    bbm1.runSlices(simdata, simDataName=opsimrun, sqlconstraint=sqlconstraint, metadata='Nondithered')
    bbm2 = sliceMetrics.BaseSliceMetric()
    bbm2.setSlicer(bb2)
    bbm2.setMetrics(metricList)
    bbm2.runSlices(simdata, simDataName=opsimrun, sqlconstraint=sqlconstraint, metadata='Dithered')

    dnum = cbm.setMetricData(bbm1, nametag='Nondithered')
    dictnums.append(dnum)
    dictnames.append('Nondithered')
    
    dnum = cbm.setMetricData(bbm2, nametag='Dithered')
    dictnums.append(dnum)
    dictnames.append('Dithered')

# Read some metric data in from files.
readSliceMetrics = True
if readSliceMetrics:
    filenames = glob.glob('*.npz')
    dictnums = []
    filelist = []
    for f in filenames:
        if 'VisitPairs' not in f:
            print 'working on %s' %(f)
            dictnum = cbm.readMetricData(f)
            dictnums.append(dictnum)
            dictnames.append(f)

####            
for num, name in zip(dictnums, dictnames):
    print num, name

        
uniqueMetrics = list(cbm.uniqueMetrics())
uniqueMetadata = list(cbm.uniqueMetadata())
uniqueSimDataNames = list(cbm.uniqueSimDataNames())
print (uniqueSimDataNames)
print (uniqueMetadata)
print (uniqueMetrics)

print ''

print 'all dicts: ', cbm.findDictNums()
print 'dicts with simname', uniqueSimDataNames[0], '=', cbm.findDictNums(simDataName = uniqueSimDataNames[0])
print 'dicts with metadata', uniqueMetadata[0], '=',  cbm.findDictNums(metadata = uniqueMetadata[0])
print 'dicts with metric', uniqueMetrics[0], '=', cbm.findDictNums(metricNames = uniqueMetrics[0])
print 'dicts with oneD slicer =', cbm.findDictNums(slicerName = 'OneDSlicer')


print ''

print 'oneD comparisons'
# Find the dict nums with oneD slicers
oneDDicts = cbm.findDictNums(slicerName='OneDSlicer')
# Find the metric names associated with those oneD binmetrics 
oneDmetrics = cbm.uniqueMetrics(dictNums=oneDDicts)
# Plot the same metrics on the same plot
for mname in oneDmetrics:
    dicts = cbm.findDictNums(metricNames=mname, slicerName='OneDSlicer')
    metricnames = [mname for d in dicts]
    cbm.plotHistograms(dicts, metricnames)

plt.show()
    
