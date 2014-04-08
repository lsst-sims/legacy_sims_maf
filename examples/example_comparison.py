import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.binMetrics as binMetrics
import lsst.sims.operations.maf.utils.getData as getData
import glob


cbm = binMetrics.ComparisonBinMetric(verbose=True)
dictnums = []
dictnames = []

# Generate some metric data from two binMetrics (dithered & nondithered)
runBinMetrics = False
if runBinMetrics:
    dbAddress = getData.getDbAddress(connectionName='LYNNE_OPSIM')
    dbTable = 'output_opsimblitz2_1007'
    opsimrun = dbTable.replace('output_', '')
    sqlconstraint = 'filter = "r"'
    seeingcol = 'finSeeing'
    metricList = []
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                        plotParams={'ylog':False, 'title':'Number of visits',
                                                    'plotMin':0, 'plotMax':300, 'cbarFormat': '%d'}))
    metricList.append(metrics.Coaddm5Metric('5sigma_modified', metricName='Coadd_m5',
                                            plotParams={'title':'Coadded m5'}))
    bb1 = binners.HealpixBinner(nside=16, spatialkey1='fieldRA', spatialkey2='fieldDec')
    bb2 = binners.HealpixBinner(nside=16, spatialkey1='hexdithra', spatialkey2='hexdithdec')
    
    datacolnames = list(metricList[0].classRegistry.uniqueCols())
    datacolnames += bb1.columnsNeeded
    datacolnames += bb2.columnsNeeded
    datacolnames = list(set(datacolnames))
    
    print datacolnames
    simdata = getData.fetchSimData(dbTable, dbAddress, sqlconstraint, datacolnames)
    
    print 'setting up binners'
    bb1.setupBinner(simdata, leafsize=1000)
    bb2.setupBinner(simdata, leafsize=1000)
    
    bbm1 = binMetrics.BaseBinMetric()
    bbm1.setBinner(bb1)
    bbm1.setMetrics(metricList)
    bbm1.runBins(simdata, simDataName=opsimrun, metadata='Nondithered')
    bbm2 = binMetrics.BaseBinMetric()
    bbm2.setBinner(bb2)
    bbm2.setMetrics(metricList)
    bbm2.runBins(simdata, simDataName=opsimrun, metadata='Dithered')

    dnum = cbm.setMetricData(bbm1, nametag='Nondithered')
    dictnums.append(dnum)
    dictnames.append('Nondithered')
    
    dnum = cbm.setMetricData(bbm2, nametag='Dithered')
    dictnums.append(dnum)
    dictnames.append('Dithered')

# Read some metric data in from files.
readBinMetrics = True
if readBinMetrics:
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
print 'dicts with oneD binner =', cbm.findDictNums(binnerName = 'OneDBinner')


print ''

print 'oneD comparisons'
# Find the dict nums with oneD binners
oneDDicts = cbm.findDictNums(binnerName='OneDBinner')
# Find the metric names associated with those oneD binmetrics 
oneDmetrics = cbm.uniqueMetrics(dictNums=oneDDicts)
# Plot the same metrics on the same plot
for mname in oneDmetrics:
    dicts = cbm.findDictNums(metricNames=mname, binnerName='OneDBinner')
    metricnames = [mname for d in dicts]
    cbm.plotHistograms(dicts, metricnames)

plt.show()
    
