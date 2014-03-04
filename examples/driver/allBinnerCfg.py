import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './Allbinners'
root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']


binList=[]
nside=64

constraints = ["filter = \'%s\'"%'r']


m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'percentileClip':80., 'units':'#'})
m2 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.} )           
metricDict = makeDict(m1,m2)
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)

m1 = makeMetricConfig('CountMetric', params=['slewDist'])
metricDict=makeDict(m1)
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'},
                          metricDict=metricDict, constraints=constraints)
binList.append(binner)


m1 = makeMetricConfig('MinMetric', params=['airmass'], plotDict={'cmap':'RdBu'})
m4 = makeMetricConfig('MeanMetric', params=['normairmass'])
m3 = makeMetricConfig('Coaddm5Metric')
m7 = makeMetricConfig('CountMetric', params=['expMJD'], plotDict={'units':"#", 'percentileClip':80.})
metricDict = makeDict(m1,m3,m4,m7)
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
binList.append(binner)



m1 = makeMetricConfig('SummaryStatsMetric')
binner = makeBinnerConfig('UniBinner', metricDict=makeDict(m1), constraints=['night < 750'] )
binList.append(binner)

m1=makeMetricConfig('HourglassMetric')
binner = makeBinnerConfig('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750',''])
binList.append(binner)



root.binners=makeDict(*binList)
