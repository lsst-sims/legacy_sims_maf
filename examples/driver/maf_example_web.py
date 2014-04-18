import numpy as np
from lsst.sims.maf.driver.mafConfig import *

# A drive config for the plots on https://confluence.lsstcorp.org/display/SIM/MAF+documentation
# and https://confluence.lsstcorp.org/display/SIM/MAF%3A++Writing+a+new+metric

root.outputDir = './Doc'
root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
#root.opsimNames = ['opsim_small']
#root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']



binList=[]

m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)
binList.append(binner)

constraints = ["filter = \'%s\'"%'r']
m1 = makeMetricConfig('MinMetric', params=['airmass'], plotDict={'cmap':'RdBu'})
metricDict = makeDict(m1)
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
root.binners=makeDict(binner)
binList.append(binner)

nside=64
constraints = ["filter = \'%s\'"%'r']
m2 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.} )          
metricDict = makeDict(m2)
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
root.binners=makeDict(binner)
binList.append(binner)

constraints = ["filter = \'%s\'"%'r']
m3 = makeMetricConfig('Coaddm5Metric')
m4 = makeMetricConfig('MeanMetric', params=['normairmass'])
metricDict = makeDict(m3,m4)
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
root.binners=makeDict(binner)
binList.append(binner)


m1 = makeMetricConfig('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)
binList.append(binner)


m1=makeMetricConfig('HourglassMetric')
binner = makeBinnerConfig('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750'] )
binList.append(binner)







# How many Healpix sides to use
nside=64
# List of SQL constraints.  If multiple constraints are listed in a binner object, they are looped over and each one is executed individualy. 
constraints = ["filter = \'%s\'"%'r']
m1 = makeMetricConfig('RmsMetric', params=['finSeeing'], plotDict={'plotMin':0., 'plotMax':0.6})
m2 =  makeMetricConfig('RobustRmsMetric', params=['finSeeing'], plotDict={'plotMin':0., 'plotMax':0.6})
metricDict = makeDict(m1,m2)
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)



m1 = makeMetricConfig('AstroPrecMetric')
m2 = makeMetricConfig('AstroPrecMetricComplex')
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = makeDict(m1,m2),setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)



# Let's show how the plotting dictionary works:



root.binners=makeDict(*binList)

m3 = makeMetricConfig('Coaddm5Metric',kwargs={'metricName':'PlotExamples1'}, plotDict={'zp':27., 'percentileClip':95, 'units':'units', 'title':'title'})
metricDict = makeDict(m3)
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)


m2 = makeMetricConfig('CountMetric', kwargs={'metricName':'plotExample2'},plotDict={'untis':'units', 'title':'title', 'legendLabel':'legendLabel', 'xlabel':'xlabel', 'ylabel':'ylabel'}, params=['slewDist'])
metricDict=makeDict(m2)
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'},
                          metricDict=metricDict, constraints=constraints)
binList.append(binner)


