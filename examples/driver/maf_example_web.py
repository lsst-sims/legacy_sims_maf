import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# A drive config for the plots on https://confluence.lsstcorp.org/display/SIM/MAF+documentation

root.outputDir = './Doc'
root.dbAddress ='sqlite:///opsim.sqlite'
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


root.binners=makeDict(*binList)
