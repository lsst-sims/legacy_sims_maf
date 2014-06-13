import numpy as np
from lsst.sims.maf.driver.mafConfig import configureBinner, configureMetric, makeDict
# A drive config for the plots on https://confluence.lsstcorp.org/display/SIM/MAF+documentation
# and https://confluence.lsstcorp.org/display/SIM/MAF%3A++Writing+a+new+metric

root.outputDir = './Doc'
root.dbAddress = {'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
root.opsimName = 'opsim'



binList=[]

m1 = configureMetric('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)
binList.append(binner)

constraints = ["filter = \'%s\'"%'r']
m1 = configureMetric('MinMetric', params=['airmass'], plotDict={'cmap':'RdBu'})
metricDict = makeDict(m1)
binner = configureBinner('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
root.binners=makeDict(binner)
binList.append(binner)

nside=64
constraints = ["filter = \'%s\'"%'r']
m2 = configureMetric('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.} )          
metricDict = makeDict(m2)
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
root.binners=makeDict(binner)
binList.append(binner)

constraints = ["filter = \'%s\'"%'r']
m3 = configureMetric('Coaddm5Metric')
m4 = configureMetric('MeanMetric', params=['normairmass'])
metricDict = makeDict(m3,m4)
binner = configureBinner('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
root.binners=makeDict(binner)
binList.append(binner)


m1 = configureMetric('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)
binList.append(binner)


m1=configureMetric('HourglassMetric')
binner = configureBinner('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750'] )
binList.append(binner)



# How many Healpix sides to use
nside=64
# List of SQL constraints.  If multiple constraints are listed in a binner object, they are looped over and each one is executed individualy. 
constraints = ["filter = \'%s\'"%'r']
m1 = configureMetric('RmsMetric', params=['finSeeing'], plotDict={'plotMin':0., 'plotMax':0.6})
m2 =  configureMetric('RobustRmsMetric', params=['finSeeing'], plotDict={'plotMin':0., 'plotMax':0.6})
metricDict = makeDict(m1,m2)
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)



m1 = configureMetric('AstroPrecMetric')
m2 = configureMetric('AstroPrecMetricComplex')
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = makeDict(m1,m2),setupKwargs={"leafsize":50000},constraints=constraints)
binList.append(binner)



# Example of doing summary stats:
nside=64
constraints = ["filter = \'%s\'"%'r']
m2 = configureMetric('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.}, kwargs={'metricName':'coadd_w_summary'}, summaryStats={'MeanMetric':{}, 'MinMetric':{}, 'MaxMetric':{}, 'RmsMetric':{}} )          
metricDict = makeDict(m2
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
root.binners=makeDict(binner)
binList.append(binner)



# XXX-summary stats are not intuitive with OneDbinner.  
#m1 = configureMetric('CountMetric', params=['slewTime'], kwargs={'metadata':'time', 'metricName':'slew_w_summary'},summaryStats={'MeanMetric':{}, 'MedianMetric':{}})
#binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=['']  )
#root.binners=makeDict(binner)
#binList.append(binner)


# Example of merging histograms
filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
for i,f in enumerate(filters):
    m1 = configureMetric('CountMetric', params=['airmass'], histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'airmass'},  metricDict=makeDict(m1), constraints=["filter = '%s'"%f])
    binList.append(binner)



root.binners=makeDict(*binList)



