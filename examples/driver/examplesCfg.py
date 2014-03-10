import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './temp'
root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']


binList=[]
nside=128




# Trying out the RadiusObs Metric #

constraints = ["fielddec < 0 and fielddec > -0.34888 and fieldra > 3.4888"]
m1 = makeMetricConfig('RadiusObsMetric')
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA",
                                  'spatialkey2':"fieldDec"},
                          metricDict = makeDict(m1),
                          setupKwargs={"leafsize":50000},
                          constraints=constraints) 

binList.append(binner)

m1 = makeMetricConfig('RadiusObsMetric', kwargs={'metricName':'Hexon', 'racol':'hexdithra','deccol':'hexdithdec'})
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"hexdithra",
                                  'spatialkey2':"hexdithdec"},
                          metricDict = makeDict(m1),
                          setupKwargs={"leafsize":50000},
                          constraints=constraints) 
binList.append(binner)



# Trying out the astrometry metrics



constraints = ["night < 730"]

m1 = makeMetricConfig('ProperMotionMetric')
m2 = makeMetricConfig('ParallaxMetric')
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA",
                                  'spatialkey2':"fieldDec"},
                          metricDict = makeDict(m1,m2),
                          setupKwargs={"leafsize":50000},
                          constraints=constraints) 

binList.append(binner)

#should actually put in a stacker config here to make sure the hexdithra is used

col = ColStackConfig()
col.name = 'ParallaxFactor'
col.kwargs_str = {'raCol':'hexdithra','decCol':'hexdithdec'}

m1 = makeMetricConfig('ProperMotionMetric', kwargs={'metricName':'PM_hex'})
m2 = makeMetricConfig('ParallaxMetric',kwargs={'metricName':'Pi_hex'})

binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"hexdithra",
                                  'spatialkey2':"hexdithdec"},
                          metricDict = makeDict(m1,m2),
                          setupKwargs={"leafsize":50000},
                          constraints=constraints, stackCols=makeDict(col)) 
binList.append(binner)




root.binners=makeDict(*binList)
