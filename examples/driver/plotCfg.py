import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './temp'
root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']


binList=[]
nside=64

constraints = ["propID = 188"]

m1 =  makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits / # WFD','units':'# visits / # WFD'}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.})
metricDict = makeDict(m1)
pc1 = makePlotConfig({'units':'wacky!', 'title':"Joint Completeness"} )
pc2 = makePlotConfig({'units':'wacky2!', 'title':r"$i$ band completeness"} )
binner = makeBinnerConfig('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints, plotConfigs={'completeness_I':pc2, 'completeness_Joint':pc1})
binList.append(binner)


binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
binList.append(binner)



root.binners=makeDict(*binList)
