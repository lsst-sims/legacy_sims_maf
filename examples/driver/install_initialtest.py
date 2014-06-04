import numpy as np
from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict

root.outputDir = './InitialTest'
root.dbAddress ={'dbAddress':'sqlite:///opsimblitz2_1039_sqlite.db'}
root.opsimNames = ['Output']


binList=[]

filters = ['r', 'g']
nside = 64

for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisits'}, 
                          plotDict={'plotMin':0, 'plotMax':200, 'units':'N Visits'},
                          summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
    m2 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'}, 
                          plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
    metricDict = makeDict(m1, m2)
    constraint = 'filter = "%s"' %(f)
    binner = makeBinnerConfig('HealpixBinner', kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                              metricDict=metricDict, constraints=[constraint,])
    root.binners=makeDict(binner)
    binList.append(binner)


root.binners=makeDict(*binList)



