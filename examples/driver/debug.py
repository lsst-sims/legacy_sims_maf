import numpy as np
from lsst.sims.maf.driver.mafConfig import *

# A drive config for the plots on https://confluence.lsstcorp.org/display/SIM/MAF+documentation
# and https://confluence.lsstcorp.org/display/SIM/MAF%3A++Writing+a+new+metric

root.outputDir = './Doc'
#root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
root.opsimNames = ['opsim_small']
root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
#root.opsimNames = ['opsim']



binList=[]

m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)
binList.append(binner)


# Example of merging histograms
filters = ['u','g','r','i','z','y']
filter_colors=['m','b','g','y','r','k']
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['airmass'], histMerge={'histNum':1, 'legendloc':'upper right', 'color':filter_colors[i],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},  metricDict=makeDict(m1), constraints=["filter = '%s'"%f])
    binList.append(binner)



root.binners=makeDict(*binList)



