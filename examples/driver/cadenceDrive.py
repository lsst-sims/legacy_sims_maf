# Test out new cadence metrics
from lsst.sims.maf.driver.mafConfig import *

root.outputDir ='./Cadence'


small = True # Use the small database included in the repo

if small:
    root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
    root.opsimNames = ['opsim_small']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?
else:
    root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
    root.opsimNames = ['opsim']
    propids = [215, 216, 217, 218, 219]
    WFDpropid = 217
    DDpropid = 219

filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}

binList=[]
nside=64

seeing_limit = 0.7 # Demand seeing better than this



# Example looking for the existence of a quality refernce image in each filter after 1 year, 2 years and 10 years
for f in filters:
    m1 = makeMetricConfig('BinaryMetric', params=['finSeeing'], summaryStats={'SumMetric':{}})
    binner = makeBinnerConfig('HealpixBinner',
                              kwargs={"nside":nside},
                              metricDict=makeDict(m1),
                              constraints=['night < 365 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),'night < 730 and filter = "%s" and finSeeing < %s'%(f,seeing_limit), 'filter = "%s" and finSeeing < %s'%(f,seeing_limit)],
                              setupKwargs={"leafsize":50000})
    binList.append(binner)


m1 = makeMetricConfig('SupernovaMetric', kwargs={'m5col':'5sigma_modified'})
binner =  makeBinnerConfig('HealpixBinner',
                              kwargs={"nside":nside},
                              metricDict=makeDict(m1), constraints=['night < 730'], setupKwargs={"leafsize":50000})
binList.append(binner)
    
root.binners = makeDict(*binList)

