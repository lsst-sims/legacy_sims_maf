# Test out new cadence metrics
from lsst.sims.maf.driver.mafConfig import *

root.outputDir ='./Cadence'


small = False # Use the small database included in the repo

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
#filters=['r']

binList=[]
nside=128
leafsize = 50000 # For KD-tree

seeing_limit = 0.7 # Demand seeing better than this



# Example looking for the existence of a quality refernce image in each filter after 1 year, 2 years and 10 years
for f in filters:
    m1 = makeMetricConfig('BinaryMetric', params=['finSeeing'], summaryStats={'SumMetric':{}})
    binner = makeBinnerConfig('HealpixBinner',kwargs={"nside":nside},metricDict=makeDict(m1),
                              constraints=['night < 365 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'night < 730 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'filter = "%s" and finSeeing < %s'%(f,seeing_limit)],
                              setupKwargs={"leafsize":leafsize})
    binList.append(binner)

# Look at the minimum seeing per field, and the fraction of observations below the "good" limit
for f in filters:
    m1 = makeMetricConfig('TemplateExistsMetric')
    m2 = makeMetricConfig('MinMetric', params=['finSeeing'])
    m3 = makeMetricConfig('FracBelowMetric', params=['finSeeing'], kwargs={'cutoff':seeing_limit})
    binner = makeBinnerConfig('HealpixBinner',kwargs={"nside":nside},metricDict=makeDict(m1,m2,m3),
                              constraints=['night < 365 and filter = "%s"'%(f),
                                           'night < 730 and filter = "%s"'%(f),
                                           'filter = "%s"'%(f)],
                              setupKwargs={"leafsize":leafsize})
    binList.append(binner)

   
m1 = makeMetricConfig('SupernovaMetric', kwargs={'m5col':'5sigma_modified', 'redshift':0.1, 'resolution':5.}, plotDict={'percentileClip':95.})
binner =  makeBinnerConfig('HealpixBinner', kwargs={"nside":nside},
                              metricDict=makeDict(m1), constraints=[''], setupKwargs={"leafsize":50000})
binList.append(binner)
    
root.binners = makeDict(*binList)

