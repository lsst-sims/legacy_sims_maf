import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './Allbinners'
root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']


binList=[]
nside=64

constraints = ["filter = \'%s\'"%'r']

binner = BinnerConfig()
binner.name = 'HealpixBinner'
binner.kwargs = {"nside":nside}
m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'percentileClip':80., 'units':'#'})
m2 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.} )           
binner.metricDict = makeDict(m1,m2)
binner.setupKwargs_float={ "leafsize":50000}
binner.setupParams=["fieldRA","fieldDec"]
binner.constraints=constraints
binList.append(binner)


binner= BinnerConfig()
binner.name='OneDBinner'
binner.setupParams=['slewDist']
m1 = makeMetricConfig('CountMetric', params=['slewDist'])
binner.metricDict=makeDict(m1)
binner.constraints=constraints
binList.append(binner)

binner=BinnerConfig()
binner.name='OpsimFieldBinner'
binner.setupParams=["fieldID","fieldRA","fieldDec"]
binner.constraints = constraints
m1 = makeMetricConfig('MinMetric', params=['airmass'], plotDict={'cmap':'RdBu'})
m4 = makeMetricConfig('MeanMetric', params=['normairmass'])
m3 = makeMetricConfig('Coaddm5Metric')
m7 = makeMetricConfig('CountMetric', params=['expMJD'], plotDict={'units':"#", 'percentileClip':80.})
binner.metricDict = makeDict(m1,m3,m4,m7)
binList.append(binner)

binner= BinnerConfig()
binner.name='UniBinner'
m1 = makeMetricConfig('SummaryStatsMetric')
binner.metricDict=makeDict(m1)
binner.constraints=['night < 750']
binList.append(binner)



root.binners=makeDict(*binList)
