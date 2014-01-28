import itertools
from mafConfig import *

root.outputDir = 'path/to/somwhere'
root.dbAddress = 'mysql://lsst:lsst@blahblah/'
root.opsimNames = ['opsim_3_61', 'opsim_happyrun', 'opsim_sadrun']

constraints = []
cons = itertools.product(['filter == '+s+' & ' for s in ['"u"','"g"','"r"','"i"','"z"','"y"']],
                         ['proposalID == ' + s+' & ' for s in ['"WFD"', '"other"']],
                         ['seeing ' +s for s in ['< 0.5','>=0.5','<1000']] )
for c in cons:
    constraints.append(''.join(c) ) 


root.constraints = constraints

binner1 = BinnerConfig()
binner1.binner = 'HealpixBinner'
m1 = makeMetricConfig('MeanMetric', params='5sigma_modified')
m2 = makeMetricConfig('RmsMetric', params='seeing')
binner1.metricDict = makeDict(m1,m2 )

binner2 = BinnerConfig()
binner2.binner = 'UniBinner'
m1 = makeMetricConfig('MeanMetric', params='5sigma_modified')
m2 = makeMetricConfig('RmsMetric', params='seeing')
binner2.metricDict = makeDict(m1,m2 )

root.binners = makeDict(binner1,binner2)

