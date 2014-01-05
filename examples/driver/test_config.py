import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import GridPackConfig 

root.outputDir = './output'
root.dbAddress ='mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'
root.opsimNames = ['output_opsim3_61_forLynne']

constraints = ["filter=\'r\'", "filter=\'r\' and night < 51000", "filter=\'i\' "]

g = GridPackConfig()
g.grid = 'GlobalGrid'
g.kwrds =  ''
g.metrics = ['MeanMetric', 'RmsMetric','MaxMetric', 'MinMetric']
g.metricParams = ['5sigma_modified','seeing', '5sigma_modified','night' ]
g.metricKwrds = ['']*4
g.constraints = constraints

root.grid1 = g

g = GridPackConfig()
g.grid = 'HealpixGrid'
g.kwrds = 'nside=128'
g.metrics = ['Coaddm5Metric', 'MeanMetric', 'MinMetric']#, 'VisitPairsMetric']
g.metricParams =['5sigma_modified', 'seeing','night']#, '']
g.metricKwrds = ['']*3#, 'deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0']
g.constraints = constraints

root.grid2 = g

