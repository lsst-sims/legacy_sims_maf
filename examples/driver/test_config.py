import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import GridPackConfig

root.outputDir = './output'
root.dbAddress ='mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'
root.opsimNames = ['output_opsim3_61_forLynne']

constraints = ["filter=\'r\'", "filter=\'r\' and night < 51000", "filter=\'i\' "]

g = GridPackConfig()
g.grid = 'GlobalGrid'
g.kwrdsForGrid =  ''
g.metricsForGrid = ['MeanMetric', 'RmsMetric','MaxMetric', 'MinMetric']
g.metricParamsForGrid = ['5sigma_modified','seeing', '5sigma_modified','night' ]
g.metricKwrdsForGrid = ['']*4
g.constraintsForGrid = constraints

root.grid1 = g

g = GridPackConfig()
g.grid = 'HealpixGrid'
g.kwrdsForGrid = 'nside=128'
g.metricsForGrid = ['Coaddm5Metric', 'MeanMetric', 'MinMetric']#, 'VisitPairsMetric']
g.metricParamsForGrid =['5sigma_modified', 'seeing','night']#, '']
g.metricKwrdsForGrid = ['']*3#, 'deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0']
g.constraintsForGrid = constraints

root.grid2 = g

