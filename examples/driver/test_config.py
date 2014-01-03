root.outputDir = './output'
root.dbAddress ='mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'
root.opsimNames = ['output_opsim3_61_forLynne']

constraints = ["filter=\'r\'", "filter=\'r\' and night < 51000", "filter=\'i\' "]


root.grid1 ='GlobalGrid'
root.kwrdsForGrid1 = ''
root.metricsForGrid1 =['MeanMetric', 'RmsMetric','MaxMetric', 'MinMetric']
root.metricParamsForGrid1 = ['5sigma_modified','seeing', '5sigma_modified','night' ]
root.metricKwrdsForGrid1 = ['']*4
root.constraintsForGrid1 = constraints


root.grid2 = 'HealpixGrid'
root.kwrdsForGrid2 = 'nside=128'
root.metricsForGrid2 = ['Coaddm5Metric', 'MeanMetric', 'MinMetric']#, 'VisitPairsMetric']
root.metricParamsForGrid2 =['5sigma_modified', 'seeing','night']#, '']
root.metricKwrdsForGrid2 = ['']*3#, 'deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0']
root.constraintsForGrid2 = constraints
