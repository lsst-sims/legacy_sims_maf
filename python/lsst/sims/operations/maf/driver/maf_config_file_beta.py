import itertools

root.outputDir = 'path/to/somwhere'
root.dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST'
root.opsimNames = ['opsim_3_61', 'opsim_happyrun', 'opsim_sadrun']

constraints = []
cons = itertools.product(['filter == '+s+' & ' for s in ['"u"','"g"','"r"','"i"','"z"','"y"']],
                         ['proposalID == ' + s+' & ' for s in ['"WFD"', '"other"']],
                         ['seeing ' +s for s in ['< 0.5','>=0.5','<1000']] )
for c in cons:
    constraints.append(''.join(c) ) 

root.grid1 ='GlobalGrid'
root.kwrdsForGrid1 = ''
root.metricsForGrid1 =['MeanMetric', 'RmsMetric','MaxMetric']
root.metricParamsForGrid1 = ['5sigma_modified','seeing', '5sigma_modified' ]
root.metricKwrdsForGrid1 = ['']*3
root.constraintsForGrid1 = constraints

root.grid2 = 'HealpixGrid'
root.kwrdsForGrid2 = 'nside=256'
root.metricsForGrid2 = ['Coaddm5Metric', 'VisitPairsMetric']
root.metricParamsForGrid2 =['5sigma_modified', '']
root.metricKwrdsForGrid2 = ['', 'deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0']
root.constraintsForGrid2 = constraints

root.grid3 = 'HealpixGrid'
root.kwrdsForGrid3 = 'nside=512'
root.metricsForGrid3 = ['ProperMotionMetric']
root.metricParamsForGrid3 = ['']
root.constraintsForGrid3 = ' | '.join(['(filter == ' + s +')' for s in ['g','r','i']])


