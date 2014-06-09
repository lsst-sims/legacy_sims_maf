from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict

root.outputDir = './AGN'
root.dbAddress ={'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
root.opsimName = 'opsimblitz1_1131'
 
m1 = makeMetricConfig('LongGapAGNMetric')
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=makeDict(m1), constraints=[''] )
root.binners=makeDict(binner)

