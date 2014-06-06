# Driver for running a declination only dither scheme

from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict
root.outputDir = './DecDith'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1039_sqlite.db', 'OutputTable':'output'}
root.opsimName = 'example'


binList = []

metric = makeMetricConfig('Coaddm5Metric')
binner = makeBinnerConfig('HealpixBinner', metricDict=makeDict(metric),
                          constraints=['filter = "r"'])

binList.append(binner)

metric = makeMetricConfig('Coaddm5Metric', kwargs={'metricName':'m5_decdith'})
binner = makeBinnerConfig('HealpixBinner', metricDict=makeDict(metric),
                          constraints=['filter = "r"'], kwargs={'spatialkey1':'fieldRA', 'spatialkey2':'decOnlyDither'})
binList.append(binner)



root.binners = makeDict(*binList)
