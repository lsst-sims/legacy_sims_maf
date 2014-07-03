# Driver for running a declination only dither scheme

from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
root.outputDir = './DecDith'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'Example'


sliceList = []

metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric),
                          constraints=['filter = "r"'])

sliceList.append(slicer)

metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'m5_decdith'})
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric),
                          constraints=['filter = "r"'], kwargs={'spatialkey1':'fieldRA', 'spatialkey2':'decOnlyDither'})
sliceList.append(slicer)



root.slicers = makeDict(*sliceList)
