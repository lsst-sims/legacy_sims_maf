from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict



root.outputDir = 'Hastacker'
root.dbAddress = {'dbAddress':'sqlite:///ops2_1065_sqlite.db'}
root.opsimName = 'ops2_1065'

sliceList = []

metric1 = configureMetric('MeanMetric', kwargs={'col':'HA'})
metric2 = configureMetric('MinMetric', kwargs={'col':'HA'})
metric3 = configureMetric('MaxMetric', kwargs={'col':'HA'})

metricDict=makeDict(metric1,metric2,metric3)
slicer=configureSlicer('HealpixSlicer', metricDict=metricDict, constraints=['filter = "r"'])

sliceList.append(slicer)
root.slicers = makeDict(*sliceList)

root.verbose = True
