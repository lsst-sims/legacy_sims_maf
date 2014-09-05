from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict



root.outputDir = 'Hastacker'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

sliceList = []

metric1 = configureMetric('MeanMetric', kwargs={'col':'HA'})
metric2 = configureMetric('MinMetric', kwargs={'col':'HA'})
metric3 = configureMetric('MaxMetric', kwargs={'col':'HA'})
metric4 = configureMetric('MedianMetric', kwargs={'col':'HA'})
metric5 = configureMetric('MedianMetric', kwargs={'col':'normairmass'})
metric6 =  configureMetric('MedianAbsMetric', kwargs={'col':'HA'})

metricDict=makeDict(metric1,metric2,metric3,metric4, metric5,metric6)
slicer=configureSlicer('HealpixSlicer', metricDict=metricDict, constraints=['filter = "r"', 'filter="g"'])

sliceList.append(slicer)
root.slicers = makeDict(*sliceList)

root.verbose = True
