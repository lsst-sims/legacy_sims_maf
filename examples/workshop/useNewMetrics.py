from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict

root.outputDir = 'Out'
root.dbAddress = {'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
root.opsimName = 'opsimblitz1_1131'

root.modules = ['exampleNewMetrics']

metric = configureMetric('exampleNewMetrics.SimplePercentileMetric', args=['airmass'])
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

root.slicers = makeDict(slicer)
