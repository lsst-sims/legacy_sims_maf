from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict

root.outputDir = 'Out'
root.dbAddress = {'dbAddress':'sqlite:///../tier1/opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

root.modules = ['exampleNewMetrics']

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric')
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=[''])

metric = configureMetric('exampleNewMetrics.SimplePercentileMetric', params=['airmass'])
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

metric = configureMetric('exampleNewMetrics.PercentileMetric', params=['airmass'],
                         kwargs={'percentile':75})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

root.slicers = makeDict(slicer)


