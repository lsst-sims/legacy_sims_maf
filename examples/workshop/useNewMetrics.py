# To use a new metric, make sure the path to the code is in your
#PYTHONPATH environement variable.  For example:
#setenv PYTHONPATH $PYTHONPATH':/some/path/here/'

from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict

root.outputDir = 'OutMetrics'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

root.modules = ['exampleNewMetrics']

sliceList = []

metric = configureMetric('exampleNewMetrics.SimplePercentileMetric', kwargs={'col':'airmass'})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.PercentileMetric', kwargs={'col':'airmass', 'percentile':75})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.DifferenceMetric', kwargs={'colA':'fieldRA', 'colB':'hexdithRA'})
slicer = configureSlicer('OpsimFieldSlicer', metricDict=makeDict(metric), constraints=[''])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.DifferenceMetric', kwargs={'colA':'fieldDec', 'colB':'hexdithDec'})
slicer = configureSlicer('OpsimFieldSlicer', metricDict=makeDict(metric), constraints=[''])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.BestSeeingCoaddedDepth')
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=[''])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric')
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=[''])
sliceList.append(slicer)

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric')
slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night', 'binsize':365},
                         metricDict=makeDict(metric), constraints=[''])
sliceList.append(slicer)

root.slicers = makeDict(*sliceList)


