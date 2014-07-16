# To use a new metric, make sure the path to the code is in your
#PYTHONPATH environement variable.  For example:
#setenv PYTHONPATH $PYTHONPATH':/some/path/here/'

from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict

root.outputDir = 'OutMetrics'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

root.modules = ['exampleNewMetrics']

slicerList = []

metric = configureMetric('exampleNewMetrics.SimplePercentileMetric', kwargs={'col':'airmass',
                                                                             'metricName':'95th Percentile Airmass'})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.SimplePercentileMetric', kwargs={'col':'airmass',
                                                                             'metricName':'95th Percentile Airmass'})
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.PercentileMetric', kwargs={'col':'airmass', 'percentile':75,
                                                                       'metricName':'75th percentile Airmass'})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.PercentileMetric', kwargs={'col':'airmass', 'percentile':75,
                                                                       'metricName':'75th Percentile Airmass'})
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=['filter="r"'])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.MaxDifferenceMetric', kwargs={'colA':'fieldRA', 'colB':'hexdithra'})
slicer = configureSlicer('OpsimFieldSlicer', metricDict=makeDict(metric), constraints=[''])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.MaxDifferenceMetric', kwargs={'colA':'fieldDec', 'colB':'hexdithdec'},
                         plotDict={'cbarFormat':'%.3f'})
slicer = configureSlicer('OpsimFieldSlicer', metricDict=makeDict(metric), constraints=[''])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.BestSeeingCoaddedDepthMetric', kwargs={'metricName':
                                                                                   'Best Seeing Coadded Depth'})
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=[''])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric', kwargs={'nFilters':3,
                                                                               'metricName':
                                                                               'Nights with >3 filters'},
                        plotDict={'cbarFormat':'%d', 'xMin':0, 'xMax':300,'colorMin':100, 'colorMax':300})
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric), constraints=[''])

slicerList.append(slicer)

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric', kwargs={'nFilters':5,
                                                                               'metricName':
                                                                               'Nights with >=5 filters'})
slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night', 'binsize':30},
                         metricDict=makeDict(metric), constraints=[''])

metric = configureMetric('exampleNewMetrics.NightsWithNFiltersMetric', kwargs={'nFilters':5,
                                                                               'metricName':
                                                                               'Nights with >=5 filters'})
slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night', 'binsize':30},
                         metricDict=makeDict(metric), constraints=['propID!=210'])


slicerList.append(slicer)

root.slicers = makeDict(*slicerList)


