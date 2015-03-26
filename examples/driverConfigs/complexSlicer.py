from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, makeDict

root.outputDir = 'BigHist'
root.dbAddress = {'dbAddress':'sqlite:///ops2_1065_sqlite.db'}
root.opsimName = 'ops2_1065'


sliceList = []
binMin = 0.5
binMax = 60
binsize= 1
metric1 = configureMetric('Tgaps',
                         kwargs={'binMin':binMin,
                                 'binMax':binMax, 'binsize':binsize})
metric15 = configureMetric('Tgaps',
                         kwargs={'binMin':0,
                                 'binMax':60, 'binsize':.05, 'metricName':'TgapsFine'})

binMax = 365.25*5

metric2 = configureMetric('Tgaps',
                         kwargs={'metricName':'AllGaps', 'allGaps':True, 'binMin':binMin,
                                 'binMax':binMax, 'binsize':binsize})

metricDict = makeDict(metric1,metric2, metric15)
slicer = configureSlicer('HealpixComplexSlicer',
                         kwargs={'nside':16},
                         metricDict=metricDict,
                         constraints=['filter = "r"'])

sliceList.append(slicer)

slicer = configureSlicer('HealpixComplexSlicer',
                         kwargs={'nside':64,
                         'spatialkey1':'ditheredRA',
                         'spatialkey2':'ditheredDec'},
                         metricDict=makeDict(metric1,metric2),
                         constraints=['filter = "r"'], metadata='dithered')

sliceList.append(slicer)

# Plot just a single Healpixel.
metric1 = configureMetric('Tgaps',
                         kwargs={'binMin':binMin,
                                 'binMax':binMax, 'binsize':binsize},
                        plotDict={'singleHist':2253})

slicer = configureSlicer('HealpixComplexSlicer',
                         kwargs={'nside':64,
                         'spatialkey1':'ditheredRA',
                         'spatialkey2':'ditheredDec'},
                         metricDict=makeDict(metric1),
                         constraints=['filter = "r"'], metadata='single Hist')

sliceList.append(slicer)





root.slicers = makeDict(*sliceList)


root.verbose = True
