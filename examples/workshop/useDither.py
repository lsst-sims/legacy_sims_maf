from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, configureStacker, makeDict

root.outputDir = 'Out2'
root.dbAddress = {'dbAddress':'sqlite:///../tier1/opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

sliceList = []
nside = 8

# "Normal" configuration of HealpixSlicer
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
#sliceList.append(slicer)

# Normal configuration, making defaults explicit
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fieldRA',
                                                  'spatialkey2':'fieldDec'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
# (Not going to add this to slicerlist as it's a duplicate of above)

# Configuring HealpixSlicer to use hexdither RA/dec
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'hexdithRA',
                                                  'spatialkey2':'hexdithDec'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
#sliceList.append(slicer)


# Use a new Stacker that does not require configuration
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'randomRADither',
                                                  'spatialkey2':'randomDecDither'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
#sliceList.append(slicer)

# Use a new Stacker with configuration
metric = configureMetric('Coaddm5Metric')
stacker = configureStacker('RandomDitherStacker', kwargs={'randomSeed':42})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'randomRADither',
                                                  'spatialkey2':'randomDecDither'},
                         metricDict=makeDict(metric), constraints=['filter="r"'],
                         stackCols=makeDict(stacker))
#sliceList.append(slicer)


# Use our new stacker 
root.modules = ['exampleNewStacker']

# Use our new stacker without configuration
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fixedRA',
                                                  'spatialkey2':'fixedDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'])
#sliceList.append(slicer)

# Use our new stack with configuration
metric = configureMetric('Coaddm5Metric')
stacker = configureStacker('exampleNewStacker.SingleFieldDitherStacker', kwargs={'fixedRA':0.16,
                                                                                 'fixedDec':-0.5})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fixedRA',
                                                  'spatialkey2':'fixedDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'],
                            stackCols=makeDict(stacker))
sliceList.append(slicer)

# Use our new stacker 
root.modules = ['exampleNewStacker']

# Use our new stacker (using defaults)
metric = configureMetric('Coaddm5Metric')
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'yearlyDitherRA',
                                                  'spatialkey2':'yearlyDitherDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)


root.slicers = makeDict(*sliceList)
