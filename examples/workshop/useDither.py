# To use a new stacker, make sure the path to the code is in your
#PYTHONPATH environement variable.  For example:
#setenv PYTHONPATH $PYTHONPATH':/some/path/here/'

from lsst.sims.maf.driver.mafConfig import configureMetric, configureSlicer, configureStacker, makeDict


root.outputDir = 'OutDither'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

sliceList = []
nside = 64

# "Normal" configuration of HealpixSlicer
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'normalDither'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)

# Normal configuration, making defaults explicit
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'normalAgain'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fieldRA',
                                                  'spatialkey2':'fieldDec'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
# (Not going to add this to slicerlist as it's a duplicate of above)

# Configuring HealpixSlicer to use hexdither RA/dec
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'hexdither'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'ditheredRA',
                                                  'spatialkey2':'ditheredDec'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)


# Use a new Stacker that does not require configuration
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'randomDither'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'randomRADither',
                                                  'spatialkey2':'randomDecDither'},
                         metricDict=makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)

# Use a new Stacker with configuration
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'randomConfigured'})
stacker = configureStacker('RandomDitherStacker', kwargs={'randomSeed':42})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'randomRADither',
                                                  'spatialkey2':'randomDecDither'},
                         metricDict=makeDict(metric), constraints=['filter="r"'],
                         stackCols=makeDict(stacker))
sliceList.append(slicer)


# Use our new stacker 
root.modules = ['exampleNewStacker']

# Use our new stacker 
root.modules = ['exampleNewStacker']

# Use our new stacker without configuration
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'coadd_singleField'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fixedRA',
                                                  'spatialkey2':'fixedDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)

# Use our new stacker with configuration
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'coadd_singleFieldConfig'})
stacker = configureStacker('SingleFieldDitherStacker', kwargs={'fixedRA':0.16,
                                                                                 'fixedDec':-0.5})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'fixedRA',
                                                  'spatialkey2':'fixedDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'],
                            stackCols=makeDict(stacker))
sliceList.append(slicer)


# Use our new stacker (using defaults)
metric = configureMetric('Coaddm5Metric', kwargs={'metricName':'coadd_yearly'})
slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside,
                                                  'spatialkey1':'yearlyDitherRA',
                                                  'spatialkey2':'yearlyDitherDec'},
                            metricDict = makeDict(metric), constraints=['filter="r"'])
sliceList.append(slicer)


root.slicers = makeDict(*sliceList)
