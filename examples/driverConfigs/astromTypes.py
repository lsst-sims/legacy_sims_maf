from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Compare the astrometric precision of different stellar types

root.outputDir = './Astrom'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'
root.verbose = True

nside = 32
slicerList=[]

m1 = configureMetric('ParallaxMetric', kwargs={'SedTemplate':'B',
                                               'metricName':'ParallaxB'})
m2 = configureMetric('ParallaxMetric', kwargs={'SedTemplate':'K',
                                               'metricName':'ParallaxK'})

m3 = configureMetric('ProperMotionMetric', kwargs={'SedTemplate':'K',
                                               'metricName':'ProperMotionK'})
m4 = configureMetric('ProperMotionMetric', kwargs={'SedTemplate':'B',
                                               'metricName':'ProperMotionB'})

metricDict = makeDict(m1,m2,m3,m4)
slicer = configureSlicer('HealpixSlicer', metricDict=metricDict,
                         kwargs={"nside":nside}, constraints=[''])

slicerList.append(slicer)

root.slicers=makeDict(*slicerList)
