from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict, configureMap

root.dbAddress ={'dbAddress':'sqlite:///'+'opsimblitz2_1060_sqlite.db'}   
root.outputDir = 'MapsDir'
root.verbose = True
slicerList=[]


nside=128
lsstFilter='r'

constraints=["filter = \'%s\'"%lsstFilter]
m1 = configureMetric('ExgalM5', kwargs={'lsstFilter':lsstFilter}, plotDict={'xMin':20, 'xMax':30})

# Note that we need to set useCache to False to ensure we calculate the metric at every point.
slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside, 'useCache':False},
                         metricDict=makeDict(m1),  constraints=constraints)

slicerList.append(slicer)

slicer = configureSlicer('OpsimFieldSlicer',
                         metricDict=makeDict(m1),  constraints=constraints)

slicerList.append(slicer)


nside=32
lsstFilter='r'
constraints=["filter = \'%s\'"%lsstFilter]
m1 = configureMetric('ExgalM5', kwargs={'lsstFilter':lsstFilter}, plotDict={'xMin':20, 'xMax':30})
mapConfig = configureMap('DustMap', kwargs={"nside":nside})
slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside, 'useCache':False},
                         metricDict=makeDict(m1),  mapsDict=makeDict(mapConfig) ,
                         constraints=constraints, metadata='nside32')

slicerList.append(slicer)




root.slicers=makeDict(*slicerList)
