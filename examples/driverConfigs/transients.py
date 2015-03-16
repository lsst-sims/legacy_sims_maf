import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.outputDir = './Transients'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

root.getConfig = False

slicerList=[]

nside=32
metricList = []
metricList.append(configureMetric('TransientMetric', kwargs={'metricName':'Detect Tophat'}))

metricList.append(configureMetric('TransientMetric',
                     kwargs={'riseSlope':-1., 'declineSlope':1.,
                             'metricName':'Detect w/slope'}) )

# Demand at least 1 filter sample the lightcurve at 4 well-spaced points
metricList.append( configureMetric('TransientMetric',
                     kwargs={'riseSlope':-1., 'declineSlope':1.,
                             'nPerLC':4 ,
                             'metricName':'fourptsPerLC'}))
# Demand at least 2 filters sample the lightcurve at 4 well-spaced points
metricList.append( configureMetric('TransientMetric',
                     kwargs={'riseSlope':-1., 'declineSlope':1.,
                             'nPerLC':4 , 'nFilters':2,
                             'metricName':'fourptsPerLC2Filt'}))



metricDict = makeDict(*metricList)
slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside},
                         metricDict=metricDict, constraints=[''])
slicerList.append(slicer)


root.slicers=makeDict(*slicerList)
