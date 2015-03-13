import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.outputDir = './Transients'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'

root.getConfig = False

slicerList=[]

nside=32

m1 = configureMetric('TransientDetectMetric', kwargs={'metricName':'Detect Tophat'})

m2 = configureMetric('TransientDetectMetric',
                     kwargs={'riseSlope':-1., 'declineSlope':1.,
                             'metricName':'Detect w/slope'})


metricDict = makeDict(m1,m2)
slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside},
                         metricDict=metricDict, constraints=[''])
slicerList.append(slicer)


root.slicers=makeDict(*slicerList)
