from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import os
import numpy as np

dbDir = '.'
runName = 'ops1_1140'
sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
root.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
root.opsimName = runName
root.outputDir = 'Chips'


nside = 128
band='r'
slicerList = []


nside = 2048

m1 = configureMetric('CountMetric', kwargs={'col':'expMJD'})
slicer = configureSlicer('HealpixSlicer',
                       kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec',
                               'useCamera':True},
                       metricDict=makeDict(*[m1]),
                       constraints=['filter="%s" and obsHistID = %i '%(band, 140704),
                                    'filter="%s" and obsHistID = %i '%(band, 174499),
                                    'filter="%s" and obsHistID = %i '%(band, 215847 )])
slicerList.append(slicer)

nside = 128
m1 = configureMetric('CountMetric', kwargs={'col':'expMJD'})
slicer = configureSlicer('HealpixSlicer',
                       kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec',
                               'useCamera':False},
                       metricDict=makeDict(*[m1]),
                       constraints=['filter="%s" and obsHistID = %i '%(band, 140704),
                                    'filter="%s" and obsHistID = %i '%(band, 174499),
                                    'filter="%s" and obsHistID = %i '%(band, 215847 )], metadata='cameraOff')
slicerList.append(slicer)

root.slicers=makeDict(*slicerList)
