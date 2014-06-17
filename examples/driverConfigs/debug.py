# to run:
# python runConfig.py allSlicerCfg.py

# Example MAF config file which runs each type of available slicer.

import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Setup Database access.  Note:  Only the "root.XXX" variables are passed to the driver.
root.outputDir = './Allslicers'
root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
root.opsimNames = ['opsim_small']
#root.dbAddress ='sqlite:///opsim.sqlite'
#root.opsimNames = ['opsim_small']


# Setup a list to hold all the slicers we want to run
slicerList=[]

# How many Healpix sides to use
nside=256

# List of SQL constraints.  If multiple constraints are listed in a slicer object, they are looped over and each one is executed individualy.  
constraints = ["filter = \'%s\'"%'r', "filter = \'%s\' and night < 730"%'r']


# Configure a OneDSlicer:
# Configure a new metric
m1 = configureMetric('CountMetric', params=['slewDist'])
metricDict=makeDict(m1)
slicer = configureSlicer('OneDSlicer', kwargs={"sliceDataColName":'slewDist'},
                          metricDict=metricDict, constraints=constraints)
slicerList.append(slicer)



# Configure a UniSlicer.  Note new SQL constraints are passed
m1 = configureMetric('MeanMetric', params=['airmass'])
slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1), constraints=['night < 750'] )
slicerList.append(slicer)



# Save all the slicers to the config
root.slicers=makeDict(*slicerList)

# Optional comment string
root.comment = 'Example script that runs each of the slicers'
