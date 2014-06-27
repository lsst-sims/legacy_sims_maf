# to run:
# python runConfig.py allBinnerCfg.py

# Example MAF config file which runs each type of available binner.

import numpy as np
from lsst.sims.maf.driver.mafConfig import configureBinner, configureMetric, makeDict

# Setup Database access.  Note:  Only the "root.XXX" variables are passed to the driver.
root.outputDir = './Allbinners'
root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
root.opsimNames = ['opsim_small']
#root.dbAddress ='sqlite:///opsim.sqlite'
#root.opsimNames = ['opsim_small']


# Setup a list to hold all the binners we want to run
binList=[]

# How many Healpix sides to use
nside=256

# List of SQL constraints.  If multiple constraints are listed in a binner object, they are looped over and each one is executed individualy.  
constraints = ["filter = \'%s\'"%'r', "filter = \'%s\' and night < 730"%'r']


# Configure a OneDBinner:
# Configure a new metric
m1 = configureMetric('CountMetric', params=['slewDist'])
metricDict=makeDict(m1)
binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'slewDist'},
                          metricDict=metricDict, constraints=constraints)
binList.append(binner)



# Configure a UniBinner.  Note new SQL constraints are passed
m1 = configureMetric('MeanMetric', params=['airmass'])
binner = configureBinner('UniBinner', metricDict=makeDict(m1), constraints=['night < 750'] )
binList.append(binner)



# Save all the binners to the config
root.binners=makeDict(*binList)

# Optional comment string
root.comment = 'Example script that runs each of the binners'
