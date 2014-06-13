# to run:
# runConfig.py allBinnerCfg.py

# Example MAF config file which runs each type of available binner.

import numpy as np
from lsst.sims.maf.driver.mafConfig import configureBinner, configureMetric, makeDict

# Setup Database access.  Note:  Only the "root.XXX" variables are passed to the driver.
root.outputDir = './Allbinners'
root.dbAddress = {'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}#, 'OutputTable':'output'}
root.opsimName = 'example'

root.verbose = True
root.getConfig = False
# Setup a list to hold all the binners we want to run
binList=[]

# How many Healpix sides to use
nside=128

# List of SQL constraints.  If multiple constraints are listed in a binner object, they are looped over and each one is executed individualy.  
constraints = ["filter = \'%s\'"%'r', "filter = \'%s\' and night < 730"%'r']

# Configure a Healpix binner:
# Configure 2 metrics to run on the Healpix binner.  
m1 = configureMetric('CountMetric', params=['expMJD'],plotDict={'percentileClip':80., 'units':'#'}, summaryStats={'MeanMetric':{},'RmsMetric':{}})
m2 = configureMetric('Coaddm5Metric', plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.})
# Combine metrics in a dictionary
metricDict = makeDict(m1,m2)
# Generate the binner configuration, passing in the metric configurations and SQL constraints
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints)
# Add the binner to the list of binners
binList.append(binner)

# Run the same metrics, but now use the hexdither field positions:
# As before, but new spatialkeys and add a metadata keyword so the previous files don't get overwritten
binner = configureBinner('HealpixBinner',
                          kwargs={"nside":nside,'spatialkey1':"hexdithra", 'spatialkey2':"hexdithdec"},
                          metricDict = metricDict,setupKwargs={"leafsize":50000},constraints=constraints, metadata='dith')
# Add this binner to the list of binners
binList.append(binner)


# Configure a OneDBinner:
# Configure a new metric
m1 = configureMetric('CountMetric', params=['slewDist'])
metricDict=makeDict(m1)
binner = configureBinner('OneDBinner', kwargs={"sliceDataColName":'slewDist'},
                          metricDict=metricDict, constraints=constraints)
binList.append(binner)


# Configure an OpsimFieldBinner:
m1 = configureMetric('MinMetric', params=['airmass'], plotDict={'cmap':'RdBu'})
m4 = configureMetric('MeanMetric', params=['normairmass'])
m3 = configureMetric('Coaddm5Metric')
m7 = configureMetric('CountMetric', params=['expMJD'], plotDict={'units':"Number of Observations", 'percentileClip':80.})
metricDict = makeDict(m1,m3,m4,m7)
binner = configureBinner('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
binList.append(binner)


# Configure a UniBinner.  Note new SQL constraints are passed
m1 = configureMetric('MeanMetric', params=['airmass'])
binner = configureBinner('UniBinner', metricDict=makeDict(m1), constraints=['night < 750'] )
binList.append(binner)

# Configure an Hourglass filter binner/metric
m1=configureMetric('HourglassMetric')
binner = configureBinner('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750',''])
binList.append(binner)


# Save all the binners to the config
root.binners=makeDict(*binList)

# Optional comment string
root.comment = 'Example script that runs each of the binners'
