# A simple driver configuration test script.

## To run:
# cd to your working directory of choice (not this directory -- somewhere outside the MAF source tree).
## copy this file to that working directory
# cd [MY_WORK_DIRECTORY]
# cp $SIMS_MAF_DIR/examples/driverConfigs/install_initialtest.py .
##  download opsim data (such as run opsimblitz2_1060) using
# curl -O  http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1060/design/opsimblitz2_1060_sqlite.db 
## run the install_initialtest.py driver config script:
# runDriver.py install_initialtest.py

# Note that 'root' is the parameter which bundles up all configurable settings and passes these
# setting into the driver.

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils

# Setup Database access. 
dbDir = '.'
runName = 'opsimblitz2_1060'
sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
root.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
root.opsimName = runName

# Set output directory.
root.outputDir = 'OutDir'

# Set up slicerList to store slicers.
slicerList = []
# Set parameter for healpix slicer resolution.
nside = 64

# Loop over g and r filters, running metrics and slicers in each bandpass.
filters = ['g','r']
for f in filters:
    # Set up metrics and slicers.
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD','metricName':'NVisits'}, 
                            plotDict={'colorMin':0, 'colorMax':200, 'units':'N Visits',
                                      'xMin':0, 'xMax':200},
                            summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
    m2 = configureMetric('Coaddm5Metric', kwargs={'m5Col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
    metricDict = makeDict(m1, m2)
    sqlconstraint = 'filter = "%s"' %(f)
    slicer = configureSlicer('HealpixSlicer',
                            kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                            metricDict=metricDict, constraints=[sqlconstraint,])
    slicerList.append(slicer)

# Bundle together metrics and slicers and pass to 'root'
root.slicers=makeDict(*slicerList)
