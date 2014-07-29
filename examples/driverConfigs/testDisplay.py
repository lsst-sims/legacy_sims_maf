# A simple driver configuration test script.

## To run:
# cd to your working directory of choice (not this directory -- somewhere outside the MAF source tree).
## copy this file to that working directory
# cd [MY_WORK_DIRECTORY]
# cp $SIMS_MAF_DIR/examples/driverConfigs/install_initialtest.py .
##  download opsim data (such as run opsimblitz2_1060) using
# curl -O  http://www.noao.edu/lsst/opsim/CadenceWorkshop2014/opsimblitz2_1060_sqlite.db 
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

root.verbose = True

# Set up slicerList to store slicers.
slicerList = []
# Set parameter for healpix slicer resolution.
nside = 4

# Loop over g and r filters, running metrics and slicers in each bandpass.
filters = ['u', 'g','r', 'i', 'z', 'y']
colors = {'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}

for i, f in enumerate(filters):
    # Set up metrics and slicers.
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD','metricName':'NVisits'}, 
                            plotDict={'units':'N Visits', 'xMin':0, 'xMax':200},
                            summaryStats={'MedianMetric':{}, 'RobustRmsMetric':{}},
                            histMerge={'histNum':1, 'xMin':0, 'xMax':200, 'color':colors[f]},
                            displayDict={'group':'Basic', 'subgroup':'Nvisits', 'order':i})
    m2 = configureMetric('Coaddm5Metric', kwargs={'m5Col':'fiveSigmaDepth'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MedianMetric':{}},
                            histMerge={'histNum':2, 'color':colors[f]},
                            displayDict={'group':'Basic', 'subgroup':'Coadd', 'order':i})
    metricDict = makeDict(m1, m2)
    sqlconstraint = 'filter = "%s"' %(f)
    slicer = configureSlicer('HealpixSlicer',
                            kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                            metricDict=metricDict, constraints=[sqlconstraint,])
    slicerList.append(slicer)

# Bundle together metrics and slicers and pass to 'root'
root.slicers=makeDict(*slicerList)
