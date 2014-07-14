# A simple flexible driver configuration test script.

## To run:
# cd to your working directory of choice (not this directory -- somewhere outside the MAF source tree).
## copy this file to that working directory
# cd [MY_WORK_DIRECTORY]
# cp $SIMS_MAF_DIR/examples/driverConfigs/install_initialtestFlexible.py .
##  download opsim data (such as run opsimblitz2_1060) using
# curl -O  http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1060/design/opsimblitz2_1060_sqlite.db 
## run the install_initialtest.py driver config script:
# runFlexibleDriver.py --runName opsimblitz2_1060 install_initialtestFlexible.py

# Note that 'config' is the parameter which bundles up all configurable settings and passes these
# setting into the driver.

# The difference between this 'flexible' version of the install_initialtest.py config script is
#  that you can specify additional parameters at the command line:
# runFlexibleDriver.py --runName [ANY_OPSIM_RUN_NAME] --dbDir [DIR_CONTAINING_SQLite_DB]
#                      --outDir [DIR_TO_STORE_OUTPUT] install_initialtestFlexible.py
#  (note that in the one-off version of install_initialtest.py runName, dbDir and outDir are
#   harded in the first few lines of the config file).


import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mConfig(config, runName, dbDir='.', outputDir='Out', **kwargs):
    """
    Set up a MAF config for a very simple, example analysis.
    """

    # Setup Database access
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName

    # Set the output directory.
    config.outputDir = outputDir

    # Set up slicerList to store slicers.
    slicerList = []
    # Set parameter for healpix slicer resolution.
    nside = 64    

    # Loop over g and r filters.
    filters = ['g','r']
    for f in filters:
        # Set up metrics and slicers.
        m1 = configureMetric('CountMetric', kwargs={'col':'expMJD','metricName':'NVisits'}, 
                            plotDict={'colorMin':0, 'colorMax':200, 'units':'N Visits'},
                           summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
        m2 = configureMetric('Coaddm5Metric', kwargs={'m5Col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
        metricDict = makeDict(m1, m2)
        sqlconstraint = 'filter = "%s"' %(f)
        slicer = configureSlicer('HealpixSlicer',
                                  kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                                metricDict=metricDict, constraints=[sqlconstraint,])
        config.slicers=makeDict(slicer)
        slicerList.append(slicer)

    # Bundle together slicers and metrics, and pass to config 
    config.slicers=makeDict(*slicerList)
    # Return config (which then is treated like 'root' from the one-off configs in runFlexibleDriver.py)
    return config
