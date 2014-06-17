# A simple config test.

#To run:
#  download opsim data (such as run opsimblitz2_1060) using
#curl -O  http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1060/design/opsimblitz2_1060_sqlite.db 
#   then run using the driver (assuming you are in this directory, and you downloaded the opsim run to this directory)
#runDriver.py install_initialtest.py --runName opsimblitz2_1039
#  (note you can be in a different directory and run this config by specifying a full pathname to this config)
#  (note you can also put the opsim dbfile in a different directory and specify its location using --dbDir)


import os
from lsst.sims.maf.driver.mafConfig import MafConfig, configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mafConfig(config, runName, dbDir='.', outputDir='Out', **kwargs):
    """
    Set up a MAF config for a very simple, example analysis.
    """

    # Setup Database access
    config.outputDir = outputDir
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName


    slicerList = []
    nside = 64

    filters = ['g','r']

    for f in filters:
        m1 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisits'}, 
                            plotDict={'plotMin':0, 'plotMax':200, 'units':'N Visits'},
                            summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
        m2 = configureMetric('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
        metricDict = makeDict(m1, m2)
        sqlconstraint = 'filter = "%s"' %(f)
        slicer = configureSlicer('HealpixSlicer',
                                  kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                                metricDict=metricDict, constraints=[sqlconstraint,])
        config.slicers=makeDict(slicer)
        slicerList.append(slicer)

    config.slicers=makeDict(*slicerList)
    return config
