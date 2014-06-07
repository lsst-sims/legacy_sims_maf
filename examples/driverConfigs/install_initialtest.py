# A simple config test.

#To run:
#curl -O  http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1039/design/opsimblitz2_1039_sqlite.db #download an OpSim database
#runDriver.py intall_initialtest.py --runName opsimblitz2_1039

import os
from lsst.sims.maf.driver.mafConfig import MafConfig, makeBinnerConfig, makeMetricConfig, makeDict
import lsst.sims.maf.utils as utils
import lsst.sims.maf.driver as driver


def mafconfig(config, runName, dbFilepath='.', outputDir='Out', **kwargs):
    """
    Set up a MAF config for a very simple, example analysis.
    """

    # Setup Database access
    config.outputDir = outputDir
    sqlitefile = os.path.join(dbFilepath, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}

    binList = []
    nside = 64

    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisits'}, 
                            plotDict={'plotMin':0, 'plotMax':200, 'units':'N Visits'},
                            summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
        m2 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
        metricDict = makeDict(m1, m2)
        sqlconstraint = 'filter = "%s"' %(f)
        binner = makeBinnerConfig('HealpixBinner',
                                  kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                                metricDict=metricDict, constraints=[sqlconstraint,])
        config.binners=makeDict(binner)
        binList.append(binner)

    config.binners=makeDict(*binList)
    return config
