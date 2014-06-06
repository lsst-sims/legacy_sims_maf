import sys, os, argparse
# Need to set matplotlib backend here.
import matplotlib
matplotlib.use('Agg')

from lsst.sims.maf.driver.mafConfig import MafConfig, makeBinnerConfig, makeMetricConfig, makeDict
import lsst.sims.maf.utils as utils
import lsst.sims.maf.driver as driver


def configSetup(config, runName, filepath, outputDir):
    """Set up the config values."""    

    # Setup Database access (user does not need to edit)
    config.outputDir = outputDir
    sqlitefile = os.path.join(filepath, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName


    ### User edit below here to define binner and metrics desired.
    
    binList=[]
    
    filters = ['r', 'g']
    nside = 64

    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisits'}, 
                            plotDict={'plotMin':0, 'plotMax':200, 'units':'N Visits'},
                            summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
        m2 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
        metricDict = makeDict(m1, m2)
        constraint = 'filter = "%s"' %(f)
        binner = makeBinnerConfig('HealpixBinner',
                                  kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                                metricDict=metricDict, constraints=[constraint,])
        config.binners=makeDict(binner)
        binList.append(binner)

    ### End of user edit for simple configs.
    config.binners=makeDict(*binList)
    return config


#############

# Boilerplate to run config for user-defined opsim run

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runName", type=str, help='Root name of the sqlite dbfile (i.e. filename minus _sqlite.db)')
    parser.add_argument("--filepath", type=str, default='.', help='Filepath to the sqlite dbfile')
    parser.add_argument("--outputDir", type=str, default='./Out', help='Output directory')
    args = parser.parse_args()

    config = MafConfig()
    config = configSetup(config, args.runName, args.filepath, args.outputDir)
    
    drive = driver.MafDriver(config)
    drive.run()

#############
