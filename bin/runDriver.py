#! /usr/bin/env python
import argparse
import matplotlib
matplotlib.use('Agg')

import lsst.sims.maf.driver as driver
from lsst.sims.maf.driver.mafConfig import MafConfig
import lsst.sims.maf.utils as utils

if __name__=="__main__":

    date, version = utils.getDateVersion()
    versionNum = version['__version__']
    fingerPrint = version['__fingerprint__']
    repoVersion = version['__repo_version__']

    versionInfo = 'Running:: version %s,  fingerprint %s,  repoversion %s' %(versionNum, fingerPrint, repoVersion)

    parser = argparse.ArgumentParser(description='Python script to interpret MAF "one-off" configuration files '
                                     'and feed them to the driver.',
                                     epilog= '%s' %(versionInfo))
    parser.add_argument("configFile", type=str, help="Name of the configuration file.")
    parser.add_argument("--plotOnly", dest='plotOnly', action='store_true', help="Restore data and regenerate plots")
    parser.set_defaults(plotOnly=False)

    args = parser.parse_args()


    # Set up configuration parameters.
    config = MafConfig()
    print 'Reading config data from %s' %(args.configFile)
    config.load(args.configFile)
    print 'Finished loading config file: %s' %(args.configFile)
    if args.plotOnly:
        config.plotOnly = True

    # Run MAF driver.
    try:
        drive = driver.MafDriver(config)
    except OSError:
        print '** %s is not a one-off driver configuration file.' %(args.configFile)
        print '** Flexible configuration files must be run using runFlexibleDriver.py.'
        print '** Try:  runFlexibleDriver.py %s --runName [runName] (etc)' %(args.configFile)
        exit()
    drive.run()
