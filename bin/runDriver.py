#! /usr/bin/env python
import os, sys, argparse
import matplotlib
matplotlib.use('Agg')

import lsst.sims.maf.driver as driver
from lsst.sims.maf.driver.mafConfig import MafConfig

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Python script to interpret MAF "one-off" configuration files '
                                     'and feed them to the driver.')
    parser.add_argument("configFile", type=str, help="Name of the configuration file.")
    parser.add_argument("--version", help="Print the current version of MAF", action="store_true")

    args = parser.parse_args()

    if args.version:
        import lsst.sims.maf.utils as utils
        date, version = utils.getDateVersion()
        print 'version = '+version['__version__']
        print 'fingerprint = '+version['__fingerprint__']
        print 'repo_version = '+version['__repo_version__']
    else:

        # Set up configuration parameters.
        config = MafConfig()
        print 'Reading config data from %s' %(args.configFile)
        config.load(args.configFile)
        print 'Finished loading config file: %s' %(args.configFile)

        # Run MAF driver.
        try:
            drive = driver.MafDriver(config)
        except OSError:
            print '** %s is not a one-off driver configuration file.' %(args.configFile)
            print '** Flexible configuration files must be run using runFlexibleDriver.py.'
            print '** Try:  runFlexibleDriver.py %s --runName [runName] (etc)' %(args.configFile)
            exit()
        drive.run()
