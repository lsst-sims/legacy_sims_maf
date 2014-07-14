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

    args = parser.parse_args()

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
