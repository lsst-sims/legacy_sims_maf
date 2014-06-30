#! /usr/bin/env python
import os, sys, argparse
import matplotlib
matplotlib.use('Agg') # May want to change in the future if we want to display plots on-screen

import lsst.sims.maf.driver as driver
from lsst.sims.maf.driver.mafConfig import MafConfig

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Python script to interpret MAF configuration files and feed them to the driver.')
    parser.add_argument("configFile", type=str, help="Name of the configuration file (a pex_config python script) ")
    parser.add_argument("--runName", type=str, default='', help='Root name of the sqlite dbfile (i.e. filename minus _sqlite.db). If provided, then configuration file is expected to contain a "mafconfig" method to define the configuration parameters. If not, then configuration file is expected to be a pex_config python script - a "one-off" configuration file, without this method.')
    parser.add_argument("--dbDir", type=str, default='.', help='Directory containing the sqlite dbfile.')
    parser.add_argument("--outputDir", type=str, default='./Out', help='Output directory for MAF outputs.')
    parser.add_argument("--slicerName", type=str, default='HealpixSlicer', help='SlicerName, for configuration methods that use this.')

    args = parser.parse_args()

    # Set up configuration parameters.
    config = MafConfig()
    if args.runName == '':
        print 'Reading config data from %s' %(args.configFile)
        config.load(args.configFile)
        print 'Finished loading config file: %s' %(args.configFile)
    else:
        # Pull out the path and filename of the config file.
        path, name = os.path.split(args.configFile)
        # And strip off an extension (.py, for example)
        name = os.path.splitext(name)[0]
        # Add the path to the configFile to the sys.path
        if len(path) > 0:
            sys.path.insert(0, path)
        else:
            sys.path.insert(0, os.getcwd())

        # Then import the module.
        print 'Reading mafConfig from %s in %s directory' %(name, path)
        conf = __import__(name)

        # Run configuration.
        config = conf.mafConfig(config, runName=args.runName, dbDir=args.dbDir, outputDir=args.outputDir, 
                                slicerName=args.slicerName)
        print 'Finished loading config from %s.mafconfig' %(name)

    # Run MAF driver.
    drive = driver.MafDriver(config)
    drive.run()
