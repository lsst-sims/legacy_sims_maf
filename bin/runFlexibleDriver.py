#! /usr/bin/env python
import os, sys, argparse
import importlib
# Set matplotlib backend (to create plots where DISPLAY is not set).
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


    parser = argparse.ArgumentParser(description='Python script to interpret MAF "flexible" configuration files'
                                     'and feed them to the driver.', 
                                     epilog='%s' %(versionInfo))
    parser.add_argument("configFile", type=str, help="Name of the configuration file.")
    parser.add_argument("--runName", type=str, default='', help='Root name of the sqlite dbfile '
                        '(i.e. filename minus _sqlite.db).' )
    parser.add_argument("--dbDir", type=str, default='.', help='Directory containing the sqlite dbfile.')
    parser.add_argument("--outDir", type=str, default='./Out', help='Output directory for MAF outputs.')    
    parser.add_argument("--slicerName", type=str, default='HealpixSlicer', help='SlicerName,'
                        'for flexible configuration driver scripts that take this argument.')
    
    args = parser.parse_args()
    
    # Set up configuration parameters.
    config = MafConfig()
    # Pull out the path and filename of the config file.
    path, configname = os.path.split(args.configFile)
    # And strip off an extension (.py, for example)
    configname = os.path.splitext(configname)[0]
    # Add the path to the configFile to the sys.path
    if len(path) > 0:
        sys.path.insert(0, path)
    else:
        sys.path.insert(0, os.getcwd())
        
    # Import the module's config method.
    try:
        conf = importlib.import_module(configname)
    except NameError:
        print '** %s is not a flexible driver configuration file.' %(configname)
        print '** One-off configuration files must be run using runDriver.py.'
        print '** Try:  runDriver.py %s' %(args.configFile)
        exit()
        
    print 'Finished loading config from %s.mConfig' %(configname)        
    # Run driver configuration script.
    config = conf.mConfig(config, runName=args.runName, dbDir=args.dbDir, outputDir=args.outDir,
                          slicerName=args.slicerName)

    # Run MAF driver.
    drive = driver.MafDriver(config)
    drive.run()
