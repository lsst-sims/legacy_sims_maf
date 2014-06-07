#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg') # May want to change in the future if we want to display plots on-screen
import lsst.sims.maf.driver as driver
import argparse
from lsst.sims.maf.driver.mafConfig import MafConfig
import lsst.sims.maf.driver.driverFuncs as driverFuncs

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", type=str, help="Name of the configuration file(a pex_config python script) OR the function name from lsst.sims.maf.driver.driverFuncs that will build the config.")
    parser.add_argument("--runName", type=str, default='', help='Root name of the sqlite dbfile (i.e. filename minus _sqlite.db)')
    parser.add_argument("--filepath", type=str, default='.', help='Filepath to the sqlite dbfile')
    parser.add_argument("--outputDir", type=str, default='./Out', help='Output directory')

    args = parser.parse_args()
    config = MafConfig()
    if args.runName == '':
        config.load(args.ConfigFile)
        print 'Finished loading config file: %s'%args.ConfigFile
    else:
        config = getattr(driverFuncs,args.ConfigFile)(config, args.runName, args.filepath, args.outputDir)
        print 'Finished loading config function %s'%args.ConfigFile
    drive = driver.MafDriver(config)
    drive.run()
