#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg') # May want to change in the future if we want to display plots on-screen
import lsst.sims.maf.driver as driver
import argparse
from lsst.sims.maf.driver.mafConfig import MafConfig

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", type=str, help="Name of the configuration file, a pex_config python script.")
    args = parser.parse_args()
    filename = args.ConfigFile
    config = MafConfig()
    config.load(filename)
    print 'Finished loading config file: %s'%filename
    drive = driver.MafDriver(config)
    drive.run()
    
