#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg') # May want to change in the future if we want to display plots on-screen
import lsst.sims.maf.driver as driver
import sys, os, argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ConfigFile", type=str, help="Name of the configuration file, a pex_config python script.")
    args = parser.parse_args()
    filename = args.ConfigFile
    drive = driver.MafDriver(filename)
    drive.run()
    
