#! /usr/bin/env python
import numpy as np
import os, sys, argparse
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plotters
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils
import healpy as hp
import matplotlib.pylab as plt


def makeBundleList():



    return bundleList

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the scheduler validation metrics')
