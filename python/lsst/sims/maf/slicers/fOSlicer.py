# Class for computing the f_0 metric.  Nearly identical
# to HealpixSlicer, but with an added plotting method

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from .healpixSlicer import HealpixSlicer
from lsst.sims.maf.metrics.summaryMetrics import fOArea, fONv

__all__ = ['fOSlicer']

class fOSlicer(HealpixSlicer):
    """fO spatial slicer"""
    def __init__(self, nside=128, lonCol ='fieldRA' , latCol='fieldDec', verbose=True, **kwargs):
        super(fOSlicer, self).__init__(verbose=verbose, lonCol=lonCol, latCol=latCol,
                                        nside=nside, **kwargs)
        # Override base plotFuncs dictionary, because we don't want to create plots from Healpix
        #  slicer (skymap, power spectrum, and histogram) -- only fO plot -- when using 'plotData'.
        self.plotFuncs = {'plotFO':self.plotFO}
