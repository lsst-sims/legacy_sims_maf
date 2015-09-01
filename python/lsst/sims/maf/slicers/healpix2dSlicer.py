import numpy as np
from .healpixSlicer import HealpixSlicer
#from lsst.sims.maf.plots.spatialPlotters import twoDPlotter

__all__ = ['Healpix2dSlicer']

class Healpix2dSlicer(HealpixSlicer):
    """
    Just like the healpix slicer, but to be used with metrics that return
    a vector, either histogramming or accumulating results.
    """

    def __init__ (self, bins=None, binCol='night', **kwargs):
        """
        Set a bin that the metric will histogram or accumulate on.
        """
        super(Healpix2dSlicer,self).__init__(**kwargs)

        # Set variables so slicer can be re-constructed
        self.slicer_init = {'nside':nside, 'lonCol':lonCol, 'latCol':latCol,
                            'radius':radius, 'bins':bins}
        self.slicePoints['bins'] = bins
        self.slicePoints['binCol'] = binCol
        self.columnsNeeded.append(binCol)
