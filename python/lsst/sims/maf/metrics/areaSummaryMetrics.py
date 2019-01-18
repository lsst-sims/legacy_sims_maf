import numpy as np
from .baseMetric import BaseMetric
import healpy as hp

__all__ = ['AreaSummaryMetric']


class AreaSummaryMetric(BaseMetric):
    """
    Find the min/max of a value in the best area. This is a handy substitute for when
    users want to know "the WFD value".

    Parameters
    ----------
    area : float (18000)
        The area to consider (sq degrees)
    decreasing : bool (True)
        Should the values be sorted by increasing or decreasing order. For values where
        "larger is better", decreasing is probably what you want. For metrics where
        "smaller is better" (e.g., astrometric precission), set decreasing to False.
    reduce_func : None
        The function to reduce the clipped values by. Will default to min/max depending on
        the bool val of the decreasing kwarg.

    """
    def __init__(self, col='metricdata', metricName='AreaSummary', area=18000., decreasing=True,
                 reduce_func=None, **kwargs):
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.area = area
        self.decreasing = decreasing
        self.reduce_func = reduce_func
        if reduce_func is None:
            if decreasing:
                self.reduce_func = np.min
            else:
                self.reduce_func = np.max
        else:
            self.reduce_func = reduce_func

    def run(self, dataSlice, slicePoint=None):
        # find out what nside we have
        nside = hp.npix2nside(dataSlice.size)
        pix_area = hp.nside2pixarea(nside, degrees=True)
        n_pix_needed = np.ceil(self.area/pix_area)

        order = np.argsort(dataSlice[self.col])
        if self.decreasing:
            order = order[::-1]
        result = self.reduce_func(dataSlice[self.col][order][0:n_pix_needed])
        return result
