from .baseMetric import BaseMetric
from scipy.interpolate import interp1d

__all__ = ['StarDensityMetric']

class StarDensityMetric(BaseMetric):
    """Just return the stellar density at every point"""

    def __init__(self, rmagLimit=25., units='stars/sq arcsec', maps=['StellarDensityMap'], **kwargs):

        super(StarDensityMetric, self).__init__(col=[],
                                                maps=maps, units=units, **kwargs)
        self.rmagLimit = rmagLimit

    def run(self, dataSlice, slicePoint=None):
        # Interpolate the data to the requested mag
        interp = interp1d(slicePoint['starMapBins'][1:], slicePoint['starLumFunc'])
        # convert from stars/sq degree to stars/sq arcsec
        result = interp(self.rmagLimit)/(3600.**2)
        return result
