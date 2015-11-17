from .baseMetric import BaseMetric

__all__ = ['StarDensityMetric']

class StarDensityMetric(BaseMetric):
    """Just return the stellar density at every point"""

    def __init__(self, units='stars/sq arcsec', maps=['StellarDensityMap'], **kwargs):

        super(StarDensityMetric, self).__init__(col=[],
                                                maps=maps, units=units, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        if slicePoint['stellarDensity'] < 0:
            return self.badval
        else:
            return slicePoint['stellarDensity']/(3600.**2)
