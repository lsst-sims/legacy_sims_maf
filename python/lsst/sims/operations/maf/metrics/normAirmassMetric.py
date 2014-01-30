import numpy as np
from .baseMetric import BaseMetric

class NormAirmassMetric(BaseMetric):
    """Calculate the normalized airmass """
    def __init__(self, metricName='normAirmass', airmassCol='airmass', decCol='fieldDec', telescope_lat = -30.2446388, reducer=np.mean):
        self.telescope_lat = telescope_lat
        self.dec = decCol
        self.airmass = airmassCol
        self.reducer = reducer
        super(NormAirmassMetric, self).__init__([self.dec,self.airmass],metricName)
        self.metricDtype='float'
        return

    def run(self, dataSlice):
        #Note that I'm using field location, not object or binpoint location, so there's a bit of approximation going on. I think things should average out...This means the calc is basically only being done at field centers, which is probably fine, since that's the only place we have the true airmasses for.
        min_z_possible = np.abs(dataSlice[self.dec]-self.telescope_lat) #minimum possible zenith angle
        min_airmass_possible = 1./np.cos(np.radians(min_z_possible))
        norm_airmass = dataSlice[self.airmass]/min_airmass_possible
        #it would be more efficient to pre-calculate a normalized airmass column and then use the simple metrics.  
        result = self.reducer(norm_airmass)
        return result
    
