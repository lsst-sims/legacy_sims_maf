import numpy as np
from .baseMetric import BaseMetric


def calcDist_cosines(RA1, Dec1, RA2, Dec2):
    #taken from simSelfCalib.py
    """Calculates distance on a sphere using spherical law of cosines.
    Give this function RA/Dec values in radians. Returns angular distance(s), in radians.
    Note that since this is all numpy, you could input arrays of RA/Decs."""
    # This formula can have rounding errors for case where distances are small.
    # Oh, the joys of wikipedia - http://en.wikipedia.org/wiki/Great-circle_distance 
    # For the purposes of these calculations, this is probably accurate enough.
    D = np.sin(Dec2)*np.sin(Dec1) + np.cos(Dec1)*np.cos(Dec2)*np.cos(RA2-RA1)
    D = np.arccos(D)
    return D



class RadiusObsMetric(BaseMetric):
    """find the radius in the focal plane. """

    def __init__(self, metricName='radiusObs', racol='fieldRA',deccol='fieldDec',
                 units='radians', **kwargs):
        cols = [racol,deccol]
        self.racol = racol
        self.deccol=deccol
        self.units=units
        super(RadiusObsMetric,self).__init__(cols,metricName=metricName, **kwargs)

    def run(self, dataSlice, sliceInfo):
        ra = sliceInfo['ra']
        dec = sliceInfo['dec']
        distances = calcDist_cosines(ra,dec, dataSlice[self.racol], dataSlice[self.deccol])
        return distances

    def reduceMean(self, distances):
        return np.mean(distances)
    def reduceRMS(self,distances):
        return np.std(distances)
    def reduceFullRange(self,distances):
        return np.max(distances)-np.min(distances)
    
