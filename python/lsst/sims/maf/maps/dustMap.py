from lsst.sims.maf.maps import BaseMap

class DustMap(BaseMap):
    """
    Compute the E(B-V) for each point in a given slicePoint
    """

    def __init__():
        self.keyname = 'ebv'

    def run(slicePoints):

        slicePoints[self.keyname] = dustValues(self.slicePoints['ra'], self.slicePoints['dec'])

        return slicePoints
    
