from lsst.sims.utils import galacticFromEquatorial
from lsst.sims.maf.maps import BaseMap

__all__ = ['galCoordsMap']

class galCoordsMap(BaseMap):
    def __init__(self):
        self.keynames = ['gall', 'galb']

    def run(self, slicePoints):
        gall, galb = galacticFromEquatorial(slicePoints['ra'],slicePoints['dec'])
        slicePoints['gall'] = gall
        slicePoints['galb'] = galb
        return slicePoints
