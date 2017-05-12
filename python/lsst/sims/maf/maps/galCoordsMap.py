from lsst.sims.utils import _galacticFromEquatorial
from lsst.sims.maf.maps import BaseMap

__all__ = ['GalCoordsMap']

class GalCoordsMap(BaseMap):
    def __init__(self):
        self.keynames = ['gall', 'galb']

    def run(self, slicePoints):
        gall, galb = _galacticFromEquatorial(slicePoints['ra'],slicePoints['dec'])
        slicePoints['gall'] = gall
        slicePoints['galb'] = galb
        return slicePoints
