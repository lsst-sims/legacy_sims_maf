import numpy as np
from lsst.sims.coordUtils import AstrometryBase
from lsst.sims.maf.maps import BaseMap


class galCoordsMap(BaseMap):
    def __init__(self):
        self.keynames = ['gall', 'galb']

    def run(self, slicePoints):
        gall, galb = AstrometryBase.equatorialToGalactic(slicePoints['ra'],slicePoints['dec'])
        slicePoints['gall'] = gall
        slicePoints['galb'] = galb
        return slicePoint
