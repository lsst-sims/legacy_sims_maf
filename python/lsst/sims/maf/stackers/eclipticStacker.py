import numpy as np
import ephem
from .baseStacker import BaseStacker
from .ditherStackers import wrapRA


__all__ = ['EclipticStacker', 'mjd2djd']


def mjd2djd(mjd):
    """
    Convert MJD to Dublin Julian Date used by ephem
    """
    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    djd = mjd-doff
    return djd

class EclipticStacker(BaseStacker):
    """
    Stack on the ecliptic coordinates of each pointing.  Optionally
    subtract off the sun's ecliptic longitude and wrap.
    """
    def __init__(self, mjdCol='expMJD', raCol='fieldRA',decCol='fieldDec',
                 subtractSunLon=False):

        self.colsReq = [mjdCol,raCol,decCol]
        self.subtractSunLon = subtractSunLon
        self.colsAdded = ['eclipLat','eclipLon']
        self.units = ['radians', 'radians']
        self.mjdCol = mjdCol
        self.raCol = raCol
        self.decCol=decCol

    def run(self,simData):
        simData=self._addStackers(simData)

        for i in np.arange(simData.size):
            coord = ephem.Equatorial(simData[self.raCol][i],simData[self.decCol][i], epoch=2000)
            ecl = ephem.Ecliptic(coord)
            simData['eclipLat'][i] = ecl.lat
            if self.subtractSunLon:
                djd = mjd2djd(simData[self.mjdCol][i])
                sun = ephem.Sun(djd)
                sunEcl = ephem.Ecliptic(sun)
                lon = wrapRA(ecl.lon - sunEcl.lon)
                simData['eclipLon'][i] = lon
            else:
                simData['eclipLon'][i] = ecl.lon

        return simData
