import numpy as np
import ephem
from lsst.sims.utils import _galacticFromEquatorial

from .baseStacker import BaseStacker
from .ditherStackers import wrapRA

__all__ = ['GalacticStacker',
           'EclipticStacker', 'mjd2djd']

def mjd2djd(mjd):
    """Convert MJD to the Dublin Julian date used by ephem.
    
    Parameters
    ----------
    mjd : float or numpy.ndarray
        The modified julian date.
    Returns
    -------
    float or numpy.ndarray
        The dublin julian date.
    """
    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    djd = mjd-doff
    return djd


class GalacticStacker(BaseStacker):
    """Add the galactic coordinates of each RA/Dec pointing.

    Parameters
    ----------
    raCol : str, opt
        Name of the RA column. Default fieldRA.
    decCol : str, opt
        Name of the Dec column. Default fieldDec.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec'):
        self.colsReq = [raCol, decCol]
        self.colsAdded = ['gall','galb']
        self.units = ['radians', 'radians']
        self.raCol = raCol
        self.decCol = decCol

    def _run(self, simData):
        # raCol and DecCol in radians, gall/b in radians.
        simData['gall'], simData['galb'] = _galacticFromEquatorial(np.radians(simData[self.raCol]),
                                                                   np.radians(simData[self.decCol]))
        return simData

class EclipticStacker(BaseStacker):
    """Add the ecliptic coordinates of each RA/Dec pointing.
    Optionally subtract off the sun's ecliptic longitude and wrap.
    
    Parameters
    ----------
    mjdCol : str, opt
        Name of the MJD column. Default expMJD.
    raCol : str, opt
        Name of the RA column. Default fieldRA.
    decCol : str, opt
        Name of the Dec column. Default fieldDec.
    subtractSunLon : bool, opt
        Flag to subtract the sun's ecliptic longitude. Default False.
    """
    def __init__(self, mjdCol='observationStartMJD', raCol='fieldRA',decCol='fieldDec',
                 subtractSunLon=False):

        self.colsReq = [mjdCol, raCol, decCol]
        self.subtractSunLon = subtractSunLon
        self.colsAdded = ['eclipLat', 'eclipLon']
        self.units = ['radians', 'radians']
        self.mjdCol = mjdCol
        self.raCol = raCol
        self.decCol=decCol

    def _run(self, simData):
        for i in np.arange(simData.size):
            coord = ephem.Equatorial(np.radians(simData[self.raCol][i]),
                                     np.radians(simData[self.decCol][i]), epoch=2000)
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
