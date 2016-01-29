import warnings
import numpy as np
from .baseStacker import BaseStacker
from lsst.sims.utils import Site

__all__ = ['M5OptimalStacker']


class M5OptimalStacker(BaseStacker):
    """
    Make a new m5 column as if observations were taken on the meridian.
    If the moon is up, assume sky brightness stays the same.

    Assumes seeing scales as airmass^0.6. Uses linear fits for sky and airmass relation.
    """

    def __init__(self, airmassCol='airmass', decCol='fieldDec',
                 skyBrightCol='filtSkyBrightness', seeingCol='FWHMeff',
                 filterCol='filter', m5Col ='fiveSigmaDepth',
                 moonAltCol='moonAlt', sunAltCol='sunAlt',
                 site='LSST'):
        """
        Parameters
        ----------
        airmassCol : str ('airmass')
            Column name for the airmass per pointing.
        decCol : str ('fieldDec')
            Column name for the pointing declination.
        skyBrightCol: str ('filtSkyBrightness')
            Column name for the sky brighntess per pointing.
        filterCol : str ('filter')
            Column name for the filter name.
        m5Col : str ('fiveSigmaDepth')
            Colum name for the five sigma limiting depth per pointing.
        moonAltCol : str ('moonAlt')
            Column name for the moon altitude per pointing.
        sunAltCol : str ('sunAltCol')
            Column name for the sun altitude column.
        site : str ('LSST')
            Name of the site.
        """
        self.site = Site(site)
        self.units = ['mags']
        self.colsAdded = ['m5Optimal']
        self.colsReq = [airmassCol, decCol, skyBrightCol,
                        seeingCol, filterCol, m5Col, moonAltCol, sunAltCol]

        self.airmassCol = airmassCol
        self.decCol = decCol
        self.skyBrightCol = skyBrightCol
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.m5Col = m5Col
        self.moonAltCol = moonAltCol
        self.sunAltCol = sunAltCol

    def _run(self, simData):
        # kAtm values from lsst.sims.operations gen_output.py
        kAtm = {'u': 0.50, 'g': 0.21, 'r': 0.13, 'i': 0.10,
                'z': 0.07, 'y': 0.18}
        # linear fits to sky brightness change, no moon, twilight, or zodiacal components
        skySlopes = {'u': -0.271, 'g': -0.511, 'r': -0.605, 'i': -0.670, 'z': -0.686, 'y': -0.678}
        min_z_possible = np.abs(simData[self.decCol] - self.site.latitude_rad)
        min_airmass_possible = 1./np.cos(min_z_possible)
        for filterName in np.unique(simData[self.filterCol]):
            deltaSky = skySlopes[filterName]*(simData[self.airmassCol] - min_airmass_possible)
            deltaSky[np.where((simData[self.moonAltCol] > 0) |
                              (simData[self.sunAltCol] >  np.radians(-18.)))] = 0
            m5Optimal = simData[self.m5Col] - \
                        0.5*deltaSky - \
                        0.15*np.log10(min_airmass_possible / simData[self.airmassCol]) - \
                        kAtm[filterName]*(min_airmass_possible - simData[self.airmassCol])
            good = np.where(simData[self.filterCol] == filterName)
            simData['m5Optimal'][good] = m5Optimal[good]
        return simData
