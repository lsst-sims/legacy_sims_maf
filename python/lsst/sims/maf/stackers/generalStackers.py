import warnings
import numpy as np
import palpy
from lsst.sims.utils import altAzPaFromRaDec
from lsst.sims.maf.utils import TelescopeInfo

from .baseStacker import BaseStacker

__all__ = ['NormAirmassStacker', 'ParallaxFactorStacker', 'HourAngleStacker',
            'FilterColorStacker', 'ZenithDistStacker', 'ParallacticAngleStacker']

### Normalized airmass
class NormAirmassStacker(BaseStacker):
    """
    Calculate the normalized airmass for each opsim pointing.
    """
    def __init__(self, airmassCol='airmass', decCol='fieldDec', telescope_lat = -30.2446388):

        self.units = ['airmass/(minimum possible airmass)']
        self.colsAdded = ['normairmass']
        self.colsReq = [airmassCol, decCol]
        self.airmassCol = airmassCol
        self.decCol = decCol
        self.telescope_lat = telescope_lat

    def run(self, simData):
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        min_z_possible = np.abs(simData[self.decCol] - np.radians(self.telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        simData=self._addStackers(simData)
        simData['normairmass'] = simData[self.airmassCol] / min_airmass_possible
        return simData

class ZenithDistStacker(BaseStacker):
    """
    Calculate the zenith distance for each pointing.
    """
    def __init__(self,altCol = 'altitude'):

        self.altCol = altCol
        self.units = ['radians']
        self.colsAdded = ['zenithDistance']
        self.colsReq = [self.altCol]

    def run(self, simData):
        """Calculate new column for zenith distance."""

        zenithDist = np.pi-simData[self.altCol]
        simData=self._addStackers(simData)
        simData['zenithDistance'] = zenithDist
        return simData


### Parallax factors
class ParallaxFactorStacker(BaseStacker):
    """
    Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='expMJD'):
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol
        self.units = ['arcsec', 'arcsec']
        self.colsAdded = ['ra_pi_amp', 'dec_pi_amp']
        self.colsReq = [raCol, decCol, dateCol]

    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """
        Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians.
        """
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y

    def run(self, simData):
        ra_pi_amp = np.zeros(np.size(simData), dtype=[('ra_pi_amp','float')])
        dec_pi_amp = np.zeros(np.size(simData), dtype=[('dec_pi_amp','float')])
        ra_geo1 = np.zeros(np.size(simData), dtype='float')
        dec_geo1 = np.zeros(np.size(simData), dtype='float')
        ra_geo = np.zeros(np.size(simData), dtype='float')
        dec_geo = np.zeros(np.size(simData), dtype='float')
        for i,ack in enumerate(simData):
            mtoa_params = palpy.mappa(2000., simData[self.dateCol][i])
            ra_geo1[i],dec_geo1[i] = palpy.mapqk(simData[self.raCol][i],simData[self.decCol][i],
                                                   0.,0.,1.,0.,mtoa_params)
            ra_geo[i],dec_geo[i] = palpy.mapqk(simData[self.raCol][i],simData[self.decCol][i],
                                                 0.,0.,0.,0.,mtoa_params)
        x_geo1,y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1, simData[self.raCol],simData[self.decCol])
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, simData[self.raCol], simData[self.decCol])
        ra_pi_amp[:] = np.degrees(x_geo1-x_geo)*3600.
        dec_pi_amp[:] = np.degrees(y_geo1-y_geo)*3600.
        simData=self._addStackers(simData)
        simData['ra_pi_amp'] = ra_pi_amp
        simData['dec_pi_amp'] = dec_pi_amp
        return simData

class HourAngleStacker(BaseStacker):
    """
    Add the Hour Angle for each observation.
    """
    def __init__(self, lstCol='lst', RaCol='fieldRA'):
        self.units = ['Hours']
        self.colsAdded = ['HA']
        self.colsReq = [lstCol, RaCol]
        self.lstCol = lstCol
        self.RaCol = RaCol

    def run(self, simData):
        """HA = LST - RA """
        if len(simData) == 0:
            return simData
        # Check that LST is reasonable
        if (np.min(simData[self.lstCol]) < 0) | (np.max(simData[self.lstCol]) > 2.*np.pi):
            warnings.warn('LST values are not between 0 and 2 pi')
        # Check that RA is reasonable
        if (np.min(simData[self.RaCol]) < 0) | (np.max(simData[self.RaCol]) > 2.*np.pi):
            warnings.warn('RA values are not between 0 and 2 pi')
        ha = simData[self.lstCol] - simData[self.RaCol]
        # Wrap the results so HA between -pi and pi
        ha = np.where(ha < -np.pi, ha+2.*np.pi, ha)
        ha = np.where(ha > np.pi, ha-2.*np.pi, ha)
        simData=self._addStackers(simData)
        # Convert radians to hours
        simData['HA'] = ha*12/np.pi
        return simData

class ParallacticAngleStacker(BaseStacker):
    """
    Add the parallactic angle (in radians) to each visit.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', mjdCol='expMJD', latRad=None,
                 lonRad=None):

        self.raCol = raCol
        self.decCol = decCol
        self.mjdCol = mjdCol
        if latRad is None:
            TI = TelescopeInfo('LSST')
            self.latRad = TI.lat
        else:
            self.latRad = latRad
        if lonRad is None:
            TI = TelescopeInfo('LSST')
            self.lonRad = TI.lon
        else:
            self.lonRad = lonRad

        self.units = ['radians']
        self.colsAdded = ['PA']
        self.colsReq = [self.raCol, self.decCol, self.mjdCol]

    def run(self,simData):
        simData = self._addStackers(simData)
        alt, az, pa = altAzPaFromRaDec(simData[self.raCol], simData[self.decCol], self.lonRad,
                                       self.latRad,simData[self.mjdCol])
        simData['PA'] = pa
        return simData


class FilterColorStacker(BaseStacker):
    """
    Translate filters ('u', 'g', 'r' ..) into RGB tuples.
    """
    def __init__(self, filterCol='filter', filterMap={'u':1, 'g':2, 'r':3, 'i':4, 'z':5, 'y':6}):
        self.filter_rgb_map = {'u':(0,0,1),   #dark blue
                                'g':(0,1,1),  #cyan
                                'r':(0,1,0),    #green
                                'i':(1,0.5,0.3),  #orange
                                'z':(1,0,0),    #red
                                'y':(1,0,1)}  #magenta
        self.filterCol = filterCol
        # self.units used for plot labels
        self.units = ['rChan', 'gChan', 'bChan']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['rRGB', 'gRGB', 'bRGB']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.filterCol]

    def run(self, simData):
        # Add new columns to simData, ready to fill with new values.
        simData = self._addStackers(simData)
        # Translate filter names into numbers.
        filtersUsed = np.unique(simData[self.filterCol])
        for f in filtersUsed:
            if f not in self.filter_rgb_map:
                raise IndexError('Filter %s not in filter_rgb_map' %(f))
            match = np.where(simData[self.filterCol] == f)[0]
            simData['rRGB'][match] = self.filter_rgb_map[f][0]
            simData['gRGB'][match] = self.filter_rgb_map[f][1]
            simData['bRGB'][match] = self.filter_rgb_map[f][2]
        return simData
