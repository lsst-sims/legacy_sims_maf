import numpy as np
import numpy.lib.recfunctions as rfn
from .baseStacker import BaseStacker
import warnings
        
### Normalized airmass
class NormAirmassStacker(BaseStacker):
    """Calculate the normalized airmass for each opsim pointing."""
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

### Parallax factors
class ParallaxFactorStacker(BaseStacker):
    """Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds"""
    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='expMJD'):
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol
        self.units = ['arcsec', 'arcsec']
        self.colsAdded = ['ra_pi_amp', 'dec_pi_amp']
        self.colsReq = [raCol, decCol, dateCol]

    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians."""
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y

    def run(self, simData):
        import palpy
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
    """ Add the Hour Angle for each observation """
    def __init__(self, lstCol='lst', RaCol='fieldRA'):
        self.units = ['Hours']
        self.colsAdded = ['HA']
        self.colsReq = [lstCol, RaCol]
        self.lstCol = lstCol 
        self.RaCol = RaCol

    def run(self, simData):
        """HA = LST - RA """
        # Check that LST is reasonable
        if (np.min(simData[self.lstCol]) < 0) | (np.max(simData[self.lstCol]) > 2.*np.pi):
            warnings.warn('LST values are not between 0 and 2 pi')
        # Check that RA is reasonable
        if (np.min(simData[self.RaCol]) < 0) | (np.max(simData[self.RaCol]) > 2.*np.pi):
            warnings.warn('RA values are not between 0 and 2 pi')
        ha = simData[self.lstCol] - simData[self.RaCol]
        # Wrap the results so HA between -pi and pi
        ha[np.where(ha < -np.pi)] = ha[np.where(ha < -np.pi)] + 2.*np.pi
        ha[np.where(ha > np.pi)] = ha[np.where(ha > np.pi)] - 2.*np.pi
        simData=self._addStackers(simData)
        # Convert radians to hours
        simData['HA'] = ha*12/np.pi 
        return simData
    
