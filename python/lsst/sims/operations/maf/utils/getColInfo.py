import numpy as np
import numpy.lib.recfunctions as rfn

class ColInfo(object):
    def __init__(self):
        self.defaultDataSource = None
        self.defaultUnit = ''
        
    def getUnits(self, colName):
        """Given a column name from OpSim, return appropriate units (for plot labels)."""
        unitDict = {'fieldID': '#',
                    'filter': 'filter',
                    'seqnNum' : '#',
                    'expMJD': 'MJD',
                    'expTime': 's',
                    'slewTime': 's',
                    'slewDist': 'rad',
                    'rotSkyPos': 'rad',
                    'rotTelPos': 'rad',
                    'rawSeeing': 'arcsec',
                    'finSeeing': 'arcsec', 
                    'seeing': 'arcsec',
                    'airmass': 'airmass',
                    'night': 'night',
                    'fieldRA': 'rad',
                    'fieldDec': 'rad', 
                    'hexdithra': 'rad', 
                    'hexdithdec': 'rad',
                    'moonRA': 'rad',
                    'moonDec': 'rad',
                    'moonAlt': 'rad',
                    'dist2Moon': 'rad', 
                    'VskyBright': 'mag/sq arcsec',
                    'perry_skybrightness': 'mag/sq arcsec',
                    'skybrightness_modified': 'mag/sq arcsec',
                    '5sigma': 'mag',
                    '5sigma_modified':'mag',
                    '5sigma_ps' : 'mag'}
        unitDict['normairmass'] = 'airmass/(minimum possible airmass)'
        if colName in unitDict:
            return unitDict[colName]
        else:
            return self.defaultUnit

    def getDataSource(self, colName):
        """Given a column name to be added to simdata, identify appropriate source. 

        For values from database, this is self.defaultDataSource ('db'). 
        For values which are precalculated for a particular column, this should be a 
        method added to this class."""
        # Note that a 'unique' list of methods should be built from the resulting returned
        #  methods, at whatever point the derived data columns will be calculated.
        sourceDict = {'normairmass': self.normAirmass,
                      'ra_pi_amp': self.parallaxFactor,
                      'dec_pi_amp': self.parallaxFactor}
        if colName in sourceDict:
            return sourceDict[colName]
        else:
            return self.defaultDataSource

    def _opsimStack(self, arrays):
        """Given a list of numpy rec arrays, it returns the merged result. """
        return rfn.merge_arrays(arrays, flatten=True, usemask=False)

    ### Normalized airmass
    
    def normAirmass(self, simData, airmassCol='airmass', decCol='fieldDec',
               telescope_lat = -30.2446388):
        """Calculate the normalized airmass for each opsim pointing."""
        min_z_possible = np.abs(simData[decCol] - np.radians(telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        norm_airmass = np.array(simData[airmassCol] / min_airmass_possible, 
                                dtype=[('normairmass', 'float')])
        return self._opsimStack([simData, norm_airmass])

    ### Parallax factors
    
    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians."""
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y
    
    def parallaxFactor(self, simData, raCol='fieldRA', decCol='fieldDec', dateCol='expMJD'):
        """Calculate the parallax factors for each opsim pointing. """
        from pyslalib import slalib as sla
        ra_pi_amp = np.zeros(np.size(simData), dtype=[('ra_pi_amp','float')])
        dec_pi_amp = np.zeros(np.size(simData), dtype=[('dec_pi_amp','float')])
        ra_geo1 = np.zeros(np.size(simData), dtype='float')
        dec_geo1 = np.zeros(np.size(simData), dtype='float')
        ra_geo = np.zeros(np.size(simData), dtype='float')
        dec_geo = np.zeros(np.size(simData), dtype='float')
        for i,ack in enumerate(simData):
            mtoa_params = sla.sla_mappa(2000., simData[dateCol][i])
            ra_geo1[i],dec_geo1[i] = sla.sla_mapqk(simData[raCol][i],simData[decCol][i],
                                                   0.,0.,1.,0.,mtoa_params)
            ra_geo[i],dec_geo[i] = sla.sla_mapqk(opsim[raCol][i],simData[decCol][i],
                                                 0.,0.,0.,0.,mtoa_params)
        x_geo1,y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1, simData[raCol],simData[decCol])
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, simData[raCol], simData[decCol])
        ra_pi_amp[:] = x_geo1-x_geo
        dec_pi_amp[:] = y_geo1-y_geo
        return self._opsimStack([simData,ra_pi_amp,dec_pi_amp]) 
