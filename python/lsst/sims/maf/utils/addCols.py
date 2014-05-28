import numpy as np
import numpy.lib.recfunctions as rfn


def opsimStack(arrays):
    """Given a list of numpy rec arrays, it returns the merged result. """
    return rfn.merge_arrays(arrays, flatten=True, usemask=False)

# If adding a new column, be sure to update getUnits and getDataSource in .getColInfo so the mafDriver can find it.  Should this be changed so that all the stackers know their units and it's done automatically?


### Normalized airmass

class NormAirmass(object):
    """Calculate the normalized airmass for each opsim pointing."""
    def __init__(self, airmassCol='airmass', decCol='fieldDec',
           telescope_lat = -30.2446388):
        self.name = 'normairmass'
        self.airmassCol=airmassCol
        self.decCol=decCol
        self.telescope_lat = telescope_lat
        self.cols=[airmassCol, decCol] 

    def run(self, simData):
        min_z_possible = np.abs(simData[self.decCol] - np.radians(self.telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        norm_airmass = np.array(simData[self.airmassCol] / min_airmass_possible, 
                            dtype=[('normairmass', 'float')])
        if 'normairmass' in simData.dtype.names:
            simData['normairmass'] = norm_airmass
        else:
            simData = opsimStack([simData, norm_airmass])
        return simData

### Parallax factors

class ParallaxFactor(object):
    """Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds"""
    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='expMJD'):
        self.name='parallaxfactor'
        self.cols = [raCol,decCol,dateCol]
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol

    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians."""
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y

    def run(self,simData):
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
        if 'ra_pi_amp' in simData.dtype.names:
            simData['ra_pi_amp'] = ra_pi_amp
            simData['dec_pi_amp'] = dec_pi_amp
        else:
            simData = opsimStack([simData,ra_pi_amp,dec_pi_amp]) 
        return simData

# Add a new dither pattern

class DecOnlyDither(object):
    """Dither the position of pointings in dec only.  """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night',
                 nightStep=1, nSteps=5, stepSize=0.2):
        self.name='decDither'
        self.cols=[raCol,decCol,nightCol]
        self.raCol = raCol
        self.decCol = decCol
        self.nightCol = nightCol
        self.nightStep=nightStep
        self.nSteps = nSteps
        self.stepSize = stepSize
        
    def run(self, simData):
        off1 = np.arange(self.nSteps+1)*self.stepSize
        off2 = off1[::-1][1:]
        off3 = -1.*off1[1:]
        off4 = off3[::-1][1:]
        offsets = np.radians(np.concatenate((off1,off2,off3,off4) ))
        uoffsets = np.size(offsets)
        nightIndex = simData[self.nightCol]%uoffsets
        decDither = np.zeros(np.size(simData), dtype=[('decOnlyDither','float')])
        decDither['decOnlyDither'] = simData[self.decCol]+offsets[nightIndex]
        simData = opsimStack([simData, decDither])
        return simData
    
                             
