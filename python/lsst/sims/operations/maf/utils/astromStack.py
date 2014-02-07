import numpy as np
from pyslalib import slalib as sla
from lsst.sims.operations.maf.utils.opsimStack import opsimStack

"""Take an opsim object and add columns for parallax factor"""

def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
    Input radians."""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y

def astromStack(opsim):
    """Compute the parallax factors for each Opsim pointing and
    add the ra and dec parallax columns to the numpy rec array.
    returns the parallax amplitude (in radians) for an object with 1 arcsec parallax"""
    #arrays to add to opsim output
    ra_pi_amp = np.zeros(np.size(opsim), dtype=[('ra_pi_amp','float')])
    dec_pi_amp = np.zeros(np.size(opsim), dtype=[('dec_pi_amp','float')])
    #temp arrays to hold slalib output
    ra_geo1 = np.zeros(np.size(opsim), dtype='float')
    dec_geo1 = np.zeros(np.size(opsim), dtype='float')
    ra_geo = np.zeros(np.size(opsim), dtype='float')
    dec_geo = np.zeros(np.size(opsim), dtype='float')
    for i,ack in enumerate(opsim):
        mtoa_params = sla.sla_mappa(2000., opsim['expMJD'][i])
        ra_geo1[i],dec_geo1[i] = sla.sla_mapqk(opsim['fieldRA'][i],opsim['fieldDec'][i],0.,0.,1.,0.,mtoa_params)
        ra_geo[i],dec_geo[i] = sla.sla_mapqk(opsim['fieldRA'][i],opsim['fieldDec'][i],0.,0.,0.,0.,mtoa_params)
    x_geo1,y_geo1 = gnomonic_project_toxy(ra_geo1, dec_geo1, opsim['fieldRA'],opsim['fieldDec'])
    x_geo, y_geo = gnomonic_project_toxy(ra_geo, dec_geo, opsim['fieldRA'],opsim['fieldDec'])
    ra_pi_amp[:] = x_geo1-x_geo
    dec_pi_amp[:] = y_geo1-y_geo
    result = opsimStack([opsim,ra_pi_amp,dec_pi_amp]) 
    return result
