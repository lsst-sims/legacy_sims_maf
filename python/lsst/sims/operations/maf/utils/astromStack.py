import numpy as np
from pyslalib import slalib as sla
import numpy.lib.recfunctions as rfn

"""Take an opsim object and add columns for parallax factor"""

def parallaxAmp(ra,dec,mjd):
    """given ra,dec, and observations, return the parallax amplitude for each observation.  Everything in Radians.  Note this is not numy-ified, so can't just pass arrays and get sensible results, need to loop."""
    mtoa_params = sla.sla_mappa(2000., mjd)
    ra_geo1, dec_geo1 = sla.sla_mapqk(ra,dec,0.,0.,1.,0.,mtoa_params)
    ra_geo, dec_geo = sla.sla_mapqk(ra,dec,0.,0.,0.,0.,mtoa_params)
    x_geo1, y_geo1 = gnomonic_project_toxy(ra_geo1, dec_geo1, ra,dec)
    x_geo, y_geo =  gnomonic_project_toxy(ra_geo, dec_geo, ra,dec)
    return x_geo1-x_geo, y_geo1-y_geo 

def astroStack(opsim):
    ra_pi_amp = np.zeros(np.size(opsimdata), dtype=('ra_pi_amp','float'))
    dec_pi_amp = np.zeros(np.size(opsimdata), dtype=('dec_pi_amp','float'))
    for i,ack in opsim:
        ra_pi_amp[i], dec_pi_amp[i] = parallaxAmp(opsim['fieldRA'],opsim['fieldDec'],opsim['expMJD'])
    result = rfn.merge_arrays([opsim,ra_pi_amp,dec_pi_amp], flatten=True, usemask=False)
    return result
