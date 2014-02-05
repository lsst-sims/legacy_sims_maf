import numpy as np
from lsst.sims.operations.maf.utils.opsimStack import opsimStack

""" Take Opsim output and add a column for normalized airmass"""

def normAMStack(simData,  airmassCol='airmass',
                decCol='fieldDec',telescope_lat = -30.2446388):

    min_z_possible = np.abs(simData[decCol] - np.radians(telescope_lat))
    min_airmass_possible = 1./np.cos(min_z_possible)
    norm_airmass = np.array(simData[airmassCol]/min_airmass_possible, dtype=[('normairmass', 'float')])
    
    result = opsimStack([simData, norm_airmass])
    return result
