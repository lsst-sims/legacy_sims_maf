# Collection of utilities for MAF that relate to Opsim specifically. 

import os, sys
import numpy as np

def connectOpsimDb(dbAddressDict):
    """Convenience function to handle connecting to database
    (because needs to be called both from driver and from config file, with same dbAddress dictionary).
    """
    import lsst.sims.maf.db as db
    if 'outputTable' in dbAddressDict:
        # Connect to just the output table (might be sqlite created from flat dat output file).
        opsimdb = db.OpsimDatabase(dbAddressDict['dbAddress'],
                                   dbTables={'outputTable':[dbAddressDict['outputTable'], 'obsHistID']},
                                   defaultdbTables = None)
    else:
        # For a basic db connection to the sqlite db files. 
        opsimdb = db.OpsimDatabase(dbAddressDict['dbAddress'])
    return opsimdb


def scaleStretchDesign(runLength):
    """Scale the benchmark numbers for total number of visits and coadded depth (both design and stretch goals),
    based on the length of the opsim run (runLength, in years).
    
    (i.e. design and stretch goals have default values: scale these from the nominal 10 years to whatever is the
    length of the current opsim run).
    Note that the number of visits is scaled to a truncated (floor) number, not rounded.
    """
    # Set baseline (default) numbers.
    # Baseline length (Years)
    baseline = 10. 
    design = {}
    stretch = {}    
    design['nvisits']={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} 
    stretch['nvisits']={'u':70,'g':100, 'r':230, 'i':230, 'z':200, 'y':200}
    design['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5} # mag/sq arcsec
    stretch['skybrightness'] = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
    design['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63} # arcsec
    stretch['seeing'] = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63} 

    design['singleVisitDepth'] = {'u':23.9,'g':25.0, 'r':24.7, 'i':24.0, 'z':23.3, 'y':22.1} 
    stretch['singleVisitDepth'] = {'u':24.0,'g':25.1, 'r':24.8, 'i':24.1, 'z':23.4, 'y':22.2} 

    filters = design['nvisits'].keys()
    # Scale the number of visits.
    if runLength != baseline:
        scalefactor = float(runLength) / float(baseline)
        for f in filters:
            design['nvisits'][f] = np.floor(design['nvisits'][f] * scalefactor)
            stretch['nvisits'][f] = np.fllor(stretch['nvisits'][f] * scalefactor)
    # Scale the coaddded depth.
    design['coaddedDepth'] = {}
    stretch['coaddedDepth'] = {}
    for f in filters:
        design['coaddedDepth'][f] = 1.25 * np.log10(design['nvisits'][f]
                                                    * 10.**(0.8*design['singleVisitDepth'][f]))
        stretch['coaddedDepth'][f] = 1.25 * np.log10(stretch['nvisits'][f]
                                                     * 10.**(0.8*stretch['singleVisitDepth'][f]))
    return design, stretch

