# Collection of utilities for MAF that relate to Opsim specifically. 

import os, sys
import numpy as np
import warnings

def scaleStretchDesign(self, opsimDatabase, runLengthParam='nRun'):
    """Scale the benchmark numbers for total number of visits and coadded depth (both design and stretch goals),
    based on the length of the opsim run.

    (i.e. design and stretch goals have default values: scale these from the nominal 10 years to whatever is the
    length of the current opsim run).
    Note that the number of visits is scaled to a truncated (floor) number, not rounded. """
    # Get the run length from the opsim database.
    runLength = opsimDatabase.fetchRunLength(runLengthParam=runLengthParam)
    # Set baseline (default) numbers.
    baseline = 10. # Baseline length (Years)
    nvisitDesign={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} # 10-year Design Specs
    nvisitStretch={'u':70,'g':100, 'r':230, 'i':230, 'z':200, 'y':200} # 10-year Stretch Specs
    skyBrighntessDesign = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5} 
    seeingDesign = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63} # Arcsec
    singleDepthDesign = {'u':23.9,'g':25.0, 'r':24.7, 'i':24.0, 'z':23.3, 'y':22.1} 
    singleDepthStretch = {'u':24.0,'g':25.1, 'r':24.8, 'i':24.1, 'z':23.4, 'y':22.2} 
    # Scale the number of visits.
    if runLength != baseline:
        for key in nvisitDesign:
            nvisitsDesign[key] = np.floor(nvisitsDesign[key] * runLength / baseline)
            nvisitStretch[key] = np.floor(nvisitStretch[key]* runLength / baseline)
    # Scale the coaddded depth.
    coaddedDepthDesign={}
    coaddedDepthStretch={}
    for key in singleDepthDesign:
        coaddedDepthDesign[key] = 1.25*np.log10(nvisitDesign[key]*10.**(0.8*singleDepthDesign[key]))
        coaddedDepthStretch[key] = 1.25*np.log10(nvisitStretch[key]*10.**(0.8*singleDepthStretch[key])) 
    return nvisitDesign, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessDesign, seeingDesign

