import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.maf.db as db

def fetchPropIDs(dbAddress):
    """Fetch the proposal IDs from the full opsim run database.  Return the full list as well as a list of proposal IDs that are wide-fast-deep (currently identified as proposal config files that contain "Universal") and Deep drilling proposals (config files containing "deep" )"""
    table = db.Table('Proposal', 'propID', dbAddress)
    propData = table.query_columns_RecArray(colnames=['propID', 'propConf', 'propName'], constraint='')
    propIDs = list(propData['propID'])
    wfdIDs = []
    ddIDs = []
    # This section needs to be updated when Opsim adds flags identifing which proposals are WFD, until then, parse on name
    for i,name in enumerate(propData['propConf']):
        if 'Universal' in name:
            wfdIDs.append(propData['propID'][i])
        if 'deep' in name:
            ddIDs.append(propData['propID'][i])
    return propIDs, wfdIDs, ddIDs

def fetchBenchmarks(dbAddress):
    """Grab the configured run length and scale selected benchmarks to match.
    Note that number of visits is truncated (floor) not rounded."""
    table = db.Table('Config', 'configID', dbAddress)
    runLength = table.query_columns_RecArray(colnames=['paramValue'], constraint=" paramName = 'nRun'")
    runLength = float(runLength['paramValue'][0]) # Years
    baseline = 10. # Baseline length (Years)
    nvisitBench={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} # 10-year Design Specs
    nvisitStretch={'u':70,'g':100, 'r':230, 'i':230, 'z':200, 'y':200} # 10-year Stretch Specs
    skyBrighntessBench = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5} 
    seeingBench = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63} # Arcsec
    singleDepthDesign = {'u':23.9,'g':25.0, 'r':24.7, 'i':24.0, 'z':23.3, 'y':22.1} 
    singleDepthStretch = {'u':24.0,'g':25.1, 'r':24.8, 'i':24.1, 'z':23.4, 'y':22.2} 
    
    if runLength != baseline:
        for key in nvisitBench:
            nvisitsBench[key] = np.floor(nvisitsBench[key] * runLength / baseline)
            nvisitStretch[key] = np.floor(nvisitStretch[key]* runLength / baseline)

    coaddedDepthDesign={}
    coaddedDepthStretch={}
    for key in singleDepthDesign:
        coaddedDepthDesign[key] = 1.25*np.log10(nvisitBench[key]*10.**(0.8*singleDepthDesign[key]))
        coaddedDepthStretch[key] = 1.25*np.log10(nvisitStretch[key]*10.**(0.8*singleDepthStretch[key])) 

    return nvisitBench, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessBench, seeingBench
