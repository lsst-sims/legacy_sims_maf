# Collection of utilities for MAF that relate to Opsim specifically.

import numpy as np

__all__ = ['connectOpsimDb', 'createSQLWhere', 'scaleBenchmarks', 'calcCoaddedDepth']

def connectOpsimDb(dbAddressDict):
    """
    Convenience function to handle connecting to database.
    (because needs to be called both from driver and from config file, with same dbAddress dictionary).
    """
    import lsst.sims.maf.db as db
    if 'summaryTable' in dbAddressDict:
        # Connect to just the summary table (might be sqlite created from flat dat output file).
        opsimdb = db.OpsimDatabase(dbAddressDict['dbAddress'],
                                   dbTables={'summaryTable':[dbAddressDict['summaryTable'], 'obsHistID']},
                                   defaultdbTables = None)
    else:
        # For a basic db connection to the sqlite db files.
        opsimdb = db.OpsimDatabase(dbAddressDict['dbAddress'])
    return opsimdb

def createSQLWhere(tag, propTags):
    """
    Create a SQL 'where' clause for 'tag' using the information in the propTags dictionary.
    (i.e. create a where clause for WFD proposals).
    Returns SQL clause.
    """
    sqlWhere = ''
    if (tag not in propTags) or (len(propTags[tag]) == 0):
        print 'No %s proposals found' %(tag)
        # Create a sqlWhere clause that will not return anything as a query result.
        sqlWhere = 'propID like "NO PROP"'
    elif len(propTags[tag]) == 1:
        sqlWhere = "propID = %d" %(propTags[tag][0])
    else:
        sqlWhere = "(" + " or ".join(["propID = %d"%(propid) for propid in propTags[tag]]) + ")"
    return sqlWhere


def scaleBenchmarks(runLength, benchmark='design'):
    """
    Given the runLength and a benchmark name ('design' or 'stretch'),
    returns the default benchmark expected values for seeing, skybrightness, single visit depth and number of visits,
     scaled to the run length.

    Note that the number of visits is scaled to a truncated number rather than rounded.
    If the benchmark name is 'requested', returns the design values of the other parameters.
    """
    # Set baseline (default) numbers for the baseline survey length (10 years).
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

    # Scale the number of visits.
    if runLength != baseline:
        scalefactor = float(runLength) / float(baseline)
        # Calculate scaled value for design and stretch values of nvisits, per filter.
        for f in design:
            design['nvisits'][f] = int(np.floor(design['nvisits'][f] * scalefactor))
            stretch['nvisits'][f] = int(np.floor(stretch['nvisits'][f] * scalefactor))

    if benchmark == 'design':
        return design
    elif benchmark == 'stretch':
        return stretch
    else:
        raise ValueError("Benchmark value %s not understood: use 'design' or 'stretch'" %(benchmark))

def calcCoaddedDepth(nvisits, singleVisitDepth):
    """
    Given dictionaries of nvisits and singleVisitDepth (per filter),
    Returns a dictionary containing the expected coadded depth.
    """
    coaddedDepth = {}
    for f in nvisits:
        if f not in singleVisitDepth:
            raise ValueError('Filter keys in nvisits and singleVisitDepth must match')
        coaddedDepth[f] = float(1.25 * np.log10(nvisits[f]) * 10**(0.8*singleVisitDepth[f]))
        if not np.isfinite(coaddedDepth[f]):
            coaddedDepth[f] = singleVisitDepth[f]
    return coaddedDepth
