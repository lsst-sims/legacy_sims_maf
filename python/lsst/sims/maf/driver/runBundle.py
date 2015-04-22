import numpy as np
import os
import re
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

def runBundle(mafBundle, verbose=True, plotOnly=False):
    """
    mafBundle should be a dict with the following key:value pairs:
    'metricList':List of instatiated MAF metrics
    'slicer': instatiated MAF slicer
    'dbAddress': string that gives the address to the desired DB
    'sqlWhere': string that gives the sql where clause for pulling the data
    optional mafBundle keys:
    'outDir': Output directory
    'stackerList': List of configured MAF stackers
    'metadata': string containing metadata
    'runName': string containing runName

    plotOnly:  Restore the metric values anr re-plot with new plotkwargs XXX-todo
    """

    # Set the optional keys with defaults if missing
    if 'outDir' not in mafBundle.keys():
        mafBundle['outDir'] = 'Output'
    if 'metadata' not in mafBundle.keys():
        mafBundle['metadata'] = ''
    if 'stackerList' not in mafBundle.keys():
        mafBundle['stackerList'] = None
    if 'runName' not in mafBundle.keys():
        runName = re.sub('.*//', '', mafBundle['dbAddress'])
        runName = re.sub('\..*', '', runName)
        mafBundle['runName'] = runName

    sm = sliceMetrics.RunSliceMetric(outDir = mafBundle['outDir'])
    sm.setMetricsSlicerStackers(mafBundle['metricList'], mafBundle['slicer'],
                                 stackerList=mafBundle['stackerList'])

    dbcols = sm.findDataCols()
    database = db.OpsimDatabase(mafBundle['dbAddress'])
    if verbose:
        print 'Reading in columns:'
        print dbcols
    simdata = utils.getSimData(database, mafBundle['sqlWhere'], dbcols)

    if verbose:
        print 'Found %i records, computing metrics' % simdata.size

    sm.runSlices(simdata, simDataName=runName, sqlconstraint=mafBundle['sqlWhere'],
                 metadata=mafBundle['metadata'])

    # Reduce any complex metrics
    sm.reduceAll()

    if verbose:
        print 'Metrics computed, writing results and plotting.'
    sm.writeAll()
    sm.plotAll()

    # Need to do the summary stats here, maybe make summaryStats a metric kwarg to keep things straight. Maybe update the runSliceMetric to hold onto the results

    # Create any needed merged histograms

    # Write the config to the output directory
    try:
        utils.writeConfigs(opsdb, outDir)
    except:
        print 'Found no OpSim config.'

    # What to return here? Just the sliceMetric object?
    return sm
