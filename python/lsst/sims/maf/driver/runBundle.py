import os, re, copy, warnings
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils

def runBundle(mafBundle, verbose=True, makePlots=True):
    """
    mafBundle should be a dict with the following key:value pairs:
    'metricList':List of MAF metric objects
    'slicer': instatiated MAF slicer
    'dbAddress': string that gives the address to the desired DB
    'sqlWhere': string that gives the sql where clause for pulling the data
    optional keys:
    'outDir': Output directory
    'stackerList': List of configured MAF stackers
    'metadata': string containing metadata
    'runName': string containing runName

    plotOnly:  Restore the metric values anr re-plot with new plotkwargs XXX-todo
    """

    # Check required input set
    reqKeys= ['metricList','slicer','dbAddress', 'sqlWhere']
    for key in reqKeys:
        if key not in mafBundle.keys():
            raise ValueError('"%s" not found and required as a key on the input dictionary.' % key)

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

    # Need to deepcopy here so we don't unexpectedly persist changes in mafBundle.
    # Otherwise, plotDict can get set and persist.
    sm.setMetricsSlicerStackers(copy.deepcopy(mafBundle['metricList']), copy.deepcopy(mafBundle['slicer']),
                                 stackerList=copy.deepcopy(mafBundle['stackerList']))

    dbcols = sm.findReqCols()
    # If there are summary stats, need to remove the 'metricdata' column
    while 'metricdata' in dbcols:
        dbcols.remove('metricdata')
    database = db.OpsimDatabase(mafBundle['dbAddress'])
    if verbose:
        print 'Reading in columns:'
        print dbcols
    simdata = utils.getSimData(database, mafBundle['sqlWhere'], dbcols)

    # Run any stackers that have been set manually or automatically
    for stacker in sm.stackerObjs:
        simdata = stacker.run(simdata)

    if verbose:
        print 'Found %i records, computing metrics' % simdata.size

    sm.runSlices(simdata, simDataName=mafBundle['runName'], sqlconstraint=mafBundle['sqlWhere'],
                 metadata=mafBundle['metadata'])

    # Reduce any complex metrics
    sm.reduceAll()
    # Throw a writeAll in here to get the results database ready for summary stats
    sm.writeAll()
    # Run all the summary stats
    if verbose:
        print "Running Summary stats"
    sm.summaryAll()

    if verbose:
        print 'Metrics computed, writing results and plotting.'
    sm.plotAll()

    # Create any needed merged histograms.  hmm, that probably needs to be a seperate scripts.
    # XXX-todo

    # Write the config to the output directory
    try:
        configSummary, configDetails = database.fetchConfig()
        f = open(os.path.join(mafBundle['outDir'],'configSummary.txt'), 'w')
        utils.outputUtils.printDict(configSummary, 'Config Summary', filehandle=f)
        f.close()
        f = open(os.path.join(mafBundle['outDir'], 'configDetails.txt'), 'w')
        utils.outputUtils.printDict(configDetails, 'Config Details', filehandle=f)
        f.close()
    except:
        warnings.warn('Found no OpSim config.')

    # What to return here? Just the sliceMetric object?
    return sm
