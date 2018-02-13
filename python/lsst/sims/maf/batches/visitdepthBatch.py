"""Sets of metrics to look at general sky coverage - nvisits/coadded depth/Teff.
"""
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.utils as mafUtils
from .colMapDict import ColMapDict, getColMap
from .common import standardSummary, filterList

__all__ = ['nvisitsM5Maps', 'tEffMetrics', 'nvisitsPerNight', 'nvisitsPerProp']


def nvisitsM5Maps(colmap=None, runName='opsim',
                  extraSql=None, extraMetadata=None,
                  nside=64, runLength=10):
    """Generate number of visits and Coadded depth per RA/Dec point in all and per filters.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    extraSql : str, opt
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    nside : int, opt
        Nside value for healpix slicer. Default 64.
        If "None" is passed, the healpixslicer-based metrics will be skipped.
    runLength : float, opt
        Length of the simulated survey, for scaling values for the plot limits.
        Default 10.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    subgroup = extraMetadata
    if subgroup is None:
        subgroup = 'All visits'

    # Set up basic all and per filter sql constraints.
    filterlist, colors, orders, sqls, metadata = filterList(all=True,
                                                            extraSql=extraSql,
                                                            extraMetadata=extraMetadata)
    # Set up some values to make nicer looking plots.
    benchmarkVals = mafUtils.scaleBenchmarks(runLength, benchmark='design')
    # Check that nvisits is not set to zero (for very short run length).
    for f in benchmarkVals['nvisits']:
        if benchmarkVals['nvisits'][f] == 0:
            print('Updating benchmark nvisits value in %s to be nonzero' % (f))
            benchmarkVals['nvisits'][f] = 1
    benchmarkVals['coaddedDepth'] = mafUtils.calcCoaddedDepth(
        benchmarkVals['nvisits'], benchmarkVals['singleVisitDepth'])
    # Scale the nvisit ranges for the runLength.
    nvisitsRange = {'u': [20, 80], 'g': [50, 150], 'r': [100, 250],
                    'i': [100, 250], 'z': [100, 300], 'y': [100, 300], 'all': [700, 1200]}
    scale = runLength / 10.0
    for f in nvisitsRange:
        for i in [0, 1]:
            nvisitsRange[f][i] = int(np.floor(nvisitsRange[f][i] * scale))

    # Generate Nvisit maps in all and per filters
    displayDict = {'group': 'Nvisits', 'subgroup': subgroup}
    metric = metrics.CountMetric(colmap['mjd'], metricName='NVisits', units='')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    for f in filterlist:
        sql = sqls[f]
        displayDict['caption'] = 'Number of visits per healpix in %s.' % metadata[f]
        displayDict['order'] = orders[f]
        binsize = 2
        if f == 'all':
            binsize = 5
        plotDict = {'xMin': nvisitsRange[f][0], 'xMax': nvisitsRange[f][1],
                    'colorMin': nvisitsRange[f][0], 'colorMax': nvisitsRange[f][1],
                    'binsize': binsize}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata[f],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Generate Coadded depth maps per filter
    displayDict = {'group': 'Coadded m5', 'subgroup': subgroup}
    metric = metrics.Coaddm5Metric(m5Col=colmap['fiveSigmaDepth'], metricName='CoaddM5')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    for f in filterlist:
        # Skip "all" for coadded depth.
        if f == 'all':
            continue
        mag_zp = benchmarkVals['coaddedDepth'][f]
        sql = sqls[f]
        displayDict['caption'] = 'Coadded depth per healpix, with %s benchmark value subtracted (%.1f) ' \
                                 'in %s.' % (f, mag_zp, metadata[f])
        displayDict['caption'] += ' More positive numbers indicate fainter limiting magnitudes.'
        displayDict['order'] = orders[f]
        plotDict = {'zp': mag_zp, 'xMin': -0.6, 'xMax': 0.6,
                    'xlabel': 'coadded m5 - %.1f' % mag_zp,
                    'colorMin': -0.6, 'colorMax': 0.6}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata[f],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def tEffMetrics(colmap=None, runName='opsim',
                extraSql=None, extraMetadata=None,
                nside=64, filterlist=('u', 'g', 'r', 'i', 'z', 'y')):
    """Generate a series of Teff metrics. Teff total, per night, and sky maps (all and per filter).

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    extraSql : str, opt
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    nside : int, opt
        Nside value for healpix slicer. Default 64.
        If "None" is passed, the healpixslicer-based metrics will be skipped.
    filterlist : list of str, opt
        List of the filternames to use for "per filter" evaluation. Default ('u', 'g', 'r', 'i', 'z', 'y').
        If None is passed, the per-filter evaluations will be skipped.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    subgroup = extraMetadata
    if subgroup is None:
        subgroup = 'All visits'

    # Set up basic all and per filter sql constraints.
    sqlconstraints = ['']
    metadata = ['all bands']
    if filterlist is not None:
        sqlconstraints += ['%s = "%s"' % (colmap['filter'], f) for f in filterlist]
    metadata += ['%s band' % f for f in filterlist]

    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        tmp = []
        for s in sqlconstraints:
            if len(s) == 0:
                tmp.append(extraSql)
            else:
                tmp.append('%s and (%s)' % (s, extraSql))
        sqlconstraints = tmp
        if extraMetadata is None:
            metadata = ['%s, %s' % (extraSql, m) for m in metadata]
    if extraMetadata is not None:
        metadata = ['%s %s' % (extraMetadata, m) for m in metadata]
    metadataCaption = extraMetadata
    if metadataCaption is None:
        metadataCaption = 'all visits'

    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Total Teff and normalized Teff.
    displayDict = {'group': 'T_eff', 'subgroup': subgroup}
    displayDict['caption'] = 'Total effective time of the survey (see Teff metric).'
    displayDict['order'] = 0
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=False, metricName='Total Teff')
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, constraint=None, displayDict=displayDict,
                             metadata=extraMetadata)
    bundleList.append(bundle)

    displayDict['caption'] = 'Normalized total effective time of the survey (see Teff metric).'
    displayDict['order'] = 1
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=True, metricName='Normalized Teff')
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, constraint=None, displayDict=displayDict,
                             metadata=extraMetadata)
    bundleList.append(bundle)

    # Generate Teff maps in all and per filters
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'], filterCol=colmap['filter'],
                                normed=True, metricName='Normalized Teff')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['order'] = -1
    for sql, meta in zip(sqlconstraints, metadata):
        displayDict['caption'] = 'Normalized effective time of the survey in %s band(s) ' \
                                 'for %s visits.' % (meta.lstrip('%s ' % extraMetadata), metadataCaption)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=meta,
                                 displayDict=displayDict, plotFuncs=subsetPlots,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerNight(colmap=None, runName='opsim', binNights=1,
                    sqlConstraint=None, metadata=None):
    """Count the number of visits per night through the survey.

    Parameters
    ----------
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    binNights : int, opt
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    sqlConstraint : str or None, opt
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    metadata : str or None, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')

    subgroup = metadata
    if subgroup is None:
        subgroup = 'All visits'

    metadataCaption = metadata
    if metadata is None:
        if sqlConstraint is not None:
            metadataCaption = sqlConstraint
        else:
            metadataCaption = 'all visits'

    bundleList = []

    displayDict = {'group': 'Per Night', 'subgroup': subgroup}
    displayDict['caption'] = 'Number of visits per night for %s.' % (metadataCaption)
    displayDict['order'] = 0
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits')
    slicer = slicers.OneDSlicer(sliceColName=colmap['mjd'], binsize=int(binNights))
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata=metadata,
                             displayDict=displayDict, summaryMetrics=standardSummary())
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerProp(opsdb, colmap=None, runName='opsim', binNights=1, sqlConstraint=None):
    """Set up a group of all and per-proposal nvisits metrics.

    Parameters
    ----------
    opsdb : lsst.sims.maf.db.Database or lsst.sims.maf.db.OpsimDatabase* object
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    binNights : int, opt
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    sqlConstraint : str or None, opt
        SQL constraint to add to all metrics.

    Returns
    -------
    metricBundle
    """
    if colmap is None:
        colmap = getColMap(opsdb)

    propids, proptags = opsdb.fetchPropInfo()

    # Calculate the total number of visits per proposal, and their fraction compared to total.
    totvisits = opsdb.fetchNVisits()
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits')
    slicer = slicers.UniSlicer()
    summaryMetrics = [metrics.IdentityMetric(metricName='Count'),
                      metrics.NormalizeMetric(normVal=totvisits, metricName='Fraction of total')]
    bundleList = []
    displayDict = {'group': 'Visit Summary', 'subgroup': 'Proposal distribution', 'order': -1}

    bdict = {}
    # All proposals.
    bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                 sqlConstraint=sqlConstraint, metadata='All props'))
    displayDict['caption'] = 'Total number of visits for all proposals'
    if sqlConstraint is not None and len(sqlConstraint) > 0:
        displayDict['caption'] += ' with constraint %s.' % sqlConstraint
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, metadata='All props',
                             displayDict=displayDict, summaryMetrics=summaryMetrics)
    bundleList.append(bundle)

    # Look for any multi-proposal groups that we should include.
    for tag in proptags:
        if len(proptags[tag]) > 1:
            pids = proptags[tag]
            sql = '('
            for pid in pids[:-1]:
                sql += '%s=%d or ' % (colmap['proposalId'], pid)
            sql += ' %s=%d)' % (colmap['proposalId'], pids[-1])
            if sqlConstraint is not None:
                sql = '(%s) and (%s)' % (sql, sqlConstraint)
            metadata = '%s' % (tag)
            bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                         sqlConstraint=sql, metadata=metadata))
            displayDict['order'] += 1
            displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
            bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                     summaryMetrics=summaryMetrics, displayDict=displayDict)
            bundleList.append(bundle)

    # And then just run each proposal separately.
    for propid in propids:
        sql = '%s=%d' % (colmap['proposalId'], propid)
        if sqlConstraint is not None:
            sql += ' and (%s)' % (sqlConstraint)
        metadata = '%s' % (propids[propid])
        bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                     sqlConstraint=sql, metadata=metadata))
        displayDict['order'] += 1
        displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
        bundle = mb.MetricBundle(metric, slicer, constraint=sql, metadata=metadata,
                                 summaryMetrics=summaryMetrics, displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    bdict.update(mb.makeBundlesDictFromList(bundleList))
    return bdict
