"""Sets of metrics to look at general sky coverage - nvisits/coadded depth/Teff.
"""
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict, getColMap
from .common import standardSummary

__all__ = ['nvisitsM5Maps', 'tEffMetrics', 'nvisitsPerNight', 'nvisitsPerProp']


def nvisitsM5Maps(colmap=None, runName='opsim',
                  extraSql=None, extraMetadata=None,
                  nside=64, filterlist=('u', 'g', 'r', 'i', 'z', 'y')):
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
    metadata = ['all']
    if filterlist is not None:
        sqlconstraints += ['%s = "%s"' % (colmap['filter'], f) for f in filterlist]
    metadata += ['%s' % f for f in filterlist]

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

    # Generate Nvisit maps in all and per filters
    displayDict = {'group': Nvisits, 'subgroup': subgroup}
    metric = metrics.CountMetric(colMap['mjd'], metricName='NVisits')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['order'] = -1
    for sql, meta in zip(sqlconstraints, metadata):
        displayDict['caption'] = 'Number of visits per healpix in %s band(s), ' \
                                 'for %s visits.' % (meta.lstrip('%s ' % extraMetadata), metadataCaption)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=meta,
                                 displayDict=displayDict,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Generate Coadded depth maps in all and per filters
    displayDict = {'group': 'Coadded m5', 'subgroup': subgroup}
    metric = metrics.CoaddM5Metric(m5Col=colmap['fiveSigmaDepth'], metricName='CoaddM5')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['order'] = -1
    for sql, meta in zip(sqlconstraints, metadata):
        displayDict['caption'] = 'Number of visits per healpix in %s band(s), ' \
                                 'for %s visits.' % (meta.lstrip('%s ' % extraMetadata), metadataCaption)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=meta,
                                 displayDict=displayDict,
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
    metadata = ['all']
    if filterlist is not None:
        sqlconstraints += ['%s = "%s"' % (colmap['filter'], f) for f in filterlist]
    metadata += ['%s' % f for f in filterlist]

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
                                 displayDict=displayDict, plots=subsetPlots,
                                 summaryMetrics=standardSummary())
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerNight(colmap=None, runName='opsim', binNights=1,
                    sql=None, metadata=None):
    """Count the number of visits per night through the survey.

    Parameters
    ----------
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    binNights : int, opt
        Number of nights to count in each bin. Default = 1, count number of visits in each night.
    sql : str, opt
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    metadata : str, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    subgroup = metadata
    if subgroup is None:
        subgroup = 'All visits'

    metadataCaption = metadata
    if metadata is None:
        if sql is not None:
            metadataCaption = sql
        else:
            metadataCaption = 'all visits'

    bundleList = []

    displayDict = {'group': 'Per Night', 'subgroup': 'Nvisits'}
    displayDict['caption'] = 'Number of visits per night for %s.' % (metadataCaption)
    displayDict['order'] = 0
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits')
    slicer = slicers.OneDSlicer(sliceColName=colmap['mjd'], binsize=int(binNights))
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                             displayDict=displayDict, summaryMetrics=standardSummary())
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def nvisitsPerProp(opsdb, colmap=None, runName='opsim', binNights=1):
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
    displayDict = {'group': 'Visit Summary', 'subgroup': 'Proposal distribution', 'order':-1}

    bdict = {}
    bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                 sql=None, metadata='All visits'))

    # Look for any multi-proposal groups that we should include.
    for tag in proptags:
        if len(proptags[tag]) > 1:
            pids = proptags[tag]
            sql = '('
            for pid in pids[:-1]:
                sql += 'proposalId=%d or ' % pid
            sql += ' proposalId=%d)' % pids[-1]
            metadata = '%s' % (tag)
            bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                         sql=sql, metadata=metadata))
            displayDict['order'] += 1
            displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
            bundle = mb.MetricBundle(metric, slicer, sql=sql, metadata=metadata,
                                     summaryMetrics=summaryMetrics, displayDict=displayDict)
            bundleList.append(bundle)

    # And then just run each proposal separately.
    for propid in propids:
        sql = 'proposalId=%d' % (propid)
        metadata = '%s' % (propids[propid])
        bdict.update(nvisitsPerNight(colmap=colmap, runName=runName, binNights=binNights,
                                     sql=sql, metadata=metadata))
        displayDict['order'] += 1
        displayDict['caption'] = 'Number of visits and fraction of total visits, for %s.' % metadata
        bundle = mb.MetricBundle(metric, slicer, sql=sql, metadata=metadata,
                                 summaryMetrics=summaryMetrics, displayDict=displayDict)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    bdict.update(mb.makeBundlesDictFromList(bundleList))
    return bdict