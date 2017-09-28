"""Some basic physical quantity metrics.
"""
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary, extendedMetrics

__all__ = ['metadataBasics', 'allMetadata']

def metadataBasics(value, colmap=None, runName='opsim',
                   valueName=None, groupName=None, extraSql=None, extraMetadata=None,
                   nside=64, filterlist=('u', 'g', 'r', 'i', 'z', 'y')):
    """Calculate basic metrics on visit metadata 'value' (e.g. airmass, normalized airmass, seeing..).

    Calculates extended standard metrics (with unislicer) on the quantity (all visits and per filter),
    makes histogram of the value (all visits and per filter),


    Parameters
    ----------
    value : str
        The column name for the quantity to evaluate. (column name in the database or created by a stacker).
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    valueName : str, opt
        The name of the value to be reported in the resultsDb and added to the metric.
        This is intended to help standardize metric comparison between sim versions.
        value = name as it is in the database (seeingFwhmGeom, etc).
        valueName = name to be recorded ('seeingGeom', etc.).  Default is None, which is set to match value.
    groupName : str, opt
        The group name for this quantity in the displayDict. Default is the same as 'value', capitalized.
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

    if groupName is None:
        groupName = value.capitalize()
        subgroup = extraMetadata
    else:
        subgroup = value.capitalize()

    displayDict = {'group': groupName, 'subgroup': subgroup}

    if valueName is None:
        valueName = value

    sqlconstraints = ['']
    metadata = ['All']
    if filterlist is not None:
        sqlconstraints += ['%s = "%s"' % (colmap['filter'], f) for f in filterlist]
        metadata += ['%s' % f for f in filterlist]
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
        metadata = ['%s, %s' % (extraMetadata, m) for m in metadata]

    # Summarize values over all and per filter (min/mean/median/max/percentiles/outliers/rms).
    slicer = slicers.UniSlicer()
    displayDict['caption'] = None
    for sql, meta in zip(sqlconstraints, metadata):
        displayDict['order'] = -1
        for m in extendedMetrics(value, replace_colname=valueName):
            displayDict['order'] += 1
            bundle = mb.MetricBundle(m, slicer, sql, metadata=meta, displayDict=displayDict)
            bundleList.append(bundle)

    # Histogram values over all and per filter.
    for sql, meta in zip(sqlconstraints, metadata):
        displayDict['caption'] = 'Histogram of %s' % (value)
        if valueName != value:
            displayDict['caption'] += ' (%s)' % (valueName)
        displayDict['caption'] += ' for %s visits.' % (meta)
        displayDict['order'] += 1
        m = metrics.CountMetric(value, metricName='%s Histogram' % (valueName))
        slicer = slicers.OneDSlicer(sliceColName=value)
        bundle = mb.MetricBundle(m, slicer, sql, metadata=meta, displayDict=displayDict)
        bundleList.append(bundle)

    # Make maps of min/median/max for all and per filter, per RA/Dec, with standard summary stats.
    mList = []
    mList.append(metrics.MinMetric(value, metricName='Min %s' % (valueName)))
    mList.append(metrics.MedianMetric(value, metricName='Median %s' % (valueName)))
    mList.append(metrics.MaxMetric(value, metricName='Max %s' % (valueName)))
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['caption'] = None
    displayDict['order'] = -1
    for sql, meta in zip(sqlconstraints, metadata):
        for m in mList:
            displayDict['order'] += 1
            bundle = mb.MetricBundle(m, slicer, sql, metadata=meta, displayDict=displayDict,
                                     summaryMetrics=standardSummary())
            bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def allMetadata(colmap=None, runName='opsim', sqlconstraint='', metadata='All props'):
    """Generate a large set of metrics about the metadata of each visit -
    distributions of airmass, normalized airmass, seeing, sky brightness, singlevisit depth,
    hour angle, distance to the moon, and solar elongation.
    The exact metadata which is analyzed is set by the colmap['metadataList'] value.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    sqlconstraint : str, opt
        Sql constraint (such as WFD only). Default is '' or no constraint.
    metadata : str, opt
        Metadata to identify the sql constraint (such as WFD). Default is 'All props'.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bdict = {}

    for valueName in colmap['metadataList']:
        if valueName in colmap:
            value = colmap[valueName]
        else:
            value = valueName
        bdict.update(metadataBasics(value, colmap=colmap, runName=runName,
                                    valueName=valueName, groupName=valueName,
                                    extraSql=sqlconstraint, extraMetadata=metadata))
    return bdict



