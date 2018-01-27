import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict, getColMap
from .common import standardSummary

__all__ = ['filtersPerNightBatch','filtersWholeSurveyBatch']

def setupSlicer(binNights=None):
    if binNights is None:
        # Setup slicer, summaryStats, and captions for metrics over the entire survey.
        metadata = ['Whole survey']
        subgroup = 'Whole Survey'
        slicer = slicers.UniSlicer()
        summaryStats = None
        captionDict = {'NChanges': 'Total filter changes over survey',
                       'MinTime': 'Minimum time between filter changes, in minutes.',
                       'NFaster10':'Number of filter changes faster than 10 minutes over the entire survey.',
                       'NFaster20':'Number of filter changes faster than 10 minutes over the entire survey.',
                       'Maxin10':'Max number of filter changes within a window of 10 minutes over the entire survey.',
                       'Maxin20':'Max number of filter changes within a window of 20 minutes over the entire survey.'}
        metricName='Total Filter Changes'

    else:
        # Setup slicer, summaryStats, and captions for metrics on a per night basis.
        if binNights == 1:
            metadata = ['Per Night']
        else:
            metedata = ['Per %.1f Nights' % (binNights)]
        subgroup = 'Per Night'
        slicer = slicers.OneDSlicer(sliceColName='night', binsize=binNights)
        summaryStats = standardSummary()
        captionDict = {'NChanges': 'Number of filter changes per night.',
                       'MinTime': 'Minimum time between filter changes, in minutes, per night',
                       'NFaster10':'Number of filter changes faster than 10 minutes, per night.',
                       'NFaster20':'Number of filter changes faster than 20 minutes, per night.',
                       'Maxin10': 'Max number of filter changes within a window of 10 minutes, per night.',
                       'Maxin20':'Max number of filter changes within a window of 10 minutes, per night.'}
        metricName='Filter Changes'

    return metadata,subgroup,slicer,summaryStats,captionDict,metricName

def nfilterChanges(colmap=None, runName='opsim', binNights=None, extraSql=None, extraMetadata=None):

    """Generate a set of metrics measuring the number and rate of filter changes.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    binNights : int, opt
        Size of night bin to use when calculating metrics. If left None, metrics
        will be calculated over the entire survey
    extraSql : str, opt
        Additional constraint to add to any sql constraints (e.g. 'propId=1' or 'fieldID=522').
        Default None, for no additional constraints.
    extraMetadata : str, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # Set up basic all and per filter sql constraints.
    sqlconstraints = ['']

    metadata, subgroup, slicer, summaryStats, captionDict, metricName = setupSlicer(binNights)
    displayDict = {'group': 'Filter Changes', 'subgroup': subgroup}

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
            metadata = ['%s %s' % (extraSql, m) for m in metadata]
    if extraMetadata is not None:
        metadata = ['%s %s' % (extraMetadata, m) for m in metadata]
    metadataCaption = extraMetadata
    if metadataCaption is None:
        metadataCaption = 'all visits'

    for sql, meta in zip(sqlconstraints, metadata):

        # Number of filter changes
        metric = metrics.NChangesMetric(col='filter', metricName=metricName)
        plotDict = {'ylabel': 'Number of Filter Changes'}
        displayDict['caption'] = captionDict['NChanges']
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

        # Minimum time between filter changes (minutes)
        metric = metrics.MinTimeBetweenStatesMetric(changeCol='filter')
        plotDict = {'yMin': 0, 'yMax': 120}
        displayDict['caption'] = captionDict['MinTime']
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

        # N Filter changes faster than 10 minutes
        metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=10)
        plotDict = {}
        displayDict['caption'] = captionDict['NFaster10']
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

        # N Filter changes faster than 20 minutes
        displayDict['caption'] = captionDict['NFaster20']
        metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=20)
        plotDict = {}
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

        # Max N Filter changes faster than 10 minutes
        metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=10)
        plotDict = {}
        displayDict['caption'] = captionDict['Maxin10']
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

        # Max N Filter changes faster than 20 minutes
        metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=20)
        plotDict = {}
        displayDict['caption'] = captionDict['Maxin20']
        bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=meta,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)

def filtersPerNightBatch(colmap=None, runName='opsim',extraSql=None, extraMetadata=None):
    return nfilterChanges(colmap=colmap, binNights=1.0, runName=runName,
                          extraSql=extraSql, extraMetadata=extraMetadata)

def filtersWholeSurveyBatch(colmap=None, runName='opsim',extraSql=None, extraMetadata=None):
    return nfilterChanges(colmap=colmap, binNights=None, runName=runName,
                          extraSql=extraSql, extraMetadata=extraMetadata)
