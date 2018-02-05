"""Sets of metrics to look at time between visits/pairs, etc.
"""
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['intraNight', 'interNight']


def intraNight(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None):
    """Generate a set of statistics about the pair/triplet/etc. rate within a night.

    Parameters
    ----------
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, opt
        Additional sql constraint to apply to all metrics.
    extraMetadata : str or None, opt
        Additional metadata to apply to all results.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    metadata = extraMetadata
    if extraSql is not None and len(extraSql) > 0:
        sqlConstraint = '(%s) and ' % extraSql
        if metadata is None:
            metadata = extraSql
    else:
        sqlConstraint = ''

    bundleList = []
    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Look for the fraction of visits in gri where there are pairs within dtMin/dtMax.
    displayDict = {'group': 'IntraNight', 'subgroup': 'Pairs', 'caption': None, 'order': 0}
    sql = '%s (filter="g" or filter="r" or filter="i")' % sqlConstraint
    md = 'gri'
    if metadata is not None:
        md += metadata
    dtMin = 15.0
    dtMax = 60.0
    metric = metrics.PairFractionMetric(mjdCol=colmap['mjd'], minGap=dtMin, maxGap=dtMax,
                                        metricName='Fraction of visits in pairs (%.0f-%.0f min)' % (dtMin,
                                                                                                    dtMax))
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['caption'] = 'Fraction of %s visits that have a paired visit' \
                             'between %.1f and %.1f minutes away. ' % (metadata, dtMin, dtMax)
    displayDict['caption'] += 'If all visits were in pairs, this fraction would be 1.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Look at the fraction of visits which have another visit within dtMax.
    dtMax = 50.0
    metric = metrics.NRevisitsMetric(mjdCol=colmap['mjd'], dT=dtMax, normed=True,
                                     metricName='Fraction of visits with a revisit < %.0f min' % dtMax)
    displayDict['caption'] = 'Fraction of %s visits that have another visit ' \
                             'within %.1f min. ' % (metadata, dtMax)
    displayDict['caption'] += 'If all visits were in pairs (only), this fraction would be 0.5.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Histogram of the time between revisits within two hours.
    binMin = 0
    binMax = 120.
    binsize = 5.
    bins_metric = np.arange(binMin / 60.0 / 24.0, (binMax + binsize) / 60. / 24., binsize / 60. / 24.)
    bins_plot = bins_metric * 24.0 * 60.0
    metric = metrics.TgapsMetric(bins=bins_metric, timesCol=colmap['mjd'], metricName='DeltaT Histogram')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    plotDict = {'bins': bins_plot, 'xlabel': 'dT (minutes)'}
    displayDict['caption'] = 'Histogram of the time between consecutive visits to a given point ' \
                             'on the sky, considering visits between %.1f and %.1f minutes' % (binMin,
                                                                                               binMax)
    displayDict['order'] += 1
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def interNight(colmap=None, runName='opsim', nside=64, extraSql=None, extraMetadata=None):
    """Generate a set of statistics about the spacing between nights with observations.

    Parameters
    ----------
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        Nside for the healpix slicer. Default 64.
    extraSql : str or None, opt
        Additional sql constraint to apply to all metrics.
    extraMetadata : str or None, opt
        Additional metadata to use for all outputs.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    metadata = extraMetadata
    if extraSql is not None and len(extraSql) > 0:
        sqlConstraint = '(%s) and ' % extraSql
        if metadata is None:
            metadata = extraSql
    else:
        sqlConstraint = ''

    filterlist = ('u', 'g', 'r', 'i', 'z', 'y', 'all')
    colors = {'u': 'cyan', 'g': 'g', 'r': 'y', 'i': 'r', 'z': 'm', 'y': 'b', 'all': 'k'}
    filterorder = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6, 'all': 0}

    displayDict = {'group': 'InterNight', 'subgroup': 'Night gaps', 'caption': None, 'order': 0}
    bins = np.arange(0, 20.5, 1)
    metric = metrics.NightgapsMetric(bins=bins, nightCol=colmap['night'], metricName='DeltaNight Histogram')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    plotDict = {'bins': bins, 'xlabel': 'dT (nights)'}
    displayDict['caption'] = 'Histogram of the number of nights between consecutive visits to a ' \
                             'given point on the sky, considering separations between %d and %d.' \
                             % (bins.min(), bins.max())
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, sqlConstraint, plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Median inter-night gap (each and all filters)
    metric = metrics.InterNightGapsMetric(metricName='Median Inter-Night Gap')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    for f in filterlist:
        if f is not 'all':
            sql = sqlConstraint + 'filter = "%s"' % f
            md = '%s band' % f
        else:
            sql = sqlConstraint
            md = 'all bands'
        if metadata is not None:
            md += metadata
        displayDict['caption'] = 'Median gap between nights with observations, %s.' % md
        displayDict['order'] = filterorder[f]
        plotDict = {'color': colors[f]}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, displayDict=displayDict,
                                 plotFuncs=subsetPlots, plotDict=plotDict,
                                 summaryMetrics=standardStats)
        bundleList.append(bundle)

    # Maximum inter-night gap (in each and all filters).
    metric = metrics.InterNightGapsMetric(metricName='Max Inter-Night Gap', reduceFunc=np.max)
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    for f in filterlist:
        if f is not 'all':
            sql = sqlConstraint + 'filter = "%s"' % f
            md = '%s band' % f
        else:
            sql = sqlConstraint
            md = 'all bands'
        if metadata is not None:
            md += metadata
        sql = sqlConstraint + 'filter = "%s"' % f
        displayDict['caption'] = 'Maximum gap between nights with observations, %s.' % md
        displayDict['order'] = filterorder[f] - 10
        plotDict = {'color': colors[f], 'percentileClip': 95.}
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=md, displayDict=displayDict,
                                 plotFuncs=subsetPlots, plotDict=plotDict, summaryMetrics=standardStats)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)