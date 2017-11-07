"""Sets of metrics to look at time between visits/pairs, etc.
"""
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['intraNight']


def intraNight(colmap=None, runName='opsim', nside=64, sqlConstraint=None):
    """Generate a set of statistics about the pair/triplet/etc. rate within a night.

    Parameters
    ----------
    colmap : dict or None, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        Nside for the healpix slicer. Default 64.
    sqlConstraint : str or None, opt
        Additional sql constraint to apply to all metrics.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []
    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    if sqlConstraint is not None:
        sqlC = '(%s) and ' % sqlConstraint
    else:
        sqlC = ''

    # Look for the fraction of visits in gri where there are pairs within dtMin/dtMax.
    displayDict = {'group': 'IntraNight', 'subgroup': 'Pairs', 'caption': None, 'order': -1}
    sql = '%s (filter="g" or filter="r" or filter="i")' % sqlC
    metadata = 'gri'
    dtMin = 15.0
    dtMax = 60.0
    metric = metrics.PairFractionMetric(timeCol=colmap['mjd'], minGap=dtMin, maxGap=dtMax,
                                        metricName='Fraction of visits in pairs (%.0f-%.0f min)' % (dtMin,
                                                                                                    dtMax))
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['caption'] = 'Fraction of %s visits that have a paired visit' \
                             'between %.1f and %.1f minutes away. ' % (metadata, dtMin, dtMax)
    displayDict['caption'] += 'If all visits were in pairs, this fraction would be 1.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Look at the fraction of visits which have another visit within dtMax.
    dtMax = 50.0
    metric = metrics.NRevisitsMetric(timeCol=colmap['mjd'], dT=dtMax, normed=True,
                                     metricName='Fraction of visits with a revisit < %.0f min' % dtMax)
    displayDict['caption'] = 'Fraction of %s visits that have another visit ' \
                             'within %.1f min. ' % (metadata, dtMax)
    displayDict['caption'] += 'If all visits were in pairs (only), this fraction would be 0.5.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata, summaryMetrics=standardStats,
                             plotFuncs=subsetPlots, displayDict=displayDict)
    bundleList.append(bundle)

    # Histogram of the time between quick revisits.
    binMin = 0
    binMax = 120.
    binsize = 5.
    bins_metric = np.arange(binMin / 60.0 / 24.0, (binMax + binsize) / 60. / 24., binsize / 60. / 24.)
    bins_plot = bins_metric * 24.0 * 60.0
    sql = sqlConstraint
    metric = metrics.TgapsMetric(bins=bins_metric, timesCol=colmap['mjd'], metricName='DeltaT Histogram')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    plotDict = {'bins': bins_plot, 'xlabel': 'dT (minutes)'}
    metadata = 'All filters'
    displayDict['caption'] = 'Histogram of the time between consecutive visits to a given point ' \
                             'on the sky, considering visits between %.1f and %.1f minutes' % (binMin, binMax)
    displayDict['order'] += 1
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                             displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def interNight(colmap=None, runName='opsim', nside=64):
    """Generate a set of statistics about the gaps between nights of observations.

     Parameters
     ----------
     colmap : dict or None opt
         A dictionary with a mapping of column names. Default will use OpsimV4 column names.
     runName : str, opt
         The name of the simulated survey. Default is "opsim".
     nside : int, opt
         Nside for the healpix slicer. Default 64.

     Returns
     -------
     metricBundleDict
     """
    pass

