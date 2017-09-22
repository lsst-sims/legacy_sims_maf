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


def intraNight(colmap=None, runName='opsim', nside=64):
    """Generate a set of statistics about the pair/triplet/etc. rate within a night.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        Nside for the healpix slicer. Default 64.

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []
    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]


    # Look for the fraction of visits in gri where there are pairs.
    displayDict = {'group': 'IntraNight', 'subgroup': 'Pairs', 'caption': None, 'order': -1}
    sql = 'filter="g" or filter="r" or filter="i"'
    metadata='gri pairs'
    metric = metrics.PairFractionMetric(timesCol=colmap['mjd'])
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    displayDict['caption'] = 'Fraction of gri observations that are in pairs.'
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
    sql = ''
    metric = metrics.TgapsMetric(bins=bins_metric, metricName='dT visits')
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    plotDict = {'bins': bins_plot, 'xlabel': 'dT (minutes)'}
    metadata = 'DeltaT histogram'
    displayDict['caption'] = 'Histogram of the time between consecutive revisits ' \
                             '(between %.1f and %.1f minutes), over entire sky.' % (binMin, binMax)
    displayDict['order'] += 1
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                            displayDict=displayDict, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
