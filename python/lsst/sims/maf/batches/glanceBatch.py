from __future__ import print_function
import warnings
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
from .colMapDict import ColMapDict
from .common import standardSummary
from .slewBatch import slewBasics

__all__ = ['glanceBatch']


def glanceBatch(colmap=None, runName='opsim',
                nside=64, filternames=('u', 'g', 'r', 'i', 'z', 'y'),
                sqlConstraint=None):
    """Generate a handy set of metrics that give a quick overview of how well a survey performed.
    This is a meta-set of other batches, to some extent.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        The nside for the healpix slicers. Default 64.
    filternames : list of str, opt
        The list of individual filters to use when running metrics.
        Default is ('u', 'g', 'r', 'i', 'z', 'y').
        There is always an all-visits version of the metrics run as well.
    sqlConstraint : str or None, opt
        Additional SQL constraint to apply to all metrics.

    Returns
    -------
    metricBundleDict
    """
    if isinstance(colmap, str):
        raise ValueError('colmap must be a dictionary, not a string')

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    if sqlConstraint is None:
        sqlC = ''
    else:
        sqlC = '(%s) and' % sqlConstraint

    sql_per_filt = ['%s %s="%s"' % (sqlC, colmap['filter'], filtername) for filtername in filternames]
    sql_per_and_all_filters = [sqlConstraint] + sql_per_filt

    standardStats = standardSummary()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Super basic things
    displayDict = {'group': 'Basic Stats', 'order': 1}
    sql = sqlConstraint
    slicer = slicers.UniSlicer()
    # Length of Survey
    metric = metrics.FullRangeMetric(col=colmap['mjd'], metricName='Length of Survey (days)')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total number of filter changes
    metric = metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total open shutter fraction
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total effective exposure time
    metric = metrics.TeffMetric(m5Col=colmap['fiveSigmaDepth'],
                                filterCol=colmap['filter'], normed=True)
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    # Number of observations, all and each filter
    metric = metrics.CountMetric(col=colmap['mjd'], metricName='Number of Exposures')
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    # The alt/az plots of all the pointings
    slicer = slicers.HealpixSlicer(nside=nside, latCol='zenithDistance',
                                   lonCol=colmap['az'], latLonDeg=colmap['raDecDeg'], useCache=False)
    stacker = stackers.ZenithDistStacker(altCol=colmap['alt'], degrees=colmap['raDecDeg'])
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits as function of Alt/Az')
    plotFuncs = [plots.LambertSkyMap()]
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs,
                                            displayDict=displayDict, stackerList=[stacker])
        bundleList.append(bundle)

    # Things to check per night
    # Open Shutter per night
    displayDict = {'group': 'Pointing Efficency', 'order': 2}
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    sql = sqlConstraint
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # Number of filter changes per night
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    metric = metrics.NChangesMetric(col=colmap['filter'], orderBy=colmap['mjd'],
                                    metricName='Filter Changes')
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # A few basic maps
    # Number of observations, coadded depths
    displayDict = {'group': 'Basic Maps', 'order': 3}
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'],
                                   latLonDeg=colmap['raDecDeg'])
    metric = metrics.CountMetric(col=colmap['mjd'])
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=standardStats, displayDict=displayDict)
        bundleList.append(bundle)

    metric = metrics.Coaddm5Metric(m5Col=colmap['fiveSigmaDepth'])
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=standardStats, displayDict=displayDict)
        bundleList.append(bundle)

    # Checking a few basic science things
    # Maybe check astrometry, observation pairs, SN
    displayDict = {'group': 'Science', 'subgroup': 'Astrometry', 'order': 4}

    stackerList = []
    stacker = stackers.ParallaxFactorStacker(raCol=colmap['ra'],
                                             decCol=colmap['dec'],
                                             degrees=colmap['raDecDeg'],
                                             dateCol=colmap['mjd'])
    stackerList.append(stacker)

    # Maybe parallax and proper motion, fraction of visits in a good pair for SS
    displayDict['caption'] = r'Parallax precision of an $r=20$ flat SED star'
    metric = metrics.ParallaxMetric(m5Col=colmap['fiveSigmaDepth'],
                                    filterCol=colmap['filter'],
                                    seeingCol=colmap['seeingGeom'])
    sql = sqlConstraint
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict, stackerList=stackerList)
    bundleList.append(bundle)
    displayDict['caption'] = r'Proper motion precision of an $r=20$ flat SED star'
    metric = metrics.ProperMotionMetric(m5Col=colmap['fiveSigmaDepth'],
                                        mjdCol=colmap['mjd'],
                                        filterCol=colmap['filter'],
                                        seeingCol=colmap['seeingGeom'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    # Solar system stuff
    displayDict['caption'] = 'Fraction of observations that are in pairs'
    displayDict['subgroup'] = 'Solar System'
    sql = '%s (filter="g" or filter="r" or filter="i")' % sqlC
    metric = metrics.PairFractionMetric(mjdCol=colmap['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)

    # Add basic slew stats.
    try:
        slewDict = slewBasics(colmap=colmap, runName=runName)
    except KeyError as e:
        warnings.warn('Could not add slew stats: missing required key %s from colmap' % (e))

    bd = metricBundles.makeBundlesDictFromList(bundleList)
    bd.update(slewDict)
    return bd

