from __future__ import print_function
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
from .colMapDict import ColMapDict
from .common import standardSummaryMetrics

__all__ = ['glanceBundle']


def glanceBundle(colmap=None, runName='opsim', nside=64):
    """Generate a handy set of metrics that give a quick overview of how well a survey performed

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".
    nside : int, opt
        The nside for the healpix slicers. Default 64.
        
    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    filternames = ['u', 'g', 'r', 'i', 'z', 'y']
    sql_per_filt = ['%s="%s"' % (colmap['filter'], filtername) for filtername in filternames]
    sql_per_and_all_filters = [''] + sql_per_filt

    standardStats =  standardSummaryMetrics()
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Super basic things
    displayDict = {'group': 'Basic Stats', 'order': 1}
    sql = ''
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
                                   lonCol=colmap['az'], useCache=False)
    stacker = stackers.ZenithDistStacker(altCol=colmap['alt'])
    metric = metrics.CountMetric(colmap['mjd'], metricName='Nvisits as function of Alt/Az')
    plotFuncs = [plots.LambertSkyMap()]
    for sql in sql_per_and_all_filters:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs,
                                            displayDict=displayDict, stackerList=[stacker])
        bundleList.append(bundle)

    # Things to check per night
    # Open Shutter per night
    displayDict = {'group': 'Pointing Efficency', 'order':2}
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    sql = None
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

    # Slewtime distribution
    slicer = slicers.OneDSlicer(sliceColName=colmap['slewtime'], binsize=2)
    metric = metrics.CountMetric(col=colmap['slewtime'], metricName='Slew Time Histogram')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict={'logScale': True, 'ylabel': 'Count'},
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # A few basic maps
    # Number of observations, coadded depths
    displayDict = {'group': 'Basic Maps', 'order': 3}
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap['dec'], lonCol=colmap['ra'])
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
                                             dateCol=colmap['mjd'])
    stackerList.append(stacker)

    # Maybe parallax and proper motion, fraction of visits in a good pair for SS
    displayDict['caption'] = r'Parallax precision of an $r=20$ flat SED star'
    metric = metrics.ParallaxMetric(m5Col=colmap['fiveSigmaDepth'],
                                    filterCol=colmap['filter'],
                                    seeingCol=colmap['seeingGeom'])
    sql = ''
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
    sql = 'filter="g" or filter="r" or filter="i"'
    metric = metrics.PairFractionMetric(timesCol=colmap['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=subsetPlots,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    for b in bundleList:
        b.runName = runName

    bd = metricBundles.makeBundlesDictFromList(bundleList)
    return bd

