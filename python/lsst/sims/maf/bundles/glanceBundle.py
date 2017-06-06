from __future__ import print_function
from builtins import zip
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils

__all__ = ['glanceBundle']


def glanceBundle(colmap_dict=None, nside=64):
    """Generate a handy set of metrics that give a quick overview of how well a survey performed

    Parameters
    ----------
    colmap_dict : dict
        A dictionary with a mapping of column names.

    Returns
    -------
    metricBundleDict
    """

    if colmap_dict is None:
        colmap_dict = utils.opsimColMapDict()

    bundleList = []

    filternames = ['u', 'g', 'r', 'i', 'z', 'y']
    sql_per_filt = ['%s="%s"' % (colmap_dict['filter'], filtername) for filtername in filternames]
    sql_per_plus = ['']
    sql_per_plus.extend(sql_per_filt)

    standardStats = [metrics.MeanMetric(),
                     metrics.RmsMetric(), metrics.MedianMetric(), metrics.CountMetric(),
                     metrics.MaxMetric(), metrics.MinMetric(),
                     metrics.NoutliersNsigmaMetric(metricName='N(+3Sigma)', nSigma=3),
                     metrics.NoutliersNsigmaMetric(metricName='N(-3Sigma)', nSigma=-3.)]

    # Super basic things
    displayDict = {'group': 'Basic Stats', 'order': 1}
    sql = ''
    slicer = slicers.UniSlicer()
    # Length of Survey
    metric = metrics.FullRangeMetric(col=colmap_dict['mjd'], metricName='Length of Survey (days)')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total number of filter changes
    metric = metrics.NChangesMetric(col=colmap_dict['filter'], orderBy=colmap_dict['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Total open shutter fraction
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap_dict['slewtime'],
                                               expTimeCol=colmap_dict['exptime'],
                                               visitTimeCol=colmap_dict['visittime'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
    bundleList.append(bundle)

    # Number of observations, all and each filter
    metric = metrics.CountMetric(col=colmap_dict['mjd'], metricName='Number of Exposures')
    for sql in sql_per_plus:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict)
        bundleList.append(bundle)

    # The alt/az plots of all the pointings
    slicer = slicers.HealpixSlicer(nside=nside, latCol='zenithDistance',
                                   lonCol=colmap_dict['az'], useCache=False)
    stacker = stackers.ZenithDistStacker(altCol=colmap_dict['alt'])
    sql = ''
    metric = metrics.CountMetric(colmap_dict['mjd'], metricName='Nvisits as function of Alt/Az')
    plotFuncs = [plots.LambertSkyMap()]

    # per filter
    for sql in sql_per_plus:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs,
                                            displayDict=displayDict, stackerList=[stacker])
        bundleList.append(bundle)

    # Things to check per night
    # Open Shutter per night
    displayDict = {'group': 'Pointing Efficency', 'order':2}
    slicer = slicers.OneDSlicer(sliceColName=colmap_dict['night'], binsize=1)
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap_dict['slewtime'],
                                               expTimeCol=colmap_dict['exptime'],
                                               visitTimeCol=colmap_dict['visittime'])
    sql = ''
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # Number of filter changes per night
    metric = metrics.NChangesMetric(col=colmap_dict['filter'], orderBy=colmap_dict['mjd'], 
                                    metricName='Filter Changes')
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # Slewtime distribution
    slicer = slicers.OneDSlicer(sliceColName=colmap_dict['slewtime'], binsize=2)
    metric = metrics.CountMetric(col=colmap_dict['slewtime'], metricName='Slew Time Histogram')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict={'logScale': True, 'ylabel': 'Count'},
                                        summaryMetrics=standardStats, displayDict=displayDict)
    bundleList.append(bundle)

    # A few basic maps
    # Number of observations, coadded depths
    displayDict = {'group': 'Basic Maps', 'order': 3}
    slicer = slicers.HealpixSlicer(nside=nside, latCol=colmap_dict['dec'], lonCol=colmap_dict['ra'])
    metric = metrics.CountMetric(col=colmap_dict['mjd'])
    for sql in sql_per_plus:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=standardStats, displayDict=displayDict)
        bundleList.append(bundle)

    metric = metrics.Coaddm5Metric(m5Col=colmap_dict['fiveSigmaDepth'])
    for sql in sql_per_filt:
        bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                            summaryMetrics=standardStats, displayDict=displayDict)
        bundleList.append(bundle)

    # Checking a few basic science things
    # Maybe check astrometry, observation pairs, SN 
    displayDict = {'group': 'Science', 'subgroup': 'Astrometry', 'order': 4}

    stackerList = []
    stacker = stackers.ParallaxFactorStacker(raCol=colmap_dict['ra'],
                                             decCol=colmap_dict['dec'],
                                             dateCol=colmap_dict['mjd'])
    stackerList.append(stacker)

    # Maybe parallax and proper motion, fraction of visits in a good pair for SS, and SN detection & LC sampling? 
    displayDict['caption'] = r'Parallax precision of an $r=20$ flat SED star'
    metric = metrics.ParallaxMetric(m5Col=colmap_dict['fiveSigmaDepth'],
                                    mjdCol=colmap_dict['mjd'],
                                    filterCol=colmap_dict['filter'],
                                    seeingCol=colmap_dict['seeingGeom'])
    sql = ''
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        displayDict=displayDict, stackerList=stackerList)
    bundleList.append(bundle)
    displayDict['caption'] = r'Proper motion precision of an $r=20$ flat SED star'
    metric = metrics.ProperMotionMetric(m5Col=colmap_dict['fiveSigmaDepth'],
                                        mjdCol=colmap_dict['mjd'],
                                        filterCol=colmap_dict['filter'],
                                        seeingCol=colmap_dict['seeingGeom'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    # Solar system stuff
    displayDict['caption'] = 'Fraction of observations that are in pairs'
    displayDict['subgroup'] = 'Solar System'
    sql = 'filter="g" or filter="r" or filter="i"'
    metric = metrics.PairFractionMetric(timesCol=colmap_dict['mjd'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql,
                                        displayDict=displayDict)
    bundleList.append(bundle)


    bd = metricBundles.makeBundlesDictFromList(bundleList)
    return bd

