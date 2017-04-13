#!/usr/bin/env python

from __future__ import print_function
import argparse
import matplotlib
# Set matplotlib backend (to create plots where DISPLAY is not set).
matplotlib.use('Agg')
import matplotlib.cm as cm
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots
import lsst.sims.maf.utils as utils


def makeBundleList(dbFile, night=1, nside=64, latCol='ditheredDec', lonCol='ditheredRA'):
    """
    Make a bundleList of things to run
    """

    # Construct sql queries for each filter and all filters
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    sqls = ['night=%i and filter="%s"' % (night, f)for f in filters]
    sqls.append('night=%i' % night)

    bundleList = []
    plotFuncs_lam = [plots.LambertSkyMap()]

    reg_slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    altaz_slicer = slicers.HealpixSlicer(nside=nside, latCol='zenithDistance',
                                         lonCol='azimuth', useCache=False)

    unislicer = slicers.UniSlicer()
    for sql in sqls:

        # Number of exposures
        metric = metrics.CountMetric('expMJD', metricName='N visits')
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.CountMetric('expMJD', metricName='N visits alt az')
        bundle = metricBundles.MetricBundle(metric, altaz_slicer, sql, plotFuncs=plotFuncs_lam)
        bundleList.append(bundle)

        metric = metrics.MeanMetric('expMJD', metricName='Mean Visit Time')
        bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
        bundleList.append(bundle)
        metric = metrics.MeanMetric('expMJD', metricName='Mean Visit Time alt az')
        bundle = metricBundles.MetricBundle(metric, altaz_slicer, sql, plotFuncs=plotFuncs_lam)
        bundleList.append(bundle)

        metric = metrics.CountMetric('expMJD', metricName='N_visits')
        bundle = metricBundles.MetricBundle(metric, unislicer, sql)
        bundleList.append(bundle)

        # Need pairs in window to get a map of how well it gathered SS pairs.

    # Moon phase.

    metric = metrics.NChangesMetric(col='filter', metricName='Filter Changes')
    bundle = metricBundles.MetricBundle(metric, unislicer, 'night=%i' % night)
    bundleList.append(bundle)

    metric = metrics.OpenShutterFractionMetric()
    bundle = metricBundles.MetricBundle(metric, unislicer, 'night=%i' % night)
    bundleList.append(bundle)

    metric = metrics.MeanMetric('slewTime')
    bundle = metricBundles.MetricBundle(metric, unislicer, 'night=%i' % night)
    bundleList.append(bundle)

    metric = metrics.MinMetric('slewTime')
    bundle = metricBundles.MetricBundle(metric, unislicer, 'night=%i' % night)
    bundleList.append(bundle)

    metric = metrics.MaxMetric('slewTime')
    bundle = metricBundles.MetricBundle(metric, unislicer, 'night=%i' % night)
    bundleList.append(bundle)

    # Make plots of the solar system pairs that were taken in the night
    metric = metrics.PairMetric()
    sql = 'night=%i and (filter ="r" or filter="g" or filter="i")' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    metric = metrics.PairMetric(metricName='z Pairs')
    sql = 'night=%i and filter="z"' % night
    bundle = metricBundles.MetricBundle(metric, reg_slicer, sql)
    bundleList.append(bundle)

    # Plot up each visit
    metric = metrics.NightPointingMetric()
    slicer = slicers.UniSlicer()
    sql = sql = 'night=%i' % night
    plotFuncs = [plots.NightPointingPlotter()]
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs)
    bundleList.append(bundle)

    return metricBundles.makeBundlesDictFromList(bundleList)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Python script to generate a report on a single night.')
    parser.add_argument('dbFile', type=str, default=None, help="full file path to the opsim sqlite file")
    parser.add_argument("--outDir", type=str, default='./Out', help='Output directory for MAF outputs.' +
                        ' Default "Out"')
    parser.add_argument("--nside", type=int, default=64,
                        help="Resolution to run Healpix grid at (must be 2^x). Default 64.")
    parser.add_argument("--lonCol", type=str, default='fieldRA',
                        help="Column to use for RA values (can be a stacker dither column)." +
                        " Default=fieldRA.")
    parser.add_argument("--latCol", type=str, default='fieldDec',
                        help="Column to use for Dec values (can be a stacker dither column)." +
                        " Default=fieldDec.")
    parser.add_argument('--night', type=int, default=1)

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    bundleDict = makeBundleList(args.dbFile, nside=args.nside, lonCol=args.lonCol, latCol=args.latCol,
                                night=args.night)

    # Set up / connect to resultsDb.
    resultsDb = db.ResultsDb(outDir=args.outDir)
    # Connect to opsimdb.
    opsdb = utils.connectOpsimDb(args.dbFile)

    # Set up metricBundleGroup.
    group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
    group.runAll()
    group.plotAll()

