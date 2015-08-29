#! /usr/bin/env python
import os, sys, argparse
import numpy as np
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import healpy as hp
import warnings

import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils
from mafContrib import PhaseGapMetric

def makeBundleList(dbFile, runName=None, nside=128,
                   lonCol='fieldRA', latCol='fieldDec'):


    opsimdb = utils.connectOpsimDb(dbFile)
    runLength = opsimdb.fetchRunLength()
    propids, propTags = opsimdb.fetchPropInfo()
    bundleList=[]
    filters = ['u','g','r','i','z','y']
    sqls = ['filter = "%s"' % f for f in filters]
    sqls.append('')

    interGroup='A: Inter-Night'
    intraGroup = 'B: Intra-Night'
    maxGapGroup = 'C: Max Gap'
    maxgapGroup = 'D: Max Day Gap'
    phaseGroup = 'E: Max Phase Gap'
    transgroup = 'F: SN Ia'
    altAzGroup = 'G: Alt Az'
    rangeGroup = 'H: Range of Dates'

    # Median inter-night gap (each and all filters)
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    metric = metrics.InterNightGapsMetric(metricName='Median Inter-Night Gap')
    displayDict = {'group':interGroup, 'caption':'Median gap between days'}
    for sql in sqls:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict, runName=runName)
        bundleList.append(bundle)

    # Median intra-night gap in the r and all bands
    metric = metrics.IntraNightGapsMetric(metricName='Median Intra-Night Gap')
    displayDict = {'group':intraGroup, 'caption':'Median gap within a night.'}
    for sql in sqls:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict, runName=runName)
        bundleList.append(bundle)

    # Max inter-night gap in r and all bands
    dslicer = slicers.HealpixSlicer(nside=nside, lonCol='ditheredRA', latCol='ditheredDec')
    metric = metrics.InterNightGapsMetric(metricName='Max Inter-Night Gap', reduceFunc=np.max)
    displayDict = {'group':maxGapGroup, 'caption':'Max gap between nights'}
    plotDict = {'percentileClip':95.}
    for sql in sqls:
        bundle = metricBundles.MetricBundle(metric, dslicer, sql, displayDict=displayDict,
                                            plotDict=plotDict, runName=runName)
        bundleList.append(bundle)

    # largest phase gap for periods
    periods = [0.1,1.0,10.,100.]
    sqls = ['filter = "u"', 'filter="r"', 'filter="g" or filter="r" or filter="i" or filter="z"',
            '']

    for sql in sqls:
        for period in periods:
            displayDict = {'group':phaseGroup, 'subgroup':'period=%.2f days'%period,
                           'caption':'Maximum phase gaps'}
            metric = PhaseGapMetric(nPeriods=1, periodMin=period, periodMax=period,
                                    metricName='PhaseGap, %.1f'%period)
            bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict, runName=runName)
            bundleList.append(bundle)

    # SNe metrics from UK workshop.
    peaks = {'uPeak':25.9, 'gPeak':23.6, 'rPeak':22.6, 'iPeak':22.7, 'zPeak':22.7,'yPeak':22.8}
    peakTime = 15.
    transDuration = peakTime+30. # Days
    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          metricName='SNDetection',**peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected in any filter'
    displayDict={'group':transgroup,  'subgroup':'Detected', 'caption':caption}
    sqlconstraint = ''
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          nPrePeak=1, metricName='SNAlert', **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected pre-peak in any filter'
    displayDict={'group':transgroup,  'subgroup':'Detected on the rise', 'caption':caption}
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.,
                                     transDuration=transDuration, peakTime=peakTime,
                                     surveyDuration=runLength, metricName='SNLots',
                                     nFilters=3, nPrePeak=3, nPerLC=2, **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are observed 6 times, 3 pre-peak, 3 post-peak, with observations in 3 filters'
    displayDict={'group':transgroup,  'subgroup':'Well observed', 'caption':caption}
    sqlconstraint = 'filter="r" or filter="g" or filter="i" or filter="z" '
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)

    # Full range of dates:
    metric = metrics.FullRangeMetric(col='expMJD')
    plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
    caption='Time span of survey.'
    sqlconstraint = ''
    plotDict={}
    displayDict={'group':rangeGroup, 'caption':caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)
    for filt in filters:
        for propid in propids:
            md = '%s, %s' % (filt, propids[propid])
            sql = 'filter="%s" and propID=%i' % (filt,propid)
            bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                                metadata=md, plotFuncs=plotFuncs,
                                                displayDict=displayDict, runName=runName)
            bundleList.append(bundle)




    # Alt az plots
    slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
    metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
    plotDict = {}
    plotFuncs = [plots.LambertSkyMap()]
    displayDict = {'group':altAzGroup, 'caption':'Alt Az pointing distribution'}
    for filt in filters:
        for propid in propids:
            md = '%s, %s' % (filt, propids[propid])
            sql = 'filter="%s" and propID=%i' % (filt,propid)
            bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                                plotFuncs=plotFuncs, metadata=md,
                                                displayDict=displayDict, runName=runName)
            bundleList.append(bundle)


    return metricBundles.makeBundlesDictFromList(bundleList)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the science performance metrics')
    parser.add_argument('dbFile', type=str, default=None,help="full file path to the opsim sqlite file")

    parser.add_argument("--outDir",type=str, default='./Out', help='Output directory for MAF outputs. Default "Out"')
    parser.add_argument("--nside", type=int, default=128,
                        help="Resolution to run Healpix grid at (must be 2^x). Default 128.")
    parser.add_argument("--lonCol", type=str, default='fieldRA',
                        help="Column to use for RA values (can be a stacker dither column). Default=fieldRA.")
    parser.add_argument("--latCol", type=str, default='fieldDec',
                        help="Column to use for Dec values (can be a stacker dither column). Default=fieldDec.")
    parser.add_argument('--benchmark', type=str, default='design',
                        help="Can be 'design' or 'requested'")
    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values from disk and re-plot them.")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    # Build metric bundles.
    head, filename = os.path.split(args.dbFile)
    runName = filename.replace('_sqlite.db','')
    bundleDict = makeBundleList(args.dbFile, nside=args.nside,
                                lonCol=args.lonCol, latCol=args.latCol, runName=runName)

    # Set up / connect to resultsDb.
    resultsDb = db.ResultsDb(outDir=args.outDir)
    # Connect to opsimdb.
    opsdb = utils.connectOpsimDb(args.dbFile)
    # Set up metricBundleGroup.
    group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
                                            outDir=args.outDir, resultsDb=resultsDb)
    # Read or run to get metric values.
    if args.plotOnly:
        group.readAll()
    else:
        group.runAll()
    # Make plots.
    group.plotAll()

    # Get config info and write to disk.
    utils.writeConfigs(opsdb, args.outDir)

    print "Finished cadence check metric calculations."
