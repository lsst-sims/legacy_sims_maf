#!/usr/bin/env python

from __future__ import print_function, division

import os
import argparse
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Needs MafSSO branch from sims_maf.
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils

# Assumes you have already created observation file,
# using make_movingObject_obs.py.

npReduce = np.mean

def setupSlicer(orbitFile, Hrange, obsFile=None):
    # Read the orbit file and set the H values for the slicer.
    slicer = slicers.MoObjSlicer()
    slicer.readOrbits(orbitFile, Hrange=Hrange)
    if obsFile is not None:
        slicer.readObs(obsFile)
    return slicer


def setupMetrics(slicer, runName, metadata, albedo, Hmark, mParams):
    # Set up the metrics.
    allBundles = {}

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce}
    summaryMetrics = [metrics.MoCompletenessMetric(),
                      metrics.MoCumulativeCompletenessMetric()]
    plotFuncs = [plots.MetricVsH()]

    # Set up discovery metrics; calculate at all years.
    allBundles['discovery'] = {}
    for nyr in mParams['nyears']:
        # First standard SNR / probabilistic visibility (SNR~5)
        # 3 pairs in 15
        constraint = 'night < %d' %(nyr * 365 + 1)
        md = metadata + ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90./60./24.,
                                         nNightsPerWindow=3, tWindow=15)
        childMetrics = {'Time': metrics.Discovery_TimeMetric(metric, i=0, tStart=59580),
                        'N_Chances': metrics.Discovery_N_ChancesMetric(metric)}
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 pairs in 30
        md = metadata + ' year %d, 3 pairs in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 4 pairs in 20
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=4, tWindow=20)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 triplets in 30
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=3, tMin=0, tMax=120. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 quads in 30
        md = metadata + ' year %d, 3 quads in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=4, tMin=0, tMax=150. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # Play with SNR.  SNR=4
        # 3 pairs in 15
        constraint = 'night < %d' % (nyr * 365 + 1)
        md = metadata + ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=15, snrLimit=4)
        childMetrics = {'Time': metrics.Discovery_TimeMetric(metric, i=0, tStart=59580),
                        'N_Chances': metrics.Discovery_N_ChancesMetric(metric)}
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 pairs in 30
        md = metadata + ' year %d, 3 pairs in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=4)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 4 pairs in 20
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=4, tWindow=20, snrLimit=4)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 triplets in 30
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=3, tMin=0, tMax=120. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=4)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 quads in 30
        md = metadata + ' year %d, 3 quads in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=4, tMin=0, tMax=150. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=4)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # Play with SNR.  SNR=3
        # 3 pairs in 15
        constraint = 'night < %d' % (nyr * 365 + 1)
        md = metadata + ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=15, snrLimit=3)
        childMetrics = {'Time': metrics.Discovery_TimeMetric(metric, i=0, tStart=59580),
                        'N_Chances': metrics.Discovery_N_ChancesMetric(metric)}
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 pairs in 30
        md = metadata + ' year %d, 3 pairs in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=3)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 4 pairs in 20
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=4, tWindow=20, snrLimit=3)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 triplets in 30
        md = metadata + ' year %d, 4 pairs in 20 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=3, tMin=0, tMax=120. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=3)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # 3 quads in 30
        md = metadata + ' year %d, 3 quads in 30 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=4, tMin=0, tMax=150. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=30, snrLimit=3)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # SNR = 0
        # 3 pairs in 15
        constraint = 'night < %d' % (nyr * 365 + 1)
        md = metadata + ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=3, tWindow=15, snrLimit=0)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # Play with strategy.
        # Single detection.
        constraint = 'night < %d' % (nyr * 365 + 1)
        md = metadata + ' year %d, Single detection' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=1, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=1, tWindow=5)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle
        # Single pair of detections.
        constraint = 'night < %d' % (nyr * 365 + 1)
        md = metadata + ' year %d, Single pair' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                         nNightsPerWindow=1, tWindow=5)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=md,
                                    childMetrics=childMetrics,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        bundle.childBundles['N_Chances'].setSummaryMetrics(summaryMetrics)
        allBundles['discovery'][md] = bundle



    allBundles['nObs'] = {}
    constraint = None
    md = metadata
    plotDict = {'nxbins': 200, 'nybins': 200, 'ylabel': 'Number of observations (#)',
                'title': '%s: Number of observations %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric()
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['nObs'][md] = bundle

    allBundles['obsArc'] = {}
    constraint = None
    md = metadata
    plotDict = {'nxbins': 200, 'nybins': 200, 'ylabel': 'Observational Arc (days)',
                'title': '%s: Observational Arc Length %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric()
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['obsArc'][md] = bundle

    allBundles['ActivityTime'] = {}
    for w in mParams['windows']:
        constraint = None
        md = metadata + ' activity lasting %.0f days' % w
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.0f day window' % w}
        metricName = 'Chances of detecting activity lasting %.0f days' % w
        metric = metrics.ActivityOverTimeMetric(w, metricName=metricName)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        allBundles['ActivityTime'][w] = bundle

    allBundles['ActivityPeriod'] = {}
    for b in mParams['bins']:
        constraint = None
        md = metadata + ' activity lasting %.2f of period' % (b/360.)
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.2f deg window' % b}
        metricName = 'Chances of detecting activity lasting %.2f of the period' % (b/360.)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs)
        allBundles['ActivityPeriod'][b] = bundle

    allBundles['lightcurveInversion'] = {}
    constraint = None
    md = metadata
    plotDict = {'nxbins': 200, 'nybins': 200,
                'yMin': 0, 'yMax': 1, 'ylabel': 'Fraction of objects',
                'title': '%s: Fraction with potential lightcurve inversion %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveInversionMetric(snrLimit=20, nObs=100, nDays=5*365)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['lightcurveInversion'][md] = bundle

    allBundles['colorDetermination'] = {}
    snrLimit = 10
    nHours = 2.0
    nPairs = 1
    constraint = None
    md = metadata + ' u-g color'
    plotDict = {'nxbins': 200, 'nybins': 200, 'label': md,
                'title': '%s: Fraction with potential u-g color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='u', bTwo='g')
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['colorDetermination'][md] = bundle

    constraint = None
    md = metadata + ' g-r color'
    plotDict = {'nxbins': 200, 'nybins': 200, 'label': md,
                'title': '%s: Fraction with potential g-r color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='g', bTwo='r')
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['colorDetermination'][md] = bundle

    constraint = None
    md = metadata + ' r-i color'
    plotDict = {'nxbins': 200, 'nybins': 200, 'label': md,
                'title': '%s: Fraction with potential r-i color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='r', bTwo='i')
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['colorDetermination'][md] = bundle

    constraint = None
    md = metadata + ' i-z color'
    plotDict = {'nxbins': 200, 'nybins': 200, 'label': md,
                'title': '%s: Fraction with potential i-z color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='i', bTwo='z')
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['colorDetermination'][md] = bundle

    constraint = None
    md = metadata + ' z-y color'
    plotDict = {'nxbins': 200, 'nybins': 200, 'label': md,
                'title': '%s: Fraction with potential z-y color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='z', bTwo='y')
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None)
    allBundles['colorDetermination'][md] = bundle

    return allBundles


def runMetrics(allBundles, outDir, resultsDb=None, Hmark=None):
    # Run metrics, write basic output in outDir.
    # Un-nest dictionaries to run all at once.
    bundleDict = {}
    count = 0
    for k, v in allBundles.iteritems():
        if isinstance(v, dict):
            for k2, v2 in v.iteritems():
                bundleDict[count] = v2
                count += 1
        else:
            bundleDict[count] = v
            count += 1
    print("Counted %d top-level metric bundles." % count)

    bg = mmb.MoMetricBundleGroup(bundleDict, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    bg.writeAll()
    bg.summaryAll()
    allBundles = addAllCompletenessBundles(allBundles, Hmark, resultsDb)
    return allBundles


def addAllCompletenessBundles(allBundles, Hmark, resultsDb):
    # Add completeness bundles and write completeness at Hmark to resultsDb.
    allBundles['DifferentialCompleteness'] = {}
    allBundles['CumulativeCompleteness'] = {}
    k = 'discovery'
    for md in allBundles[k]:
        b = allBundles[k][md].childBundles['N_Chances']
        allBundles['DifferentialCompleteness'][md] = \
            makeCompletenessBundle(b, summaryName='Completeness', Hmark=Hmark, resultsDb=resultsDb)
        allBundles['CumulativeCompleteness'][md] = \
            makeCompletenessBundle(b, summaryName='CumulativeCompleteness', Hmark=Hmark, resultsDb=resultsDb)
    return allBundles

def makeCompletenessBundle(bundle, summaryName='CumulativeCompleteness',
                           Hmark=None, resultsDb=None):
    # Make a 'mock' metric bundle from a bundle which had the
    # MoCompleteness or MoCumulativeCompleteness summary metrics run.
    # This lets us use a normal plotHandler to generate combined plots.
    try:
        bundle.summaryValues[summaryName]
    except (TypeError, KeyError):
        if summaryName == 'Completeness':
            metric = metrics.MoCompletenessMetric()
        else:
            metric = metrics.MoCumulativeCompletenessMetric()
        bundle.setSummaryMetrics(metric)
        bundle.computeSummaryStats(resultsDb)
    completeness = ma.MaskedArray(data=bundle.summaryValues[summaryName]['value'],
                                  mask=np.zeros(len(bundle.summaryValues[summaryName]['value'])),
                                  fill_value=0)
    mb = mmb.MoMetricBundle(metrics.MoCompletenessMetric(metricName=summaryName),
                            bundle.slicer, constraint=None, metadata=bundle.metadata,
                            runName=bundle.runName)
    plotDict = {}
    plotDict.update(bundle.plotDict)
    plotDict['label'] = bundle.metadata
    mb.metricValues = completeness
    if Hmark is not None:
        metric = metrics.ValueAtHMetric(Hmark=Hmark)
        mb.setSummaryMetrics(metric)
        mb.computeSummaryStats(resultsDb)
        val = mb.summaryValues['Value At H=%.1f' % Hmark]
        if summaryName == 'Completeness':
            plotDict['label'] += ' : @ H(=%.1f) = %.1f%s' % (Hmark, val*100, '%')
        else:
            plotDict['label'] += ' : @ H(<=%.1f) = %.1f%s' % (Hmark, val*100, '%')
    mb.setPlotDict(plotDict)
    return mb


def plotMetrics(allBundles, outDir, metadata, runName, mParams, Hmark=None, resultsDb=None):
    # Make the plots.

    colorlist = ['cyan', 'g', 'burlywood', 'r', 'm', 'b', 'wheat']
    # Combine differential completeness summary values, over multiple years for discoveryChances.
    ph = plots.PlotHandler(outDir=outDir, savefig=True, resultsDb=resultsDb,
                           figformat='pdf', dpi=600, thumbnail=True)

    # Make basic plots of metric values that we want to see on a single plot.
    for k in ['nObs', 'obsArc', 'lightcurveInversion']:
        for md in allBundles[k]:
            if k == 'nObs':
                keyname = 'number of observations'
                subgroup = 'N Obs'
            elif k == 'obsArc':
                keyname = 'timespan from first to last observation'
                subgroup = 'Obs Arc'
            elif k == 'lightcurveInversion':
                keyname = 'likelihood of being able to invert the sparsely sampled lightcurve'
                subgroup = 'Lightcurve Inversion'
            else:
                keyname = k.capitalize()
                subgroup = k.capitalize()
            caption = '%s (across the population with a given H value) %s for an object ' \
                      % (npReduce.__name__.capitalize(), keyname)
            caption += 'as a function of H magnitude, for %s objects.' % (metadata)
            displayDict = {'group': 'B: Characterization', 'subgroup': subgroup, 'order': 0,
                           'caption': caption}
            allBundles[k][md].setDisplayDict(displayDict=displayDict, resultsDb=resultsDb)
            allBundles[k][md].plot(plotHandler=ph)

    years = [mParams['nyears'].max()]
    if max(years) > 10:
        years = [12] + years
    order = 0
    for year in years:
        # Plot the discovery chances at 'year', for different discovery strategies.
        k = 'discovery'
        strategies = ['3 pairs in 15 nights', '3 pairs in 30 nights',
                      '4 pairs in 20 nights', '3 triplets in 30 nights',
                      '3 quads in 30 nights']
        plotbundles = []
        plotDicts = []
        basePlotDict = {'title': '%s Discovery Chances at year %d - %s' % (runName, year, metadata),
                        'legendloc': 'upper right'}
        caption = 'Mean number of discovery chances, for various discovery strategies, ' \
                  'at the end of year %d.' % (year)
        displayDict = {'group':'A: Discovery', 'subgroup': 'Number of chances',
                       'order': order, 'caption': caption}
        for i, strategy in enumerate(strategies):
            md = '%s year %d, %s' % (metadata, year, strategy)
            plotbundles.append(allBundles[k][md].childBundles['N_Chances'])
            tmpPlotDict = {'color': colorlist[i], 'label': strategy}
            tmpPlotDict.update(basePlotDict)
            plotDicts.append(tmpPlotDict)
        ph.setMetricBundles(plotbundles)
        ph.jointMetadata = '%s Year %d: 3 pairs in 15 nights -- 3 quads in 30 nights' % (metadata, year)
        ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
        plt.close()

        # Plot the differential completeness at 'year', for different discovery strategies.
        k = 'DifferentialCompleteness'
        strategies = ['3 pairs in 15 nights', '3 pairs in 30 nights',
                      '4 pairs in 20 nights', '3 triplets in 30 nights', '3 quads in 30 nights']
        plotbundles = []
        plotDicts = []
        basePlotDict = {'title': '%s Differential Completeness at year %d - %s' % (runName, year, metadata),
                        'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                        'legendloc': 'lower left'}
        caption = 'Differential completeness (fraction of population with H=X) discovered at year %d,' % (year)
        caption += ' for a variety of discovery strategies.'
        displayDict = {'group': 'A: Discovery', 'subgroup': 'Differential Completeness',
                       'order': order, 'caption': caption}
        for i, strategy in enumerate(strategies):
            md = '%s year %d, %s' % (metadata, year, strategy)
            plotbundles.append(allBundles[k][md])
            tmpPlotDict = {'color': colorlist[i]}
            tmpPlotDict.update(basePlotDict)
            plotDicts.append(tmpPlotDict)
        ph.setMetricBundles(plotbundles)
        ph.jointMetadata = '%s Year %d: 3 pairs in 15 nights -- 3 quads in 30 nights' % (metadata, year)
        ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
        plt.close()

        # Plot the cumulative completeness at 'year', for different discovery strategies.
        k = 'CumulativeCompleteness'
        strategies = ['3 pairs in 15 nights', '3 pairs in 30 nights',
                      '4 pairs in 20 nights', '3 triplets in 30 nights', '3 quads in 30 nights']
        plotbundles = []
        plotDicts = []
        basePlotDict = {'title': '%s Cumulative Completeness at year %d - %s' % (runName, year, metadata),
                        'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                        'legendloc': 'lower left'}
        caption = 'Cumulative completeness (fraction of population with H<=X) discovered at year %d,' % (year)
        caption += ' for a variety of discovery strategies.'
        displayDict = {'group': 'A: Discovery', 'subgroup': 'Cumulative completeness',
                       'order': order, 'caption': caption}
        for i, strategy in enumerate(strategies):
            md = '%s year %d, %s' % (metadata, year, strategy)
            plotbundles.append(allBundles[k][md])
            tmpPlotDict = {'color': colorlist[i]}
            tmpPlotDict.update(basePlotDict)
            plotDicts.append(tmpPlotDict)
        ph.setMetricBundles(plotbundles)
        ph.jointMetadata = '%s Year %d: 3 pairs in 15 nights -- 3 quads in 30 nights' % (metadata, year)
        ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
        plt.close()

        # Plot the differential completeness at 'year', for the odd discovery strategies.
        order += 1
        k = 'DifferentialCompleteness'
        print(allBundles[k].keys())
        strategies = ['3 pairs in 15 nights', 'Single detection', 'Single pair',
                      '3 pairs in 15 nights, SNR=3', '3 pairs in 15 nights, SNR=0']
        plotbundles = []
        plotDicts = []
        basePlotDict = {'title': '%s Differential Completeness at year %d - %s' % (runName, year, metadata),
                        'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                        'legendloc': 'lower left'}
        caption = 'Differential completeness (fraction of population with H=X) discovered at year %d,' % (year)
        caption += ' comparing the standard discovery strategy against an infinitely sensitive LSST (SNR=0),'
        caption += ' or where we only require SNR=3 for each detection,'
        caption += ' or one which had no cadence constraints (Single detection).'
        displayDict = {'group': 'A: Discovery', 'subgroup': 'Differential Completeness',
                       'order': order, 'caption': caption}
        for i, strategy in enumerate(strategies):
            md = '%s year %d, %s' % (metadata, year, strategy)
            plotbundles.append(allBundles[k][md])
            tmpPlotDict = {'color': colorlist[i]}
            tmpPlotDict.update(basePlotDict)
            plotDicts.append(tmpPlotDict)
        ph.setMetricBundles(plotbundles)
        ph.jointMetadata = '%s Year %d: non-realistic discovery options' % (metadata, year)
        ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
        plt.close()

        # Plot the cumulative completeness at 'year', for the odd discovery strategies.
        k = 'CumulativeCompleteness'
        strategies = ['3 pairs in 15 nights', 'Single detection', 'Single pair',
                      '3 pairs in 15 nights, SNR=3', '3 pairs in 15 nights, SNR=0']
        plotbundles = []
        plotDicts = []
        basePlotDict = {'title': '%s Differential Completeness at year %d - %s' % (runName, year, metadata),
                        'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                        'legendloc': 'lower left'}
        caption = 'Cumulative completeness (fraction of population with H<=X) discovered at year %d,' % (year)
        caption += ' comparing the standard discovery strategy against an infinitely sensitive LSST (SNR=0)'
        caption += ' or where we only require SNR=3 for each detection,'
        caption += ' or one which had no cadence constraints (Single detection).'
        displayDict = {'group': 'A: Discovery', 'subgroup': 'Cumulative completeness',
                       'order': order, 'caption': caption}
        for i, strategy in enumerate(strategies):
            md = '%s year %d, %s' % (metadata, year, strategy)
            plotbundles.append(allBundles[k][md])
            tmpPlotDict = {'color': colorlist[i]}
            tmpPlotDict.update(basePlotDict)
            plotDicts.append(tmpPlotDict)
        ph.setMetricBundles(plotbundles)
        ph.jointMetadata = '%s Year %d: non-realistic discovery options' % (metadata, year)
        ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
        plt.close()
        order += 1

    # Plot the differential completeness values @ each year for std discovery strategy, as a function of H.
    order += 1
    k = 'DifferentialCompleteness'
    strategy = '3 pairs in 15 nights'
    mdmatch = ['%s year %d, %s' % (metadata, nyr, strategy) for nyr in mParams['nyears']]
    plotbundles = []
    plotDicts = []
    basePlotDict = {'title': '%s %s Differential Completeness' % (runName, metadata),
                    'ylabel': 'Completeness @ H', 'yMin': 0, 'yMax': 1,
                    'legendloc': 'lower left'}
    caption = 'Differential completeness (fraction of population with H=X) discovered at different years.'
    caption += ' Assumes standard discovery strategy of 3 pairs in 15 nights.'
    displayDict = {'group': 'A: Discovery', 'subgroup': 'Differential Completeness',
                   'order': order, 'caption': caption}
    for i, md in enumerate(mdmatch):
        plotbundles.append(allBundles[k][md])
        tmpPlotDict = {'color': colorlist[i % len(colorlist)]}
        tmpPlotDict.update(basePlotDict)
        plotDicts.append(tmpPlotDict)
    ph.setMetricBundles(plotbundles)
    ph.jointMetadata = '%s %s Years %d to %d' % (metadata, strategy,
                                                 mParams['nyears'].min(), mParams['nyears'].max())
    ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
    plt.close()

    # Plot the cumulative completeness values @ each year for std discovery strategy, as a function of H.
    k = 'CumulativeCompleteness'
    strategy = '3 pairs in 15 nights'
    mdmatch = ['%s year %d, %s' % (metadata, nyr, strategy) for nyr in mParams['nyears']]
    plotbundles = []
    plotDicts = []
    basePlotDict = {'title': '%s %s Cumulative Completeness ' % (runName, metadata),
                    'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                    'legendloc': 'lower left'}
    caption = 'Cumulative completeness (fraction of population with H<=X) discovered at different years.'
    caption += ' Assumes standard discovery strategy of 3 pairs in 15 nights.'
    caption += ' Each observation must have a SNR of 5 (including trailing losses).'
    displayDict = {'group': 'A: Discovery', 'subgroup': 'Cumulative completeness',
                   'order': order, 'caption': caption}
    for i, md in enumerate(mdmatch):
        plotbundles.append(allBundles[k][md])
        tmpPlotDict = {'color': colorlist[i % len(colorlist)]}
        tmpPlotDict.update(basePlotDict)
        plotDicts.append(tmpPlotDict)
    ph.setMetricBundles(plotbundles)
    ph.jointMetadata = '%s %s Years %d to %d' % (metadata, strategy,
                                                 mParams['nyears'].min(), mParams['nyears'].max())
    ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)
    plt.close()

    # Plot the cumulative completeness at a particular value of H for std discovery, as a function of years.
    order += 1
    k = 'CumulativeCompleteness'
    strategy = '3 pairs in 15 nights'
    yrs = np.concatenate([[0], mParams['nyears']])
    completeness_at_year = np.zeros(len(yrs), float)
    b = allBundles[k]['%s year 10, %s' % (metadata, strategy)]
    # Pick a point to 'count' the completeness at.
    if Hmark is not None:
        hIdx = np.abs(b.slicer.Hrange - Hmark).argmin()
    else:
        hIdx = int(len(b.slicer.Hrange) / 3)
    completeness_at_year[0] = 0
    for i, nyr in enumerate(mParams['nyears']):
        md = '%s year %d, %s' % (metadata, nyr, strategy)
        b = allBundles[k][md]
        completeness_at_year[i+1] = b.metricValues[hIdx]
    fig = plt.figure()
    plt.plot(yrs, completeness_at_year, label='%s H<=%.2f' % (metadata, b.slicer.Hrange[hIdx]))
    plt.xlabel('Years into survey')
    plt.ylabel('Completeness @ H <= %.2f' % (b.slicer.Hrange[hIdx]))
    plt.title('Cumulative completeness as a function of time')
    plt.legend(loc=(0.7, 0.1), fancybox=True, fontsize='smaller')
    plotmetadata = 'years %s' % (' '.join(['%d' % nyr for nyr in mParams['nyears']]))
    caption = 'Cumulative completeness at H=%.2f, as a function of time. ' % (b.slicer.Hrange[hIdx])
    caption += 'Assumes the standard discovery strategy, 3 pairs of visits within 15 nights.'
    displayDict = {'group': 'A: Discovery', 'subgroup': 'Cumulative completeness',
                   'order': order, 'caption': caption}
    filename = '%s_%s_CompletenessOverTime_%.0f' % (b.runName, metadata, b.slicer.Hrange[hIdx])
    filename = utils.nameSanitize(filename)
    ph.saveFig(fig.number, filename, 'Combo', 'Cumulative completeness as a function of time',
               b.slicer.slicerName, b.runName, b.constraint, plotmetadata, displayDict=displayDict)

    # Make joint 'chance of detecting activity over time' plots, for the brightest objects.
    meanFraction = np.zeros(len(mParams['windows']), float)
    minFraction = np.zeros(len(mParams['windows']), float)
    maxFraction = np.zeros(len(mParams['windows']), float)
    Hidx = 0
    for i, win in enumerate(mParams['windows']):
        b = allBundles['ActivityTime'][win]
        meanFraction[i] = np.mean(b.metricValues.swapaxes(0, 1)[Hidx])
        minFraction[i] = np.min(b.metricValues.swapaxes(0, 1)[Hidx])
        maxFraction[i] = np.max(b.metricValues.swapaxes(0, 1)[Hidx])
    fig = plt.figure()
    plt.plot(mParams['windows'], meanFraction, 'r', label='Mean')
    plt.plot(mParams['windows'], minFraction, 'b:', label='Min')
    plt.plot(mParams['windows'], maxFraction, 'g--', label='Max')
    plt.xlabel('Length of activity (days)')
    plt.ylabel('Probability of detecting activity')
    plt.title('Chances of detecting activity (for H=%.1f %s)' % (b.slicer.Hrange[Hidx],
                                                                 metadata))
    plt.grid()
    plotmetadata = 'windows from %.1f to %.1f days' % (mParams['windows'][0], mParams['windows'][-1])
    caption = 'Min/Mean/Max chance of detecting activity, for objects with H=%.2f, ' % (b.slicer.Hrange[Hidx])
    caption += 'as a function of typical activity length (in days).'
    caption += 'Activity is presumed to be detected if an observation occured within one of the time bins.'
    displayDict = {'group': 'B: Characterization', 'subgroup': 'Activity', 'order':0, 'caption': caption}
    filename = '%s_%s_Activity_%s' % (b.runName, metadata, plotmetadata)
    filename = utils.nameSanitize(filename)
    ph.saveFig(fig.number, filename, 'Combo', 'Chances of detecting Activity lasting X days',
               b.slicer.slicerName, b.runName, b.constraint, plotmetadata, displayDict=displayDict)

    # Make a joint 'chance of detecting activity over period' plots, for the brightest objects.
    meanFraction = np.zeros(len(mParams['bins']), float)
    minFraction = np.zeros(len(mParams['bins']), float)
    maxFraction = np.zeros(len(mParams['bins']), float)
    Hidx = 0
    for i, bin in enumerate(mParams['bins']):
        b = allBundles['ActivityPeriod'][bin]
        meanFraction[i] = np.mean(b.metricValues.swapaxes(0, 1)[Hidx])
        minFraction[i] = np.min(b.metricValues.swapaxes(0, 1)[Hidx])
        maxFraction[i] = np.max(b.metricValues.swapaxes(0, 1)[Hidx])
    fig = plt.figure()
    plt.plot(mParams['bins'] / 360., meanFraction, 'r', label='Mean')
    plt.plot(mParams['bins'] / 360., minFraction, 'b:', label='Min')
    plt.plot(mParams['bins'] / 360., maxFraction, 'g--', label='Max')
    plt.xlabel('Length of activity (fraction of period)')
    plt.ylabel('Probabilty of detecting activity')
    plt.title('Chances of detecting activity (for H=%.1f %s)' % (b.slicer.Hrange[Hidx],
                                                                                  metadata))
    plt.grid()
    plotmetadata = 'bins from %.2f to %.2f' % (mParams['bins'][0], mParams['bins'][-1])
    caption = 'Min/Mean/Max chance of detecting recurring activity, '
    caption += 'for objects with H=%.2f, ' % (b.slicer.Hrange[Hidx])
    caption += 'as a function of typical activity length (in fraction of the period). '
    caption += 'Activity is presumed to be detected if an observation occured within one of the time bins.'
    displayDict = {'group': 'B: Characterization', 'subgroup': 'Activity', 'order': 1, 'caption': caption}
    filename = '%s_%s_Activity_%s' % (b.runName, metadata, plotmetadata)
    filename = utils.nameSanitize(filename)
    ph.saveFig(fig.number, filename, 'Combo', 'Chances of detecting Activity lasting X of period',
               b.slicer.slicerName, b.runName, b.constraint, plotmetadata, displayDict=displayDict)

    # Make a plot of the fraction of objects which could get colors.
    plotbundles = []
    plotDicts = []
    colors = {'%s u-g color' % metadata: 'cyan',
              '%s g-r color' % metadata: 'g',
              '%s r-i color' % metadata: 'burlywood',
              '%s i-z color' % metadata: 'magenta',
              '%s z-y color' % metadata: 'k'}
    b = allBundles['colorDetermination'].values()[0]
    caption = 'Mean likelihood of obtaining observations suitable for gathering a high-quality color '
    caption += 'measurement, as a function of H magnitude. '
    caption += 'Assumes that if %d pair(s) of observations are taken within %.2f hours, with SNR>%.2f, ' \
               % (b.metric.nPairs, b.metric.nHours, b.metric.snrLimit)
    caption += 'that a good color can be measured.'
    displayDict = {'group': 'B: Characterization', 'subgroup': 'Colors', 'order': 0, 'caption': caption}
    for md in ['u-g', 'g-r', 'r-i', 'i-z', 'z-y']:
        name = '%s %s color' % (metadata, md)
        plotbundles.append(allBundles['colorDetermination'][name])
        plotDicts.append({'label': name, 'npReduce': np.mean, 'color': colors[name], 'Hmark': None,
                          'ylabel': 'Fraction of objects', 'yMin': 0, 'yMax': 1,
                          'title': '%s: %s with color measurements' % (runName, metadata)})
    ph.setMetricBundles(plotbundles)
    ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDicts, displayDict=displayDict)

    return


def readMetricValues(bundle, tmpdir):
    filename = os.path.join(tmpdir, bundle.fileRoot + '.h5')
    metricValues, slicer = bundle.slicer.readData(filename)
    bundle.metricValues = metricValues
    bundle.metricValues.fill_value = 0
    bundle.slicer = slicer
    return bundle


def readAll(allBundles, orbitFile, outDir):
    # Read all (except completeness) bundles back from disk.
    missingBundles = []
    for k in allBundles:
        for md in allBundles[k]:
            b = allBundles[k][md]
            try:
                b = readMetricValues(b, outDir)
                b.slicer.readOrbits(orbitFile, Hrange=b.slicer.Hrange)
            except IOError as e:
                print('Problems with bundle %s %s, so skipping. \n %s'
                      % (k, md, e))
                missingBundles.append([k, md])
    for i in missingBundles:
        del allBundles[i[0]][i[1]]
    return allBundles


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--obsFile", type=str,
                        help="File containing the observations of the moving objects.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output directory for moving object metrics. Default '.'")
    parser.add_argument("--hMin", type=float, default=5.0, help="Minimum H value. Default 5.")
    parser.add_argument("--hMax", type=float, default=27.0, help="Maximum H value. Default 27.")
    parser.add_argument("--hStep", type=float, default=0.25, help="Stepsizes in H values.")
    parser.add_argument("--metadata", type=str, default='',
                        help="Base string to add to all metric metadata. Typically the object type.")
    parser.add_argument("--albedo", type=float, default=None,
                        help="Albedo value, to add diameters to upper scales on plots. Default None.")
    parser.add_argument("--hMark", type=float, default=None,
                        help="Add vertical lines at H=hMark on plots. Default None.")
    parser.add_argument("--nYearsMax", type=int, default=10,
                        help="Maximum number of years out to which to evaluate completeness."
                             "Default 10.")
    parser.add_argument("--plotOnly", action='store_true', default=False,
                        help="Reload metric values from disk and replot them.")
    args = parser.parse_args()

    if args.orbitFile is None:
        print('Must specify an orbitFile')
        exit()

    # Default parameters for metric setup.
    nyears = np.arange(2, args.nYearsMax+1, 2)
    nyears = np.concatenate([[1], nyears])
    if args.nYearsMax not in nyears:
        nyears = np.concatenate([nyears, [args.nYearsMax]])
    bins = np.arange(5, 95, 10.)  # binsize to split period (360deg)
    windows = np.arange(1, 200, 15)  # binsize to split time (days)
    mParams = {'nyears': nyears, 'bins': bins, 'windows': windows}

    if args.plotOnly:
        # Set up resultsDb.
        resultsDb = db.ResultsDb(outDir=args.outDir)
        tmpslicer = slicers.MoObjSlicer()
        allBundles = setupMetrics(tmpslicer, runName=args.opsimRun, metadata=args.metadata,
                                  albedo=args.albedo, Hmark=args.hMark, mParams=mParams)
        allBundles = readAll(allBundles, args.orbitFile, args.outDir)
        allBundles = addAllCompletenessBundles(allBundles, args.hMark, resultsDb=None)

    else:
        if args.obsFile is None:
            print('Must specify an obsFile when calculating the metrics.')
            exit()
        # Set up resultsDb.
        if not (os.path.isdir(args.outDir)):
            os.makedirs(args.outDir)
        resultsDb = db.ResultsDb(outDir=args.outDir)

        Hrange = np.arange(args.hMin, args.hMax + args.hStep, args.hStep)
        slicer = setupSlicer(args.orbitFile, Hrange, obsFile=args.obsFile)
        allBundles = setupMetrics(slicer, runName=args.opsimRun, metadata=args.metadata,
                                  albedo=args.albedo, Hmark=args.hMark, mParams=mParams)
        allBundles = runMetrics(allBundles, args.outDir, resultsDb, args.hMark)

    plotMetrics(allBundles, args.outDir, args.metadata, args.opsimRun, mParams,
                Hmark=args.hMark, resultsDb=resultsDb)
