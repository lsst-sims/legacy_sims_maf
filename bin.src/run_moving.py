#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np

import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches

# Assumes you have already created observation file,
# This is currently incomplete compared to movingObjects.py! No plotting, no automatic completeness bundles.

def setupMetrics(slicer, runName, metadata, mParams, albedo=None, Hmark=None):
    """
    Set up the standard metrics to analyze each opsim run.

    Parameters
    ----------
    slicer : ~lsst.sims.maf.slicer.MoObjSlicer
        Slicer to use to evaluate metrics.
    runName : str
        The name of the opsim run (e.g. "minion_1016") to add to plots, output filenames, etc.
    metadata : str
        Any standard metadata string to add to each output plot/filename, etc.
    mParams : dict
        Dictionary containing 'nyears', 'bins', and 'windows' to configure the discovery and activity metrics.
    albedo : float, optional
        Albedo to specify for the plotting dictionary. Default None (and so no 'size' marked on plots).
    Hmark : float, optional
        Hmark to specify for the plotting dictionary. Default None.

    Returns
    -------
    dict of metricBundles
    """
    # Set up the metrics.
    allBundles = {}

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce}

    summaryMetrics = [metrics.MoCompletenessAtTimeMetric(times=mParams['times'], Hval=Hmark,
                                                         cumulative=False, Hindex=0.33),
                      metrics.MoCompletenessAtTimeMetric(times=mParams['times'], Hval=Hmark,
                                                         cumulative=True, Hindex=0.33)]
    simpleSummaryMetrics = [metrics.MoCompletenessMetric(cumulative=False, Hindex=0.33),
                            metrics.MoCompletenessMetric(cumulative=True, Hindex=0.33)]

    plotFuncs = [plots.MetricVsH()]

    # Add different mag/vis stacker.
    stackerDet = stackers.MoMagStacker(lossCol='dmagDetect')
    stackerTrail = stackers.MoMagStacker(lossCol='dmagTrail')

    # Little subroutine to configure child discovery metrics in each year.
    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childname = 'Time'
        childmetric = metrics.Discovery_TimeMetric(parentMetric)
        childMetrics[childname] = childmetric
        for nyr in mParams['nyears']:
            childname = 'N_Chances_yr_%d' % nyr
            childmetric = metrics.Discovery_N_ChancesMetric(parentMetric, nightEnd=(nyr * 365),
                                                            metricName = 'Discovery_N_Chances_yr_%d' % nyr)
            childMetrics[childname] = childmetric
        return childMetrics
    def _configure_child_bundles(parentBundle):
        dispDict = {'group': groups['discovery'], 'subgroup': subgroups['completenessTable']}
        parentBundle.childBundles['Time'].metadata = parentBundle.metadata
        parentBundle.childBundles['Time'].setDisplayDict(dispDict)
        parentBundle.childBundles['Time'].setSummaryMetrics(summaryMetrics)
        for nyr in mParams['nyears']:
            parentBundle.childBundles['N_Chances_yr_%d' % nyr].metadata = parentBundle.metadata + \
                                                                          ' yr %d' % nyr
            parentBundle.childBundles['N_Chances_yr_%d' % nyr].setSummaryMetrics(simpleSummaryMetrics)
            parentBundle.childBundles['N_Chances_yr_%d' % nyr].setDisplayDict(dispDict)

    displayDict = {'group': groups['discovery']}
    # Set up discovery metrics; calculate at all years using child metrics.
    allBundles['discovery'] = {}
    # First standard SNR / probabilistic visibility (SNR~5)
    # 3 pairs in 15
    constraint = None
    md = metadata + ' 3 pairs in 15 nights'
    # Set up plot dict.
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    # Set basic metric.
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90./60./24.,
                                     nNightsPerWindow=3, tWindow=15)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    # Add summary statistics to each of the N_Chances child bundles.
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 12
    md = metadata + ' 3 pairs in 12 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=12)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 20
    md = metadata + ' 3 pairs in 20 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=20)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 25
    md = metadata + ' 3 pairs in 25 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=25)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 30
    md = metadata + ' 3 pairs in 30 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 4 pairs in 20
    md = metadata + ' 4 pairs in 20 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=4, tWindow=20)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 triplets in 30
    md = metadata + ' 3 triplets in 30 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=3, tMin=0, tMax=120. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 quads in 30
    md = metadata + ' 3 quads in 30 nights'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=4, tMin=0, tMax=150. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # Play with SNR.  SNR=4. Normal detection losses.
    # 3 pairs in 15
    md = metadata + ' 3 pairs in 15 nights, SNR=4'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=4)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 30, SNR=4
    md = metadata + ' 3 pairs in 30 nights, SNR=4'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, snrLimit=4)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # Play with SNR.  SNR=3
    # 3 pairs in 15, SNR=3
    md = metadata + ' 3 pairs in 15 nights, SNR=3'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=3)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 30, SNR=3
    md = metadata + ' 3 pairs in 30 nights, SNR=3'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, snrLimit=3)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # SNR = 0
    # 3 pairs in 15, SNR=0
    md = metadata + ' 3 pairs in 15 nights, SNR=0'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=0)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # Look at swapping to trailing losses instead of detection losses.
    # 3 pairs in 15, trailing loss
    md = metadata + ' 3 pairs in 15 nights trailing loss'
    # Set up plot dict.
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    # Set basic metric.
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerTrail],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 30, trailing loss
    md = metadata + ' 3 pairs in 30 nights trailing loss'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerTrail],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 15, SNR=4, trailing loss
    md = metadata + ' 3 pairs in 15 nights trailing loss, SNR=4'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=4)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerTrail],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # 3 pairs in 30, trailing loss, SNR=4
    md = metadata + ' 3 pairs in 30 nights trailing loss, SNR=4'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, snrLimit=4)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerTrail],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # Play with weird strategies.
    # Single detection.
    md = metadata + ' Single detection'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=1, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=1, tWindow=5)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # Single pair of detections.
    md = metadata + ' Single pair'
    plotDict = {'nxbins': 200, 'nybins': 200,
                'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=1, tWindow=5)
    childMetrics = _setup_child_metrics(metric)
    bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[stackerDet],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    allBundles['discovery'][md] = bundle

    # High velocity detections and 'magic' detections.
    displayDict = {'group': groups['discovery'], 'subgroup': subgroups['completenessTable']}
    allBundles['velocity'] = {}
    allBundles['magic'] = {}
    for nyr in mParams['nyears']:
        constraint = 'night <= %d' % (nyr * 365)
        # High velocity.
        md = metadata + ' High velocity pair yr %d' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.HighVelocityNightsMetric(psfFactor=2., nObsPerNight=2)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    stackerList=[stackerDet],
                                    runName=runName, metadata=md,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    summaryMetrics=simpleSummaryMetrics,
                                    displayDict=displayDict)
        allBundles['velocity'][md] = bundle
        # Magic. 6 detections in 60 days (at SNR=5).
        md = metadata + ' 6 detections in 60 nights yr %d' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, md)}
        plotDict.update(basicPlotDict)
        metric = metrics.MagicDiscoveryMetric(nObs=6, tWindow=60)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    stackerList=[stackerDet],
                                    runName=runName, metadata=md,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    summaryMetrics=simpleSummaryMetrics,
                                    displayDict=displayDict)
        allBundles['magic'][md] = bundle

    displayDict = {'group':groups['characterization']}

    # Number of observations.
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
                                summaryMetrics=None,
                                displayDict=displayDict)
    allBundles['nObs'][md] = bundle

    # Observational arc.
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
                                summaryMetrics=None,
                                displayDict=displayDict)
    allBundles['obsArc'][md] = bundle

    # Activity detection.
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
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
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
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        allBundles['ActivityPeriod'][b] = bundle

    # Lightcurve inversion.
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
                                summaryMetrics=None,
                                displayDict=displayDict)
    allBundles['lightcurveInversion'][md] = bundle

    # Color determination.
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
                                summaryMetrics=None,
                                displayDict=displayDict)
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
                                summaryMetrics=None,
                                displayDict=displayDict)
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
                                summaryMetrics=None,
                                displayDict=displayDict)
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
                                summaryMetrics=None,
                                displayDict=displayDict)
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
                                summaryMetrics=None,
                                displayDict=displayDict)
    allBundles['colorDetermination'][md] = bundle

    return allBundles


def runMetrics(bdict, outDir, resultsDb=None, Hmark=None):
    """
    Run metrics, write basic output in OutDir.

    Parameters
    ----------
    bdict : dict
        The metric bundles to run.
    outDir : str
        The output directory to store results.
    resultsDb : ~lsst.sims.maf.db.ResultsDb, optional
        The results database to use to track metrics and summary statistics.
    Hmark : float, optional
        The Hmark value to add to the completeness bundles plotDicts.

    Returns
    -------
    dict of metricBundles
        The bundles in this dict now contain the metric values as well.
    """
    print("Calculating metric values.")
    bg = mmb.MoMetricBundleGroup(bdict, outDir=outDir, resultsDb=resultsDb)
    # Just calculate here, we'll create the (mostly custom) plots later.
    bg.runAll()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--obsFile", type=str,
                        help="File containing the observations of the moving objects.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output directory for moving object metrics. Default '.'")
    parser.add_argument("--opsimDb", type=str, default=None,
                        help="Path and filename of opsim db, to write config* files to output directory."
                        " Optional: if not provided, config* files won't be created but analysis will run.")
    parser.add_argument("--hMin", type=float, default=5.0, help="Minimum H value. Default 5.")
    parser.add_argument("--hMax", type=float, default=27.0, help="Maximum H value. Default 27.")
    parser.add_argument("--hStep", type=float, default=0.5, help="Stepsizes in H values.")
    parser.add_argument("--metadata", type=str, default='',
                        help="Base string to add to all metric metadata. Typically the object type.")
    parser.add_argument("--albedo", type=float, default=None,
                        help="Albedo value, to add diameters to upper scales on plots. Default None.")
    parser.add_argument("--hMark", type=float, default=None,
                        help="Add vertical lines at H=hMark on plots. Default None.")
    parser.add_argument("--nYearsMax", type=int, default=10,
                        help="Maximum number of years out to which to evaluate completeness."
                             "Default 10.")
    parser.add_argument("--startTime", type=float, default=59580,
                        help="Time at start of survey (to set time for summary metrics).")
    parser.add_argument("--plotOnly", action='store_true', default=False,
                        help="Reload metric values from disk and replot them.")
    args = parser.parse_args()

    if args.orbitFile is None:
        print('Must specify an orbitFile')
        exit()

    # Default parameters for metric setup.
    stepsize = 365/2.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime
    bins = np.arange(5, 95, 10.)  # binsize to split period (360deg)
    windows = np.arange(1, 200, 15)  # binsize to split time (days)

    if args.plotOnly:
        # Set up resultsDb.
        pass

    else:
        if args.obsFile is None:
            print('Must specify an obsFile when calculating the metrics.')
            exit()
        # Set up resultsDb.
        if not (os.path.isdir(args.outDir)):
            os.makedirs(args.outDir)
        resultsDb = db.ResultsDb(outDir=args.outDir)

        Hrange = np.arange(args.hMin, args.hMax + args.hStep, args.hStep)
        slicer = batches.setupSlicer(args.orbitFile, Hrange, obsFile=args.obsFile)
        opsdb = db.OpsimDatabase(args.opsimDb)
        colmap = batches.getColMap(opsdb)
        bdict = batches.discoveryBatch(slicer, colmap=colmap, runName=args.opsimRun, metadata=args.metadata,
                                       albedo=args.albedo, Hmark=args.hMark, times=times)
        runMetrics(bdict, args.outDir, resultsDb, args.hMark)

    #plotMetrics(allBundles, args.outDir, args.metadata, args.opsimRun, mParams,
    #            Hmark=args.hMark, resultsDb=resultsDb)

    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        utils.writeConfigs(opsdb, args.outDir)
