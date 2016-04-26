#!/usr/bin/env python

from __future__ import print_function, division

import os
import argparse
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

# Needs MafSSO branch from sims_maf.
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb

# Assumes you have already created observation file,
# using make_movingObject_obs.py.


def readObservations(orbitFile, obsFile, Hrange):
    # Read the orbit file and set the H values for the slicer.
    slicer = slicers.MoSlicer(orbitFile, Hrange=Hrange)
    slicer.readObs(obsFile)
    return slicer


def setupMetrics(slicer, runName=None, metadata='', albedo=None, Hmark=None):
    # Set up the metrics.
    allBundles = {}

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark}

    # Basic discovery/completeness metric, calculate at several years.
    nyears = [2, 4, 6, 8, 10, 12, 15]
    nyears = [10]
    summaryMetrics = [metrics.MO_CompletenessMetric(), metrics.MO_CumulativeCompletenessMetric()]
    allBundles['discoveryChances'] = {}
    for nyr in nyears:
        bundleName = 'year %d' % nyr
        constraint = 'night < %d' %(nyr * 365 + 1)
        metadata += ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: Discovery Chances %s' % (runName, metadata)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryChancesMetric(nObsPerNight=2, tNight=90./60./24.,
                                                nNightsPerWindow=3, tWindow=15)
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=metadata,
                                    summaryMetrics=summaryMetrics,
                                    plotDict=plotDict)
        allBundles['discoveryChances'][bundleName] = bundle

    # More complicated discovery metric, with child metrics.
    allBundles['discovery'] = {}
    for nyr in nyears:
        bundleName = 'year %d' % nyr
        constraint = 'night < %d' %(nyr * 365 + 1)
        metadata += ' year %d, 3 pairs in 15 nights' % nyr
        plotDict = {'nxbins': 200, 'nybins': 200,
                    'title': '%s: %s' % (runName, metadata)}
        plotDict.update(basicPlotDict)
        metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90./60./24.,
                                         nNightsPerWindow=3, tWindow=15)
        childMetrics = {'DiscTime': metrics.Discovery_TimeMetric(metric, i=0),
                        'NChances': metrics.Discovery_N_ChancesMetric(metric)}
        bundle = mmb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=metadata,
                                    childMetrics=childMetrics,
                                    summaryMetrics=summaryMetrics,
                                    plotDict=plotDict)
        allBundles['discovery'][bundleName] = bundle

    return allBundles



def makeCompletenessBundle(bundle, summaryName='CumulativeCompleteness', plotDict={}):
    # Make a 'mock' metric bundle from a bundle which had the MO_Completeness or MO_CumulativeCompleteness
    # summary metrics run. This lets us use a normal plotHandler to generate combined plots.
    completeness = ma.MaskedArray(data=bundle.summaryValues[summaryName][0],
                                  mask=np.zeros(len(bundle.summaryValues[summaryName][0])),
                                  fill_value=0)
    mb = mmb.MoMetricBundle(metrics.MO_CompletenessMetric(metricName=summaryName),
                            slicer, constraint=None,
                            runName=runName, plotDict=plotDict)
    mb.metricValues = completeness
    return mb


def runMetrics(allBundles, outDir):
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
    print("Counted %d metric bundles total." % count)

    # Set up resultsDb.
    resultsDb = db.ResultsDb(outDir=outDir)

    bg = mmb.MoMetricBundleGroup(bundleDict, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    bg.summaryAll()
    bg.plotAll()

    # Combine differential completeness summary values, over multiple years for discoveryChances.
    ph = plots.PlotHandler(outDir=outDir, savefig=True, resultsDb=resultsDb,
                           figformat='pdf', dpi=600, thumbnail=True, closefigs=True)
    plotbundles = []
    for k in ('discoveryChances', 'discovery'):
        for nyr in allBundles[k]:
            if k == 'discoveryChances':
                b = allBundles[k][nyr]
            elif k == 'discovery':
                b = allBundles[k][nyr].childBundles['N_Chances']
            plotbundles.append(makeCompletenessBundle(b),
                                summaryName='Completeness')
    ph.setMetricBundles(compbundles)
    b = allBundles['discovery'].keys()[0]
    plotDict = {'title': '%s Differential Completeness - 3 pairs in 15 nights' %(b.runName),
                'ylabel': 'Completeness @ H', 'yMin': 0, 'yMax': 1,
                'albedo':albedo, 'legendloc': 'upper right'}
    ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDict)

    plotbundles = []
    for k in ('discoveryChances', 'discovery'):
        for nyr in allBundles[k]:
            if k == 'discoveryChances':
                b = allBundles[k][nyr]
            elif k == 'discovery':
                b = allBundles[k][nyr].childBundles['N_Chances']
            plotbundles.append(makeCompletenessBundle(b),
                               summaryName='CumulativeCompleteness')
    ph.setMetricBundles(compbundles)
    b = allBundles['discovery'].keys()[0]
    plotDict = {'title': '%s Cumulative Completeness - 3 pairs in 15 nights' %(b.runName),
                'ylabel': 'Completeness <= H', 'yMin': 0, 'yMax': 1,
                'albedo':albedo, 'legendloc': 'upper right'}
    ph.plot(plotFunc=plots.MetricVsH(), plotDicts=plotDict)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--opsimRun", type=str, help="Name of opsim run")
    parser.add_argument("--outDir", type=str, default='.', help="Output directory for moving object metrics.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--obsFile", type=str, help="File containing the observations of the moving objects.")
    parser.add_argument("--hMin", type=float, default=5.0, help="Minimum H value. Default 5.")
    parser.add_argument("--hMax", type=float, default=27.0, help="Maximum H value. Default 27.")
    parser.add_argument("--hStep", type=float, default=0.25, help="Stepsizes in H values.")
    parser.add_argument("--metadata", type=str, default='', help="Base string to add to all metric metadata.")
    parser.add_argument("--albedo", type=float, default=None,
                        help="Albedo value, to add diameters to upper scales on plots. Default None.")
    parser.add_argument("--hMark", type=float, default=None,
                        help="Add vertical lines at H=hMark on plots. Default None.")
    args = parser.parse_args()

    Hrange = np.arange(args.hMin, args.hMax+args.hStep, args.hStep)
    slicer = readObservations(args.orbitFile, args.obsFile, Hrange)

    bundleDict = setupMetrics(slicer, runName=args.opsimRun, metadata=args.metadata,
                              albedo=args.albedo, Hmark=args.hMark)

    runMetrics(bundleDict, args.opsimRun)
