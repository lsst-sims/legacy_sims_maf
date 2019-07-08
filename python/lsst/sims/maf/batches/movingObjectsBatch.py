from __future__ import print_function, division
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import summaryCompletenessAtTime, summaryCompletenessOverH, fractionPopulationAtThreshold

__all__ = ['setupMoSlicer', 'quickDiscoveryBatch', 'discoveryBatch',
           'runCompletenessSummary', 'runFractionSummary',
           'characterizationAsteroidBatch', 'characterizationOuterBatch',
           'readAndCombine', 'combineSubsets']


def setupMoSlicer(orbitFile, Hrange, obsFile=None):
    """
    Set up the slicer and read orbitFile and obsFile from disk.

    Parameters
    ----------
    orbitFile : str
        The file containing the orbit information.
    Hrange : numpy.ndarray or None
        The Hrange parameter to pass to slicer.readOrbits
    obsFile : str, optional
        The file containing the observations of each object, optional.
        If not provided (default, None), then the slicer will not be able to 'slice', but can still plot.

    Returns
    -------
    ~lsst.sims.maf.slicer.MoObjSlicer
    """
    # Read the orbit file and set the H values for the slicer.
    slicer = slicers.MoObjSlicer(Hrange=Hrange)
    slicer.setupSlicer(orbitFile=orbitFile, obsFile=obsFile)
    return slicer


def quickDiscoveryBatch(slicer, colmap=None, runName='opsim', detectionLosses='detection', metadata='',
                        albedo=None, Hmark=None, npReduce=np.mean, constraint=None):
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []
    plotBundles = []

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce,
                     'nxbins': 200, 'nybins': 200}
    plotFuncs = [plots.MetricVsH()]
    displayDict ={'group': 'Discovery'}

    if detectionLosses not in ('detection', 'trailing'):
        raise ValueError('Please choose detection or trailing as options for detectionLosses.')
    if detectionLosses == 'trailing':
        magStacker = stackers.MoMagStacker(lossCol='dmagTrail')
        detectionLosses = ' trailing loss'
    else:
        magStacker = stackers.MoMagStacker(lossCol='dmagDetect')
        detectionLosses = ' detection loss'

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {'mjdCol': colmap['mjd'], 'seeingCol': colmap['seeingGeom'],
                 'expTimeCol': colmap['exptime'], 'm5Col': colmap['fiveSigmaDepth'],
                 'nightCol': colmap['night'], 'filterCol': colmap['filter']}

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics['Time'] = metrics.Discovery_TimeMetric(parentMetric, **colkwargs)
        childMetrics['N_Chances'] = metrics.Discovery_N_ChancesMetric(parentMetric, **colkwargs)
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {'group': 'Discovery', 'subgroup': 'Time',
                    'caption': 'Time of discovery of objects', 'order': 0}
        parentBundle.childBundles['Time'].setDisplayDict(dispDict)
        dispDict = {'group': 'Discovery', 'subgroup': 'NChances',
                    'caption': 'Number of chances for discovery of objects', 'order': 0}
        parentBundle.childBundles['N_Chances'].setDisplayDict(dispDict)
        return

    # 3 pairs in 15
    md = metadata + ' 3 pairs in 15 nights SNR=5' + detectionLosses
    # Set up plot dict.
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90./60./24.,
                                     nNightsPerWindow=3, tWindow=15, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 30
    md = metadata + ' 3 pairs in 30 nights SNR=5' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, snrLimit=5, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles


def discoveryBatch(slicer, colmap=None, runName='opsim', detectionLosses='detection', metadata='',
                   albedo=None, Hmark=None, npReduce=np.mean, constraint=None):
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []
    plotBundles = []

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce,
                     'nxbins': 200, 'nybins': 200}
    plotFuncs = [plots.MetricVsH()]
    displayDict ={'group': 'Discovery'}

    if detectionLosses not in ('detection', 'trailing'):
        raise ValueError('Please choose detection or trailing as options for detectionLosses.')
    if detectionLosses == 'trailing':
        # These are the SNR-losses only.
        magStacker = stackers.MoMagStacker(lossCol='dmagTrail')
        detectionLosses = ' trailing loss'
    else:
        # This is SNR losses, plus additional loss due to detecting with stellar PSF.
        magStacker = stackers.MoMagStacker(lossCol='dmagDetect')
        detectionLosses = ' detection loss'

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {'mjdCol': colmap['mjd'], 'seeingCol': colmap['seeingGeom'],
                 'expTimeCol': colmap['exptime'], 'm5Col': colmap['fiveSigmaDepth'],
                 'nightCol': colmap['night'], 'filterCol': colmap['filter']}

    def _setup_child_metrics(parentMetric):
        childMetrics = {}
        childMetrics['Time'] = metrics.Discovery_TimeMetric(parentMetric, **colkwargs)
        childMetrics['N_Chances'] = metrics.Discovery_N_ChancesMetric(parentMetric, **colkwargs)
        # Could expand to add N_chances per year, but not really necessary.
        return childMetrics

    def _configure_child_bundles(parentBundle):
        dispDict = {'group': 'Discovery', 'subgroup': 'Time',
                    'caption': 'Time of discovery of objects', 'order': 0}
        parentBundle.childBundles['Time'].setDisplayDict(dispDict)
        dispDict = {'group': 'Discovery', 'subgroup': 'NChances',
                    'caption': 'Number of chances for discovery of objects', 'order': 0}
        parentBundle.childBundles['N_Chances'].setDisplayDict(dispDict)
        return

    # First standard SNR / probabilistic visibility (SNR~5)
    # 3 pairs in 15
    md = metadata + ' 3 pairs in 15 nights' + detectionLosses
    # Set up plot dict.
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90./60./24.,
                                     nNightsPerWindow=3, tWindow=15, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 12
    md = metadata + ' 3 pairs in 12 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=12, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 20
    md = metadata + ' 3 pairs in 20 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=20, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 25
    md = metadata + ' 3 pairs in 25 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=25, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 30
    md = metadata + ' 3 pairs in 30 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 4 pairs in 20
    md = metadata + ' 4 pairs in 20 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=4, tWindow=20, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 triplets in 30
    md = metadata + ' 3 triplets in 30 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=3, tMin=0, tMax=120. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 quads in 30
    md = metadata + ' 3 quads in 30 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=4, tMin=0, tMax=150. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with SNR.  SNR=4. Normal detection losses.
    # 3 pairs in 15
    md = metadata + ' 3 pairs in 15 nights SNR=4' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=4, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # 3 pairs in 30, SNR=4
    md = metadata + ' 3 pairs in 30 nights SNR=4' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=30, snrLimit=4, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with SNR.  SNR=3
    # 3 pairs in 15, SNR=3
    md = metadata + ' 3 pairs in 15 nights SNR=3' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=3, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # SNR = 0
    # 3 pairs in 15, SNR=0
    md = metadata + ' 3 pairs in 15 nights SNR=0' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=3, tWindow=15, snrLimit=0, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Play with weird strategies.
    # Single detection.
    md = metadata + ' Single detection' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=1, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=1, tWindow=5, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # Single pair of detections.
    md = metadata + ' Single pair' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.DiscoveryMetric(nObsPerNight=2, tMin=0, tMax=90. / 60. / 24.,
                                     nNightsPerWindow=1, tWindow=5, **colkwargs)
    childMetrics = _setup_child_metrics(metric)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                childMetrics=childMetrics,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    _configure_child_bundles(bundle)
    bundleList.append(bundle)

    # High velocity discovery.
    displayDict['subgroup'] = 'High Velocity'

    # High velocity.
    md = metadata + ' High velocity pair' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.HighVelocityNightsMetric(psfFactor=2., nObsPerNight=2, **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # "magic" detection - 6 in 60 days (at SNR=5).
    md = metadata + ' 6 detections in 60 nights' + detectionLosses
    plotDict = {'title': '%s: %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.MagicDiscoveryMetric(nObs=6, tWindow=60, **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                stackerList=[magStacker],
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles


def runCompletenessSummary(bdict, Hmark, times, outDir, resultsDb):
    """
    Calculate completeness and create completeness bundles from all N_Chances and Time child metrics
    of the (discovery) bundles in bdict, and write completeness at Hmark to resultsDb, save bundle to disk.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : dict of metricBundles
        Dict containing ~lsst.sims.maf.MoMetricBundles,
        including bundles we're expecting to contain completeness.
    Hmark : float
        Hmark value to add to completeness plotting dict.
    times : np.ndarray
        The times at which to calculate completeness (over time).
    outDir : str
        Output directory to save completeness bundles to disk.
    resultsDb : ~lsst.sims.maf.db.ResultsDb
        Results database to save information about completeness bundle.

    Returns
    -------
    dict of metricBundles
        A dictionary of the new completeness bundles. Keys match original keys,
        with additions of "[Differential,Cumulative]Completeness@Time"
        and "[Differential,Cumulative]Completeness" to distinguish new entries.
    """
    # Add completeness bundles and write completeness at Hmark to resultsDb.
    completeness = {}
    group = 'Discovery'
    subgroup = 'Completeness @ H=%.1f' % (Hmark)

    # Set up the summary metrics.
    summaryTimeMetrics = summaryCompletenessAtTime(times, Hval=Hmark, Hindex=0.33)
    summaryHMetrics = summaryCompletenessOverH(requiredChances=1, Hindex=0.33)

    def _compbundles(b, bundle, Hmark, resultsDb):
        comp = {}
        # Bundle = single metric bundle. Add differential and cumulative completeness.
        if 'Time' in bundle.metric.name:
            for metric in summaryTimeMetrics:
                newkey = b + ' ' + metric.name
                comp[newkey] = mb.makeCompletenessBundle(bundle, metric,
                                                         Hmark=None, resultsDb=resultsDb)
        else:
            for metric in summaryHMetrics:
                newkey = b + ' ' + metric.name
                comp[newkey] = mb.makeCompletenessBundle(bundle, metric,
                                                         Hmark=Hmark, resultsDb=resultsDb)
        return comp

    # Generate the completeness bundles for the various discovery metrics.
    for b, bundle in bdict.items():
        if isinstance(bundle.metric, metrics.DiscoveryMetric):
            childkeys = ['Time', 'N_Chances']
            for k in bundle.childBundles:
                if k in childkeys:
                    childbundle = bundle.childBundles[k]
                    completeness.update(_compbundles(b, childbundle, Hmark, resultsDb))
        if isinstance(bundle.metric, metrics.HighVelocityNightsMetric):
            completeness.update(_compbundles(b, bundle, Hmark, resultsDb))
        if isinstance(bundle.metric, metrics.MagicDiscoveryMetric):
            completeness.update(_compbundles(b, bundle, Hmark, resultsDb))

    # Write the completeness bundles to disk, so we can re-read them later.
    # (also set the display dict properties, for the resultsDb output).
    for b, bundle in completeness.items():
        bundle.setDisplayDict({'group': group, 'subgroup': subgroup})
        bundle.write(outDir=outDir, resultsDb=resultsDb)

    return completeness


def characterizationAsteroidBatch(slicer, colmap=None, runName='opsim', metadata='',
                                  albedo=None, Hmark=None, constraint=None, npReduce=np.mean,
                                  windows=None, bins=None):

    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []
    plotBundles = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {'mjdCol': colmap['mjd'], 'seeingCol': colmap['seeingGeom'],
                 'expTimeCol': colmap['exptime'], 'm5Col': colmap['fiveSigmaDepth'],
                 'nightCol': colmap['night'], 'filterCol': colmap['filter']}

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce,
                     'nxbins': 200, 'nybins': 200}
    plotFuncs = [plots.MetricVsH()]
    displayDict ={'group': 'Characterization'}

    # Stackers
    magStacker = stackers.MoMagStacker(lossCol='dmagDetect')
    eclStacker = stackers.EclStacker()
    stackerList = [magStacker, eclStacker]

    # Windows are the different 'length of activity'
    if windows is None:
        windows = np.arange(10, 200, 30.)
    # Bins are the different 'anomaly variations' of activity
    if bins is None:
        bins = np.arange(5, 185, 20.)

    # Number of observations.
    md = metadata
    plotDict = {'ylabel': 'Number of observations (#)',
                'title': '%s: Number of observations %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Observational arc.
    md = metadata
    plotDict = {'ylabel': 'Observational Arc (days)',
                'title': '%s: Observational Arc Length %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Activity detection.
    for w in windows:
        md = metadata + ' activity lasting %.0f days' % w
        plotDict = {'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.0f day window' % w}
        metricName = 'Chances of detecting activity lasting %.0f days' % w
        metric = metrics.ActivityOverTimeMetric(w, metricName=metricName, **colkwargs)
        bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                   stackerList=stackerList,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    for b in bins:
        md = metadata + ' activity covering %.0f deg' % (b)
        plotDict = {'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.0f deg window' % b}
        metricName = 'Chances of detecting activity covering %.0f deg' % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName, **colkwargs)
        bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                   stackerList=stackerList,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    # Lightcurve inversion.
    md = metadata
    plotDict = {'yMin': 0, 'yMax': 1, 'ylabel': 'Fraction of objects',
                'title': '%s: Fraction with potential lightcurve inversion %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveInversion_AsteroidMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Color determination.
    md = metadata
    plotDict = {'yMin': 0, 'yMax': 1, 'ylabel': 'Fraction of objects',
                'title': '%s: Fraction of population with colors in X filters %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.Color_AsteroidMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles


def characterizationOuterBatch(slicer, colmap=None, runName='opsim', metadata='',
                               albedo=None, Hmark=None, constraint=None, npReduce=np.mean,
                               windows=None, bins=None):

    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []
    plotBundles = []

    # Set up a dictionary to pass to each metric for the column names.
    colkwargs = {'mjdCol': colmap['mjd'], 'seeingCol': colmap['seeingGeom'],
                 'expTimeCol': colmap['exptime'], 'm5Col': colmap['fiveSigmaDepth'],
                 'nightCol': colmap['night'], 'filterCol': colmap['filter']}

    basicPlotDict = {'albedo': albedo, 'Hmark': Hmark, 'npReduce': npReduce,
                     'nxbins': 200, 'nybins': 200}
    plotFuncs = [plots.MetricVsH()]
    displayDict ={'group': 'Characterization'}

    # Stackers
    magStacker = stackers.MoMagStacker(lossCol='dmagDetect')
    eclStacker = stackers.EclStacker()
    stackerList = [magStacker, eclStacker]

    # Windows are the different 'length of activity'
    if windows is None:
        windows = np.arange(10, 200, 30.)
    # Bins are the different 'anomaly variations' of activity
    if bins is None:
        bins = np.arange(5, 185, 20.)

    # Number of observations.
    md = metadata
    plotDict = {'ylabel': 'Number of observations (#)',
                'title': '%s: Number of observations %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                               runName=runName, metadata=md,
                               plotDict=plotDict, plotFuncs=plotFuncs,
                               displayDict=displayDict)
    bundleList.append(bundle)

    # Observational arc.
    md = metadata
    plotDict = {'ylabel': 'Observational Arc (days)',
                'title': '%s: Observational Arc Length %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                               runName=runName, metadata=md,
                               plotDict=plotDict, plotFuncs=plotFuncs,
                               displayDict=displayDict)
    bundleList.append(bundle)

    # Activity detection.
    for w in windows:
        md = metadata + ' activity lasting %.0f days' % w
        plotDict = {'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.0f day window' % w}
        metricName = 'Chances of detecting activity lasting %.0f days' % w
        metric = metrics.ActivityOverTimeMetric(w, metricName=metricName, **colkwargs)
        bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                   stackerList=stackerList,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    for b in bins:
        md = metadata + ' activity covering %.0f deg' % (b)
        plotDict = {'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.2f deg window' % b}
        metricName = 'Chances of detecting activity covering %.0f deg' % (b)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName, **colkwargs)
        bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                   stackerList=stackerList,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    # Color determination.
    md = metadata
    plotDict = {'yMin': 0, 'yMax': 1, 'ylabel': 'Fraction of objects',
                'title': '%s: Fraction of population with colors in X filters %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveColor_OuterMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                               stackerList=stackerList,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles


def runFractionSummary(bdict, Hmark, outDir, resultsDb):
    """
    Calculate fractional completeness of the population for color and lightcurve metrics.

    This should be done after combining any sub-sets of the metric results.

    Parameters
    ----------
    bdict : dict of metricBundles
        Dict containing ~lsst.sims.maf.MoMetricBundles,
        including bundles we're expecting to contain lightcurve/color evaluations.
    Hmark : float
        Hmark value to add to completeness plotting dict.
    times : np.ndarray
        The times at which to calculate completeness (over time).
    outDir : str
        Output directory to save completeness bundles to disk.
    resultsDb : ~lsst.sims.maf.db.ResultsDb
        Results database to save information about completeness bundle.

    Returns
    -------
    dict of metricBundles
        Now the resulting metricBundles also includes new nested dicts with keys "FracPop_*".
    """
    fractions = {}
    group = 'Characterization'
    subgroup = 'Fraction of Population with Color/Lightcurve'

    # Look for metrics from asteroid or outer solar system color/lightcurve metrics.
    inversionSummary = fractionPopulationAtThreshold([1], ['inversion'])
    asteroidColorSummary = fractionPopulationAtThreshold([4, 3, 2, 1], ['6 of ugrizy', '5 of grizy',
                                                                        '4 of grizy',
                                                                        '2 of g, r or i, z or y'])
    asteroidSummaryMetrics = {'LightcurveInversion_Asteroid': inversionSummary,
                              'Color_Asteroid': asteroidColorSummary}

    outerColorSummary = fractionPopulationAtThreshold([6, 5, 4, 3, 2, 1], ['6 filters', '5 filters',
                                                                           '4 filters', '3 filters',
                                                                           '2 filters', '1 filters'])
    outerSummaryMetrics = {'LightcurveColor_Outer': outerColorSummary}

    for b, bundle in bdict.items():
        for k in asteroidSummaryMetrics:
            if k in b:
                for metric in asteroidSummaryMetrics[k]:
                    newkey = b + ' ' + metric.name
                    fractions[newkey] = mb.makeCompletenessBundle(bundle, metric,
                                                                  Hmark=Hmark, resultsDb=resultsDb)
        for k in outerSummaryMetrics:
            if k in b:
                for metric in outerSummaryMetrics[k]:
                    newkey = b + ' ' + metric.name
                    fractions[newkey] = mb.makeCompletenessBundle(bundle, metric,
                                                                  Hmark=Hmark, resultsDb=resultsDb)
    # Write the fractional populations bundles to disk, so we can re-read them later.
    # (also set the display dict properties, for the resultsDb output).
    for b, bundle in fractions.items():
        bundle.setDisplayDict({'group': group, 'subgroup': subgroup})
        bundle.write(outDir=outDir, resultsDb=resultsDb)

    return fractions

def readAndCombine(orbitRoot, baseDir, splits, metricfile):
    """Read and combine the metric results from split locations, returning a single bundle.

    This will read the files from
    baseDir/orbitRoot_[split]/metricfile
    where split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], etc. (the subsets the original orbit file was split into).

    Parameters
    ----------
    orbitRoot: str
        The root of the orbit file - l7_5k, mbas_5k, etc.
    baseDir: str
        The root directory containing the subset directories. (e.g. '.' often)
    splits: np.ndarray or list of ints
        The integers describing the split directories (e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    metricfile: str
        The metric filename.

    Returns
    -------
    ~lsst.sims.maf.bundle
        A single metric bundle containing the combined data from each of the subsets.

    Note that this won't work for particularly complex metric values, such as the parent Discovery metrics.
    However, you can read and combine their child metrics, as for these we can propagate the data masks.
    """
    subsets = {}
    for i in splits:
        subsets[i] = mb.createEmptyMoMetricBundle()
        ddir = os.path.join(baseDir, orbitRoot + '_%d' % i)
        subsets[i].read(os.path.join(ddir, metricfile))
    bundle = combineSubsets(subsets)
    return bundle


def combineSubsets(mbSubsets):
    # Combine the data from the subset metric bundles.
    # The first bundle will be used a template for the slicer.
    if isinstance(mbSubsets, dict):
        first = mbSubsets[list(mbSubsets.keys())[0]]
    else:
        first = mbSubsets[0]
        subsetdict = {}
        for i, b in enumerate(mbSubsets):
            subsetdict[i] = b
        mbSubsets = subsetdict
    joint = mb.createEmptyMoMetricBundle()
    # Check if they're the same slicer.
    slicer = deepcopy(first.slicer)
    for i in mbSubsets:
        if np.any(slicer.slicePoints['H'] != mbSubsets[i].slicer.slicePoints['H']):
            if np.any(slicer.slicePoints['orbits'] != mbSubsets[i].slicer.slicePoints['orbits']):
                raise ValueError('Bundle %s has a different slicer than the first bundle' % (i))
    # Join metric values.
    joint.slicer = slicer
    joint.metric = first.metric
    # Don't just use the slicer shape to define the metricValues, because of CompletenessBundles.
    metricValues = np.zeros(first.metricValues.shape, float)
    metricValuesMask = np.zeros(first.metricValues.shape, bool)
    for i in mbSubsets:
        metricValues += mbSubsets[i].metricValues.filled(0)
        metricValuesMask = np.where(metricValuesMask & mbSubsets[i].metricValues.mask, True, False)
    joint.metricValues = ma.MaskedArray(data=metricValues, mask=metricValuesMask, fill_value=0)
    joint.metadata = first.metadata
    joint.runName = first.runName
    joint.fileRoot = first.fileRoot.replace('.npz', '')
    return joint