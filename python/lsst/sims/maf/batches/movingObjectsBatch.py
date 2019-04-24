from __future__ import print_function, division
from copy import deepcopy
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import summaryCompletenessAtTime, summaryCompletenessOverH
import warnings

__all__ = ['setupMoSlicer', 'quickDiscoveryBatch', 'discoveryBatch', 'addMoCompletenessBundles',
           'characterizationBatch', 'combineSubsets']


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
                        albedo=None, Hmark=None, npReduce=np.mean, times=None, constraint=None):
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

    if times is None:
        try:
            timestep = 30
            times = np.arange(slicer.obs[colmap['mjd']].min(), slicer.obs[colmap['mjd']].max() + timestep/2,
                              timestep)
        except AttributeError:
            raise warnings.warn('Cannot set times for completeness summary metrics. Will set up bundles, '
                                'but without summary metrics.')

    if Hmark is None:
        Hval = slicer.Hrange.mean()
    else:
        Hval = Hmark

    # Set up the summary metrics.
    if times is not None:
        summaryTimeMetrics = summaryCompletenessAtTime(times, Hval=Hval, Hindex=0.33)
    else:
        summaryTimeMetrics = None
    summaryHMetrics = summaryCompletenessOverH(requiredChances=1, Hindex=0.33)

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
        parentBundle.childBundles['Time'].setSummaryMetrics(summaryTimeMetrics)
        dispDict = {'group': 'Discovery', 'subgroup': 'NChances',
                    'caption': 'Number of chances for discovery of objects', 'order': 0}
        parentBundle.childBundles['N_Chances'].setDisplayDict(dispDict)
        parentBundle.childBundles['N_Chances'].setSummaryMetrics(summaryHMetrics)
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
                   albedo=None, Hmark=None, npReduce=np.mean, times=None, constraint=None):
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

    if times is None:
        try:
            timestep = 30
            times = np.arange(slicer.obs[colmap['mjd']].min(), slicer.obs[colmap['mjd']].max() + timestep/2,
                              timestep)
        except AttributeError:
            raise warnings.warn('Cannot set times for completeness summary metrics. Will set up bundles, '
                                'but without summary metrics.')

    if Hmark is None:
        Hval = slicer.Hrange.mean()
    else:
        Hval = Hmark

    # Set up the summary metrics.
    if times is not None:
        summaryTimeMetrics = summaryCompletenessAtTime(times, Hval=Hval, Hindex=0.33)
    else:
        summaryTimeMetrics = None
    summaryHMetrics = summaryCompletenessOverH(requiredChances=1, Hindex=0.33)

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
        parentBundle.childBundles['Time'].setSummaryMetrics(summaryTimeMetrics)
        dispDict = {'group': 'Discovery', 'subgroup': 'NChances',
                    'caption': 'Number of chances for discovery of objects', 'order': 0}
        parentBundle.childBundles['N_Chances'].setDisplayDict(dispDict)
        parentBundle.childBundles['N_Chances'].setSummaryMetrics(summaryHMetrics)
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
                                summaryMetrics=summaryTimeMetrics,
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
                                summaryMetrics=summaryHMetrics,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles

def addMoCompletenessBundles(bdict, Hmark, outDir, resultsDb):
    """
    Generate completeness bundles from all N_Chances and Time child metrics of the (discovery) bundles in
    bdict, and write completeness at Hmark to resultsDb, save bundle to disk.

    Parameters
    ----------
    bdict : dict of metricBundles
        Dict containing ~lsst.sims.maf.MoMetricBundles,
        including bundles we're expecting to contain completeness.
    Hmark : float
        Hmark value to add to completeness plotting dict.
    outDir : str
        Output directory to save completeness bundles to disk.
    resultsDb : ~lsst.sims.maf.db.ResultsDb
        Results database to save information about completeness bundle.

    Returns
    -------
    dict of metricBundles
        Now the resulting metricBundles also includes new nested dicts with keys "DifferentialCompleteness"
        and "CumulativeCompleteness", which contain bundles of completeness metrics at each year.
    """
    # Add completeness bundles and write completeness at Hmark to resultsDb.
    completeness = {}
    group = 'Discovery'
    subgroup = 'Completeness @ H=%.1f' % (Hmark)

    def _compbundles(b, bundle, Hmark, resultsDb):
        comp = {}
        newkey = b + ' differential completeness'
        comp[newkey] = mb.makeCompletenessBundle(bundle, summaryName='DifferentialCompleteness',
                                                 Hmark=Hmark, resultsDb=resultsDb)
        newkey = b + ' cumulative completeness'
        comp[newkey] = mb.makeCompletenessBundle(bundle, summaryName='CumulativeCompleteness',
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


def characterizationBatch(slicer, colmap=None, runName='opsim', metadata='',
                          albedo=None, Hmark=None, npReduce=np.mean, constraint=None,
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

    if windows is None:
        windows = np.arange(1, 200, 15.)
    if bins is None:
        bins = np.arange(5, 95, 10.)

    # Number of observations.
    md = metadata
    plotDict = {'ylabel': 'Number of observations (#)',
                'title': '%s: Number of observations %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.NObsMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Observational arc.
    md = metadata
    plotDict = {'ylabel': 'Observational Arc (days)',
                'title': '%s: Observational Arc Length %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ObsArcMetric(**colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
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
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    for b in bins:
        md = metadata + ' activity lasting %.2f of period' % (b/360.)
        plotDict = {'title': '%s: Chances of detecting %s' % (runName, md),
                    'ylabel': 'Probability of detection per %.2f deg window' % b}
        metricName = 'Chances of detecting activity lasting %.2f of the period' % (b/360.)
        metric = metrics.ActivityOverPeriodMetric(b, metricName=metricName, **colkwargs)
        bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                    runName=runName, metadata=metadata,
                                    plotDict=plotDict, plotFuncs=plotFuncs,
                                    displayDict=displayDict)
        bundleList.append(bundle)

    # Lightcurve inversion.
    md = metadata
    plotDict = {'yMin': 0, 'yMax': 1, 'ylabel': 'Fraction of objects',
                'title': '%s: Fraction with potential lightcurve inversion %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.LightcurveInversionMetric(snrLimit=20, nObs=100, nDays=5*365, **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Color determination.
    snrLimit = 10
    nHours = 2.0
    nPairs = 1
    md = metadata + ' u-g color'
    plotDict = {'label': md,
                'title': '%s: Fraction with potential u-g color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='u', bTwo='g', **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    md = metadata + ' g-r color'
    plotDict = {'label': md,
                'title': '%s: Fraction with potential g-r color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='g', bTwo='r', **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    md = metadata + ' r-i color'
    plotDict = {'label': md,
                'title': '%s: Fraction with potential r-i color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='r', bTwo='i', **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    md = metadata + ' i-z color'
    plotDict = {'label': md,
                'title': '%s: Fraction with potential i-z color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='i', bTwo='z', **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    md = metadata + ' z-y color'
    plotDict = {'label': md,
                'title': '%s: Fraction with potential z-y color measurement %s' % (runName, md)}
    plotDict.update(basicPlotDict)
    metric = metrics.ColorDeterminationMetric(nPairs=nPairs, snrLimit=snrLimit, nHours=nHours,
                                              bOne='z', bTwo='y', **colkwargs)
    bundle = mb.MoMetricBundle(metric, slicer, constraint,
                                runName=runName, metadata=md,
                                plotDict=plotDict, plotFuncs=plotFuncs,
                                summaryMetrics=None,
                                displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList), plotBundles


def combineSubsets(mbSubsets):
    # Combine the data from the subsets.
    # The first bundle will be used as a bit of a template.
    joint = mb.createEmptyMoMetricBundle()
    # Check if they're the same slicer.
    slicer = deepcopy(mbSubsets[0].slicer)
    for i in mbSubsets:
        if np.any(slicer.slicePoints['H'] != mbSubsets[i].slicer.slicePoints['H']):
            if np.any(slicer.slicePoints['orbits'] != mbSubsets[i].slicer.slicePoints['orbits']):
                raise ValueError('Bundle %s has a different slicer than the first bundle' % (i))
    # Join metric values.
    joint.slicer = slicer
    joint.metric = mbSubsets[0].metric
    # Don't just use the slicer shape to define the metricValues, because of CompletenessBundles.
    metricValues = np.zeros(mbSubsets[0].metricValues.shape, float)
    metricValuesMask = np.zeros(mbSubsets[0].metricValues.shape, bool)
    for i in mbSubsets:
        metricValues += mbSubsets[i].metricValues.filled(0)
        metricValuesMask = np.where(metricValuesMask & mbSubsets[i].metricValues.mask, True, False)
    joint.metricValues = ma.MaskedArray(data=metricValues, mask=metricValuesMask, fill_value=0)
    joint.metadata = mbSubsets[0].metadata
    joint.runName = mbSubsets[0].runName
    joint.fileRoot = metricfile.replace('.npz', '')
    return joint