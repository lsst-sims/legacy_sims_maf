"""Sets of slew metrics.
"""
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardMetrics

__all__ = ['slewBasics', 'slewAngles', 'slewSpeeds', 'slewActivities']


def slewBasics(colmap=None, runName='opsim'):
    """Generate a simple set of statistics about the slew times and distances.
    These slew statistics can be run on the summary or default tables.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".

    Returns
    -------
    metricBundleDict
    """

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    bundleList = []

    # Calculate basic stats on slew times. (mean/median/min/max + total).
    sql = ''
    slicer = slicers.UniSlicer()

    metadata = 'All visits'
    displayDict = {'group': 'Slew', 'subgroup': 'Slew Basics', 'order': -1, 'caption': None}
    # Add total number of slews.
    metric = metrics.CountMetric(colmap['slewtime'], metricName='Slew Count')
    displayDict['caption'] = 'Total number of slews recorded in summary table.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata, displayDict=displayDict)
    bundleList.append(bundle)
    for metric in standardMetrics(colmap['slewtime']):
        displayDict['caption'] =  '%s in seconds.' % (metric.name)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata, displayDict=displayDict)
        bundleList.append(bundle)

    # Slew Time histogram.
    sql = ''
    slicer = slicers.OneDSlicer(sliceColName=colmap['slewtime'], binsize=2)
    metric = metrics.CountMetric(col=colmap['slewtime'], metricName='Slew Time Histogram')
    metadata = 'All visits'
    plotDict = {'logScale': True, 'ylabel': 'Count'}
    displayDict['caption'] = 'Histogram of slew times (seconds) for all visits.'
    displayDict['order'] += 1
    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                             plotDict=plotDict, displayDict=displayDict)
    bundleList.append(bundle)

    # Slew distance histogram, if available.
    if colmap['slewdist'] is not None:
        sql = ''
        slicer = slicers.OneDSlicer(sliceColName=colmap['slewdist'])
        metric = metrics.CountMetric(col=colmap['slewdist'], metricName='Slew Distance Histogram')
        plotDict = {'logScale': True, 'ylabel': 'Count'}
        displayDict['caption'] = 'Histogram of slew distances (angle) for all visits.'
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 plotDict=plotDict, displayDict=displayDict)
        bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)



def slewAngles(colmap=None, runName='opsim'):
    """Generate a set of slew statistics focused on the angles of the various components (dome and telescope).
    These slew statistics must be run on the SlewFinalState or SlewInitialState table in opsimv4,
    and on the SlewState table in opsimv3.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics are run with a unislicer.
    slicer = slicers.UniSlicer()
    # And on all of the slew state data.
    sqlconstraint = ''

    # For each angle, we will compute mean/median/min/max and rms.
    # Note that these angles can range over more than 360 degrees, because of cable wrap.
    # This is why we're not using the Angle metrics - here 380 degrees is NOT the same as 20 deg.
    # Stats for angle:
    angles = ['Tel Alt', 'Tel Az', 'Rot Tel Pos']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Angles', 'order': -1, 'caption': None}
    for angle in angles:
        metadata = angle
        metriclist = standardMetrics(colmap[angle], strip_colname=True)
        metriclist += [metrics.RmsMetric(colmap[angle], metricName='RMS')]
        for metric in metriclist:
            displayDict['caption'] = '%s %s' % (metric.name, angle)
            displayDict['order'] += 1
            bundle = mb.MetricBundle(metric, slicer, sqlconstraint,
                                     displayDict=displayDict, metadata=metadata)
            bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def slewSpeeds(colmap=None, runName='opsim'):
    """Generate a set of slew statistics focused on the speeds of the various components (dome and telescope).
    These slew statistics must be run on the SlewMaxSpeeds table in opsimv4 and opsimv3.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
        Note that for these metrics, the column names are distinctly different in v3/v4.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics run with a unislicer, on all the slew data.
    slicer = slicers.UniSlicer()
    sqlconstraint = ''

    speeds = ['Dome Alt Speed', 'Dome Az Speed', 'Tel Alt Speed', 'Tel Az Speed', 'Rotator Speed']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Speeds', 'order': -1, 'caption': None}
    for speed in speeds:
        metadata = speed
        metric = metrics.AbsMaxMetric(col=colmap[speed], metricName='Max (Abs)')
        displayDict['caption'] = 'Maximum absolute value of %s.' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.AbsMeanMetric(col=colmap[speed], metricName='Mean (Abs)')
        displayDict['caption'] = 'Mean absolute value of %s.' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.AbsMaxPercentMetric(col=colmap[speed], metricName='% @ Max')
        displayDict['caption'] = 'Percent of slews at the maximum %s (absolute value).' % speed
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)

    return mb.makeBundlesDictFromList(bundleList)


def slewActivities(totalSlewN, colmap=None, runName='opsim'):
    """Generate a set of slew statistics focused on finding the contributions to the overall slew time.
    These slew statistics must be run on the SlewActivities table in opsimv4 and opsimv3.

    Note that the type of activities listed are different between v3 and v4.

    Parameters
    ----------
    totalSlewN : int
        The total number of slews in the simulated survey.
        Used to calculate % of slew activities for each component.
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    run_name : str, opt
        The name of the simulated survey. Default is "opsim".

    Returns
    -------
    metricBundleDict
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    # All of these metrics run with a unislicer, on all the slew data.
    slicer = slicers.UniSlicer()

    slewTypes = colMap['slewactivities']

    displayDict = {'group': 'Slew', 'subgroup': 'Slew Activities', 'order': -1, 'caption': None}

    for slewType in slewTypes:
        metadata = slewType
        tableValue = colMap[slewType]

        # Metrics for all activities of this type.
        sqlconstraint = 'activityDelay>0 and activity="%s"' % tableValue

        metric = metrics.CountRatioMetric(col='activityDelay', normVal=totalSlewN / 100.0,
                                          metricName='ActivePerc')
        displayDict['caption'] = 'Percent of total slews which include %s movement.' % slewType
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.MeanMetric(col='activityDelay', metricName='ActiveAve')
        displayDict['caption'] = 'Mean amount of time (in seconds) for %s movements.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.MaxMetric(col='activityDelay', metricName='Max')
        displayDict['caption'] = 'Max amount of time (in seconds) for %s movement.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        # Metrics for activities of this type which are in the critical path.
        sqlconstraint = 'activityDelay>0 and inCriticalPath="True" and activity="%s"' % tableValue

        metric = metrics.CountRatioMetric(col='activityDelay', normVal=totalSlewN / 100.0,
                                          metricName='ActivePerc in crit')
        displayDict['caption'] = 'Percent of total slew which include %s movement, ' \
                                 'and are in critical path.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

        metric = metrics.MeanMetric(col='activityDelay', metricName='ActiveAve in crit')
        displayDict['caption'] = 'Mean time (in seconds) for %s movements, ' \
                                 'when in critical path.' % (slewType)
        displayDict['order'] += 1
        bundle = mb.MetricBundle(metric, slicer, sqlconstraint, displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
