"""Sets of metrics to look at the SRD metrics.
"""
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['astrometryBatch']


def astrometryBatch(colmap=None, runName='opsim',
                    extraSql=None, extraMetadata=None,
                    nside=64):
    # Allow user to add dithering.
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    sql = ''
    metadata = 'All visits'
    # Add additional sql constraint (such as wfdWhere) and metadata, if provided.
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraMetadata is None:
            metadata = extraSql.replace('filter =', '').replace('filter=', '')
            metadata = metadata.replace('"', '').replace("'", '')
    if extraMetadata is not None:
        metadata = extraMetadata

    if subgroup is None:
        subgroup = metadata

    raCol = colmap['ra']
    decCol = colmap['dec']
    degrees = colmap['raDecDeg']

    # Set up stackers.


    # Set up parallax metrics.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)
    displayDict = {'group': 'Parallax', 'subgroup': subgroup,
                   'order': 0}
    rmag = 20.0
    displayDict['caption'] = 'Parallax precision at r=%.1f (without refraction).' % (rmag)
    metric = metrics.ParallaxMetric(metricName='Parallax %.1f' % (rmag), rmag=rmag,
                                    seeingCol=colmap['seeingFWHM'])
    plotDict = {'cbarFormat': '%.1f', 'xMin': 0, 'xMax': 3}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=standardSummary(),
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxMetric(metricName='Parallax 24', rmag=24, seeingCol=seeingCol)
    plotDict = {'cbarFormat': '%.1f', 'xMin': 0, 'xMax': 10}
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption': 'Parallax precision at r=24. (without refraction).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxMetric(metricName='Parallax Normed', rmag=24, normalize=True,
                                    seeingCol=seeingCol)
    plotDict = {'xMin': 0.5, 'xMax': 1.0}
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption':
                   'Normalized parallax (normalized to optimum observation cadence, 1=optimal).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage 20', rmag=20, seeingCol=seeingCol)
    plotDict = {}
    caption = "Parallax factor coverage for an r=20 star (0 is bad, 0.5-1 is good). "
    caption += "One expects the parallax factor coverage to vary because stars on the ecliptic "
    caption += "can be observed when they have no parallax offset while stars at the pole are always "
    caption += "offset by the full parallax offset."""
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage 24', rmag=24, seeingCol=seeingCol)
    plotDict = {}
    caption = "Parallax factor coverage for an r=24 star (0 is bad, 0.5-1 is good). "
    caption += "One expects the parallax factor coverage to vary because stars on the ecliptic "
    caption += "can be observed when they have no parallax offset while stars at the pole are always "
    caption += "offset by the full parallax offset."""
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy 20', rmag=20,
                                            seeingCol=seeingCol)
    plotDict = {}
    caption = 'Correlation between parallax offset magnitude and hour angle an r=20 star.'
    caption += ' (0 is good, near -1 or 1 is bad).'
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy 24', rmag=24,
                                            seeingCol=seeingCol)
    plotDict = {}
    caption = 'Correlation between parallax offset magnitude and hour angle an r=24 star.'
    caption += ' (0 is good, near -1 or 1 is bad).'
    displayDict = {'group': reqgroup, 'subgroup': 'Parallax', 'order': order,
                   'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1

    metric = metrics.ProperMotionMetric(metricName='Proper Motion 20', rmag=20, seeingCol=seeingCol)

    summaryStats = allStats
    plotDict = {'xMin': 0, 'xMax': 3}
    displayDict = {'group': reqgroup, 'subgroup': 'Proper Motion', 'order': order,
                   'caption': 'Proper Motion precision at r=20.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ProperMotionMetric(rmag=24, metricName='Proper Motion 24', seeingCol=seeingCol)
    summaryStats = allStats
    plotDict = {'xMin': 0, 'xMax': 10}
    displayDict = {'group': reqgroup, 'subgroup': 'Proper Motion', 'order': order,
                   'caption': 'Proper Motion precision at r=24.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ProperMotionMetric(rmag=24, normalize=True, metricName='Proper Motion Normed',
                                        seeingCol=seeingCol)
    plotDict = {'xMin': 0.2, 'xMax': 0.7}
    caption = 'Normalized proper motion at r=24. '
    caption += '(normalized to optimum observation cadence - start/end. 1=optimal).'
    displayDict = {'group': reqgroup, 'subgroup': 'Proper Motion', 'order': order,
                   'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
