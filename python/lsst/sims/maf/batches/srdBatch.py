"""Sets of metrics to look at the SRD metrics.
"""
import healpy as hp
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary, radecCols

__all__ = ['fOBatch', 'astrometryBatch', 'rapidRevisitBatch']


def fOBatch(colmap=None, runName='opsim', extraSql=None, extraMetadata=None, nside=64,
            benchmarkArea=18000, benchmarkNvisits=825, ditherStacker=None):

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    raCol, decCol, degrees, ditherStacker = radecCols(ditherStacker, colmap)

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

    subgroup = metadata

    # Set up fO metric.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)

    displayDict = {'group': 'FO metrics', 'subgroup': subgroup, 'order': 0}

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountMetric(col=colmap['mjd'], metricName='fO')
    plotDict = {'xlabel': 'Number of Visits', 'Asky': benchmarkArea,
                'Nvisit': benchmarkNvisits, 'xMin': 0, 'xMax': 1500}
    summaryMetrics = [metrics.fOArea(nside=nside, norm=False, metricName='fOArea: Nvisits (#)',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fOArea(nside=nside, norm=True, metricName='fOArea: Nvisits/benchmark',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=False, metricName='fONv: Area (sqdeg)',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=True, metricName='fONv: Area/benchmark',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits)]
    caption = 'The FO metric evaluates the overall efficiency of observing. '
    caption += ('fOArea: Nvisits = %.1f sq degrees receive at least this many visits out of %d. '
                % (benchmarkArea, benchmarkNvisits))
    caption += ('fONv: Area = this many square degrees out of %.1f receive at least %d visits.'
                % (benchmarkArea, benchmarkNvisits))
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                             stackerList = [ditherStacker],
                             displayDict=displayDict, summaryMetrics=summaryMetrics,
                             plotFuncs=[plots.FOPlot()], metadata=metadata)
    bundleList.append(bundle)
    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def astrometryBatch(colmap=None, runName='opsim',
                    extraSql=None, extraMetadata=None,
                    nside=64, ditherStacker=None):
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

    subgroup = metadata

    raCol, decCol, degrees, ditherStacker = radecCols(ditherStacker, colmap)

    rmags_para = [22.4, 24.0]
    rmags_pm = [20.5, 24.0]

    # Set up stackers.
    parallaxStacker = stackers.ParallaxFactorStacker(raCol=raCol, decCol=decCol,
                                                     dateCol=colmap['mjd'], degrees=degrees)
    dcrStacker = stackers.DcrStacker(filterCol=colmap['filter'], altCol=colmap['alt'], degrees=degrees,
                                     raCol=raCol, decCol=decCol, lstCol=colmap['lst'],
                                     site='LSST', mjdCol=colmap['mjd'])

    # Set up parallax metrics.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    displayDict = {'group': 'Parallax', 'subgroup': subgroup,
                   'order': 0, 'caption': None}
    # Expected error on parallax at 10 AU.
    plotmaxVals = (2.0, 15.0)
    for rmag, plotmax in zip(rmags_para, plotmaxVals):
        plotDict = {'xMin': 0, 'xMax': plotmax, 'colorMin': 0, 'colorMax': plotmax}
        metric = metrics.ParallaxMetric(metricName='Parallax Error @ %.1f' % (rmag), rmag=rmag,
                                        seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                        m5Col=colmap['fiveSigmaDepth'], normalize=False)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Parallax normalized to 'best possible' if all visits separated by 6 months.
    # This separates the effect of cadence from depth.
    for rmag in rmags_para:
        metric = metrics.ParallaxMetric(metricName='Normalized Parallax @ %.1f' % (rmag), rmag=rmag,
                                        seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                        m5Col=colmap['fiveSigmaDepth'], normalize=True)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict,
                                 summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Parallax factor coverage.
    for rmag in rmags_para:
        metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage @ %.1f' % (rmag),
                                                rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                                mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                                seeingCol=colmap['seeingGeom'])
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[parallaxStacker, ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Parallax problems can be caused by HA and DCR degeneracies. Check their correlation.
    for rmag in rmags_para:
        metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy @ %.1f' % (rmag),
                                                rmag=rmag, seeingCol=colmap['seeingEff'],
                                                filterCol=colmap['filter'], m5Col=colmap['fiveSigmaDepth'])
        caption = 'Correlation between parallax offset magnitude and hour angle for a r=%.1f star.' % (rmag)
        caption += ' (0 is good, near -1 or 1 is bad).'
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[dcrStacker, parallaxStacker, ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Proper Motion metrics.
    displayDict = {'group': 'Proper Motion', 'subgroup': subgroup, 'order': 0, 'caption': None}
    # Proper motion errors.
    plotmaxVals = (1.0, 5.0)
    for rmag, plotmax in zip(rmags_pm, plotmaxVals):
        plotDict = {'xMin': 0, 'xMax': plotmax, 'colorMin': 0, 'colorMax': plotmax}
        metric = metrics.ProperMotionMetric(metricName='Proper Motion Error @ %.1f' % rmag,
                                            rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                            mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                            seeingCol=colmap['seeingGeom'], normalize=False)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[ditherStacker],
                                 displayDict=displayDict, plotDict=plotDict,
                                 summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1
    # Normalized proper motion.
    for rmag in rmags_pm:
        metric = metrics.ProperMotionMetric(metricName='Normalized Proper Motion @ %.1f' % rmag,
                                            rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                            mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                            seeingCol=colmap['seeingGeom'], normalize=True)
        bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                 stackerList=[ditherStacker],
                                 displayDict=displayDict, summaryMetrics=standardSummary(),
                                 plotFuncs=subsetPlots)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)


def rapidRevisitBatch(colmap=None, runName='opsim',
                      extraSql=None, extraMetadata=None, nside=64, ditherStacker=None):
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

    subgroup = metadata

    raCol, decCol, degrees, ditherStacker = radecCols(ditherStacker, colmap)

    # Set up parallax metrics.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=raCol, latCol=decCol, latLonDeg=degrees)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    displayDict = {'group': 'Rapid Revisits', 'subgroup': subgroup,
                   'order': 0, 'caption': None}

    # Calculate the uniformity (KS test) of the quick revisits.
    dTmin = 40.0  # seconds
    dTmax = 30.0  # minutes
    minNvisit = 100
    pixArea = float(hp.nside2pixarea(nside, degrees=True))
    scale = pixArea * hp.nside2npix(nside)
    m1 = metrics.RapidRevisitMetric(metricName='RapidRevisitUniformity', mjdCol=colmap['mjd'],
                                    dTmin=dTmin / 60.0 / 60.0 / 24.0, dTmax=dTmax / 60.0 / 24.0,
                                    minNvisits=minNvisit)

    plotDict = {'xMin': 0, 'xMax': 1}
    cutoff1 = 0.20
    summaryStats = [metrics.FracBelowMetric(cutoff=cutoff1, scale=scale, metricName='Area (sq deg)')]
    summaryStats.extend(standardSummary())
    caption = 'Deviation from uniformity for short revisit timescales, between %s and %s seconds, ' % (
        dTmin, dTmax)
    caption += 'for pointings with at least %d visits in this time range. ' % (minNvisit)
    caption += 'Summary statistic "Area" indicates the area on the sky which has a '
    caption += 'deviation from uniformity of < %.2f.' % (cutoff1)
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(m1, slicer, sql, plotDict=plotDict, plotFuncs=subsetPlots,
                             stackerList=[ditherStacker],
                             metadata=metadata, displayDict=displayDict, summaryMetrics=summaryStats)
    bundleList.append(bundle)
    displayDict['order'] += 1

    # Calculate the actual number of quick revisits.
    dTmax = dTmax   # time in minutes
    m2 = metrics.NRevisitsMetric(dT=dTmax, mjdCol=colmap['mjd'], normed=False)
    plotDict = {'xMin': 0.1, 'xMax': 2000, 'logScale': True}
    cutoff2 = 800
    summaryStats = [metrics.FracAboveMetric(cutoff=cutoff2, scale=scale, metricName='Area (sq deg)')]
    summaryStats.extend(standardSummary())
    caption = 'Number of consecutive visits with return times faster than %.1f minutes, ' % (dTmax)
    caption += 'in any filter, all proposals. '
    caption += 'Summary statistic "Area" indicates the area on the sky which has more than '
    caption += '%d revisits within this time window.' % (cutoff2)
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(m2, slicer, sql, plotDict=plotDict, plotFuncs=subsetPlots,
                             stackerList=[ditherStacker],
                             metadata=metadata, displayDict=displayDict, summaryMetrics=summaryStats)
    bundleList.append(bundle)
    displayDict['order'] += 1

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)

