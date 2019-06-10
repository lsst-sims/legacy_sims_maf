import healpy as hp
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
import numpy as np
from lsst.sims.utils import hpid2RaDec, angularSeparation

__all__ = ['scienceRadarBatch']


def scienceRadarBatch(colmap=None, runName='', extraSql=None, extraMetadata=None, nside=64,
                      benchmarkArea=18000, benchmarkNvisits=825, DDF=True):
    """A batch of metrics for looking at survey performance relative to the SRD and the main
    science drivers of LSST.

    Parameters
    ----------

    """
    # Hide dependencies
    from mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended
    from mafContrib import Plasticc_metric, plasticc_slicer, load_plasticc_lc

    if colmap is None:
        colmap = ColMapDict('opsimV4')

    if extraSql is None:
        extraSql = ''
    if extraSql == '':
        joiner = ''
    else:
        joiner = ' and '

    bundleList = []

    healslicer = slicers.HealpixSlicer(nside=nside)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    # Load up the plastic light curves
    models = ['SNIa-normal', 'KN']
    plasticc_models_dict = {}
    for model in models:
        plasticc_models_dict[model] = list(load_plasticc_lc(model=model).values())

    #########################
    # SRD, DM, etc
    #########################
    sql = extraSql
    displayDict = {'group': 'SRD', 'subgroup': 'fO', 'order': 0, 'caption': None}
    metric = metrics.CountMetric(col=colmap['mjd'], metricName='fO')
    plotDict = {'xlabel': 'Number of Visits', 'Asky': benchmarkArea,
                'Nvisit': benchmarkNvisits, 'xMin': 0, 'xMax': 1500}
    summaryMetrics = [metrics.fOArea(nside=nside, norm=False, metricName='fOArea',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fOArea(nside=nside, norm=True, metricName='fOArea/benchmark',
                                     Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=False, metricName='fONv',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits),
                      metrics.fONv(nside=nside, norm=True, metricName='fONv/benchmark',
                                   Asky=benchmarkArea, Nvisit=benchmarkNvisits)]
    caption = 'The FO metric evaluates the overall efficiency of observing. '
    caption += ('foNv: out of %.2f sq degrees, the area receives at least X and a median of Y visits '
                '(out of %d, if compared to benchmark). ' % (benchmarkArea, benchmarkNvisits))
    caption += ('fOArea: this many sq deg (out of %.2f sq deg if compared '
                'to benchmark) receives at least %d visits. ' % (benchmarkArea, benchmarkNvisits))
    displayDict['caption'] = caption
    bundle = mb.MetricBundle(metric, healslicer, sql, plotDict=plotDict,
                             displayDict=displayDict, summaryMetrics=summaryMetrics,
                             plotFuncs=[plots.FOPlot()])
    bundleList.append(bundle)
    displayDict['order'] += 1

    displayDict = {'group': 'SRD', 'subgroup': 'Gaps', 'order': 0, 'caption': None}
    plotDict = {'percentileClip': 95.}
    for filtername in 'ugrizy':
        sql = extraSql + joiner + 'filter ="%s"' % filtername
        metric = metrics.MaxGapMetric()
        summaryMetrics = [metrics.PercentileMetric(percentile=95, metricName='95th percentile of Max gap, %s' % filtername)]
        bundle = mb.MetricBundle(metric, healslicer, sql, plotFuncs=subsetPlots,
                                 summaryMetrics=summaryMetrics, displayDict=displayDict, plotDict=plotDict)
        bundleList.append(bundle)
        displayDict['order'] += 1

    #########################
    # Solar System
    #########################

    # XXX -- may want to do Solar system seperatly

    # XXX--fraction of NEOs detected (assume some nominal size and albido)
    # XXX -- fraction of MBAs detected
    # XXX -- fraction of KBOs detected
    # XXX--any others? Planet 9s? Comets? Neptune Trojans?

    #########################
    # Cosmology
    #########################

    displayDict = {'group': 'Cosmology', 'subgroup': 'galaxy counts', 'order': 0, 'caption': None}
    plotDict = {'percentileClip': 95.}
    sql = extraSql + joiner + 'filter="i"'
    metric = GalaxyCountsMetric_extended(filterBand='i', redshiftBin='all', nside=nside)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.sum, decreasing=True, metricName='N Galaxies (WFD)')]
    summary.append(metrics.SumMetric(metricName='N Galaxies (all)'))
    # make sure slicer has cache off
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    bundle = mb.MetricBundle(metric, slicer, sql, plotDict=plotDict,
                             displayDict=displayDict, summaryMetrics=summary,
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1

    # let's put Type Ia SN in here
    displayDict['subgroup'] = 'SNe Ia'
    metadata = ''
    # XXX-- use the light curves from PLASTICC here
    displayDict['Caption'] = 'Fraction of normal SNe Ia'
    sql = ''
    slicer = plasticc_slicer(plcs=plasticc_models_dict['SNIa-normal'], seed=42, badval=0)
    metric = Plasticc_metric(metricName='SNIa')
    # Set the maskval so that we count missing objects as zero.
    summary_stats = [metrics.MeanMetric(maskVal=0)]
    plotFuncs = [plots.HealpixSkyMap()]
    bundle = mb.MetricBundle(metric, slicer, sql, runName=runName, summaryMetrics=summary_stats,
                             plotFuncs=plotFuncs, metadata=metadata, displayDict=displayDict)
    bundleList.append(bundle)
    displayDict['order'] += 1

    # XXX--need some sort of metric for weak lensing and camera rotation.

    #########################
    # Variables and Transients
    #########################
    displayDict = {'group': 'Variables and Transients', 'subgroup': 'Periodic Stars',
                   'order': 0, 'caption': None}
    periods = [0.1, 0.5, 1., 2., 5., 10., 20.]  # days

    plotDict = {}
    metadata = ''
    sql = extraSql
    displayDict['Caption'] = 'Measure of how well a periodic signal can be measured combining amplitude and phase coverage. 1 is perfect, 0 is no way to fit'
    for period in periods:
        summary = metrics.PercentileMetric(percentile=10., metricName='10th %%-ile Periodic Quality, Period=%.1f days' % period)
        metric = metrics.PeriodicQualityMetric(period=period, starMag=20., metricName='Periodic Stars, P=%.1f d' % period)
        bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                                 displayDict=displayDict, plotDict=plotDict,
                                 plotFuncs=subsetPlots, summaryMetrics=summary)
        bundleList.append(bundle)
        displayDict['order'] += 1

    # XXX add some PLASTICC metrics for kilovnova and tidal disruption events.
    displayDict['subgroup'] = 'KN'
    displayDict['caption'] = 'Fraction of Kilonova (from PLASTICC)'
    sql = ''
    slicer = plasticc_slicer(plcs=plasticc_models_dict['KN'], seed=43, badval=0)
    metric = Plasticc_metric(metricName='KN')
    summary_stats = [metrics.MeanMetric(maskVal=0)]
    plotFuncs = [plots.HealpixSkyMap()]
    bundle = mb.MetricBundle(metric, slicer, sql, runName=runName, summaryMetrics=summary_stats,
                             plotFuncs=plotFuncs, metadata=metadata,
                             displayDict=displayDict)
    bundleList.append(bundle)

    displayDict['order'] += 1

    # XXX -- would be good to add some microlensing events, for both MW and LMC/SMC.

    #########################
    # Milky Way
    #########################

    # Let's do the proper motion, parallax, and DCR degen of a 20nd mag star
    rmag = 20.
    displayDict = {'group': 'Milky Way', 'subgroup': 'Astrometry',
                   'order': 0, 'caption': None}

    sql = extraSql
    metadata = ''
    plotDict = {'percentileClip': 95.}
    metric = metrics.ParallaxMetric(metricName='Parallax Error r=%.1f' % (rmag), rmag=rmag,
                                    seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                    m5Col=colmap['fiveSigmaDepth'], normalize=False)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=False, metricName='Median Parallax Error (WFD)')]
    summary.append(metrics.PercentileMetric(percentile=95, metricName='95th Percentile Parallax Error'))
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, plotDict=plotDict,
                             plotFuncs=subsetPlots, summaryMetrics=summary)
    bundleList.append(bundle)
    displayDict['order'] += 1

    metric = metrics.ProperMotionMetric(metricName='Proper Motion Error r=%.1f' % rmag,
                                        rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                        mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                        seeingCol=colmap['seeingGeom'], normalize=False)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=False, metricName='Median Proper Motion Error (WFD)')]
    summary.append(metrics.PercentileMetric(metricName='95th Percentile Proper Motion Error'))
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, plotDict=plotDict,
                             summaryMetrics=summary, plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1

    metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy r=%.1f' % (rmag),
                                            rmag=rmag, seeingCol=colmap['seeingEff'],
                                            filterCol=colmap['filter'], m5Col=colmap['fiveSigmaDepth'])
    caption = 'Correlation between parallax offset magnitude and hour angle for a r=%.1f star.' % (rmag)
    caption += ' (0 is good, near -1 or 1 is bad).'
    # XXX--not sure what kind of summary to do here
    summary = [metrics.MeanMetric(metricName='Mean DCR Degeneracy')]
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, summaryMetrics=summary,
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1

    for b in bundleList:
        b.setRunName(runName)

    #########################
    # DDF
    #########################

    if DDF:
        # Hide this import to avoid adding a dependency.
        from lsst.sims.featureScheduler.surveys import generate_dd_surveys
        ddf_surveys = generate_dd_surveys()
        # For doing a high-res sampling of the DDF for co-adds
        ddf_radius = 3.0  # Degrees
        ddf_nside = 512

        ra, dec = hpid2RaDec(ddf_nside, np.arange(hp.nside2npix(ddf_nside)))

        displayDict = {'group': 'DDF depths', 'subgroup': None,
                       'order': 0, 'caption': None}

        for survey in ddf_surveys:
            displayDict['subgroup'] = survey.survey_name
            # Crop off the u-band only DDF
            if survey.survey_name[0:4] != 'DD:u':
                dist_to_ddf = angularSeparation(ra, dec, np.degrees(survey.ra), np.degrees(survey.dec))
                goodhp = np.where(dist_to_ddf <= ddf_radius)
                slicer = slicers.UserPointsSlicer(ra=ra[goodhp], dec=dec[goodhp], useCamera=False)
                for filtername in ['u', 'g', 'r', 'i', 'z', 'y']:
                    metric = metrics.Coaddm5Metric(metricName=survey.survey_name+', ' + filtername)
                    summary = [metrics.MedianMetric(metricName='median depth ' + survey.survey_name+', ' + filtername)]
                    sql = extraSql + joiner + 'filter = "%s"' % filtername
                    bundle = mb.MetricBundle(metric, slicer, sql, metadata=metadata,
                                             displayDict=displayDict, summaryMetrics=summary,
                                             plotFuncs=[])
                    bundleList.append(bundle)
                    displayDict['order'] += 1

        displayDict = {'group': 'DDF Transients', 'subgroup': None,
                       'order': 0, 'caption': None}
        for survey in ddf_surveys:
            displayDict['subgroup'] = survey.survey_name
            if survey.survey_name[0:4] != 'DD:u':
                slicer = plasticc_slicer(plcs=plasticc_models_dict['SNIa-normal'], seed=42,
                                         ra_cen=survey.ra, dec_cen=survey.dec, radius=np.radians(3.), useCamera=False)
                metric = Plasticc_metric(metricName=survey.survey_name+' SNIa')
                sql = ''
                summary_stats = [metrics.MeanMetric(maskVal=0)]
                plotFuncs = [plots.HealpixSkyMap()]
                bundle = mb.MetricBundle(metric, slicer, sql, runName=runName, summaryMetrics=summary_stats,
                                         plotFuncs=plotFuncs, metadata=metadata,
                                         displayDict=displayDict)
                bundleList.append(bundle)

    displayDict['order'] += 1

    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)
