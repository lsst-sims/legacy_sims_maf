import healpy as hp
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
import numpy as np
import os
from mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended
from lsst.sims.maf.metrics.snCadenceMetric import SNcadenceMetric
from lsst.sims.maf.metrics.snSNRMetric import SNSNRMetric

from lsst.sims.maf.utils.snUtils import Lims, Reference_Data

__all__ = ['scienceRadarBatch']


def scienceRadarBatch(colmap=None, runName='', extraSql=None, extraMetadata=None, nside=64,
                      benchmarkArea=18000, benchmarkNvisits=825):
    """A batch of metrics for looking at survey performance relative to the SRD and the main
    science drivers of LSST.

    Parameters
    ----------

    """

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


    # XXX--Maybe a template available metric?


    #########################
    # Solar System
    #########################

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
    sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
    Li_files = [os.path.join(sims_maf_contrib_dir, 'data', 'Li_SNCosmo_-2.0_0.2.npy')]
    mag_to_flux_files = [os.path.join(sims_maf_contrib_dir, 'data', 'Mag_to_Flux_SNCosmo.npy')]
    config_fake = os.path.join(sims_maf_contrib_dir, 'data', 'Fake_cadence.yaml')
    SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
    mag_range = [21., 25.5]  # WFD mag range
    dt_range = [0.5, 30.]  # WFD dt range
    band = 'r'
    plotDict = {'percentileClip': 95.}
    lim_sn = Lims(Li_files, mag_to_flux_files, band, SNR[band], mag_range=mag_range, dt_range=dt_range)
    metric = SNcadenceMetric(lim_sn=lim_sn, coadd=False)
    sql = extraSql
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=True, metricName='Median SN Ia redshift (WFD)')]
    summary.append(metrics.MedianMetric(metricName='Median SN Ia redsihft (all)'))
    bundle = mb.MetricBundle(metric, healslicer, sql, displayDict=displayDict, plotFuncs=subsetPlots,
                             plotDict=plotDict, summaryMetrics=summary)
    bundleList.append(bundle)
    displayDict['order'] += 1

    names_ref = ['SNCosmo']
    z = 0.3
    lim_sn = Reference_Data(Li_files, mag_to_flux_files, band, z)
    metric = SNSNRMetric(lim_sn=lim_sn, coadd=False, names_ref=names_ref,
                         season=1, z=0.3, config_fake=config_fake)
    summary = [metrics.AreaSummaryMetric(area=18000, reduce_func=np.median, decreasing=True, metricName='Median SN Ia detection fraction (WFD)')]
    bundle = mb.MetricBundle(metric, healslicer, sql, displayDict=displayDict, plotFuncs=subsetPlots,
                             plotDict=plotDict, summaryMetrics=summary)
    bundleList.append(bundle)
    displayDict['order'] += 1


    # XXX--need some sort of metric for weak lensing and telescope rotation.

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

    # What are some good DDF-specific things? High resolution SN on each DDF or something?



    return mb.makeBundlesDictFromList(bundleList)


