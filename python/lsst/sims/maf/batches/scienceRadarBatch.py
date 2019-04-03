import healpy as hp
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import radecCols, combineMetadata

__all__ = ['scienceRadarBatch']


def scienceRadarBatch(colmap=None, runName=None, extraSql=None, extraMetadata=None, nside=64,
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

    bundleList = []

    healslicer = slicers.HealpixSlicer(nside=nside)

    #########################
    # SRD
    #########################

    #########################
    # Solar System
    #########################

    #########################
    # Cosmology
    #########################

    #########################
    # Variables
    #########################

    #########################
    # Milky Way
    #########################

    # Let's do the proper motion, parallax, and DCR degen of a 20nd mag star
    rmag = 20.
    displayDict = {'group': 'Milky Way', 'subgroup': 'Astrometry',
                   'order': 0, 'caption': None}

    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
    sql = extraSql
    metadata = ''
    plotDict = {}
    metric = metrics.ParallaxMetric(metricName='Parallax Error @ %.1f' % (rmag), rmag=rmag,
                                    seeingCol=colmap['seeingGeom'], filterCol=colmap['filter'],
                                    m5Col=colmap['fiveSigmaDepth'], normalize=False)
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, plotDict=plotDict,
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1

    metric = metrics.ProperMotionMetric(metricName='Proper Motion Error @ %.1f' % rmag,
                                        rmag=rmag, m5Col=colmap['fiveSigmaDepth'],
                                        mjdCol=colmap['mjd'], filterCol=colmap['filter'],
                                        seeingCol=colmap['seeingGeom'], normalize=False)
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, plotDict=plotDict,
                             summaryMetrics=standardSummary(),
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1

    metric = metrics.ParallaxDcrDegenMetric(metricName='Parallax-DCR degeneracy @ %.1f' % (rmag),
                                            rmag=rmag, seeingCol=colmap['seeingEff'],
                                            filterCol=colmap['filter'], m5Col=colmap['fiveSigmaDepth'])
    caption = 'Correlation between parallax offset magnitude and hour angle for a r=%.1f star.' % (rmag)
    caption += ' (0 is good, near -1 or 1 is bad).'
    bundle = mb.MetricBundle(metric, healslicer, sql, metadata=metadata,
                             displayDict=displayDict, summaryMetrics=standardSummary(),
                             plotFuncs=subsetPlots)
    bundleList.append(bundle)
    displayDict['order'] += 1
