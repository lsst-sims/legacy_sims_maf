from __future__ import print_function
from builtins import zip
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils

__all__ = ['glanceBundle']


def glanceBundle(colmap_dict=None):
    """Generate a handy set of metrics that give a quick overview of how well a survey performed

    Parameters
    ----------
    colmap_dict : dict
        A dictionary with a mapping of column names.

    Returns
    -------
    metricBundleDict
    """

    if colmap_dict is None:
        colmap_dict = utils.opsimColMapDict()

    bundleList = []

    filternames = ['u', 'g', 'r', 'i', 'z', 'y']
    sql_per_filt = ['%s="%s"' % (colmap_dict['filter'], filtername) for filtername in filternames]

    # Is the survey pointing intelligently?

    # The alt/az plots of all the pointings
    slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance',
                                   lonCol=colmap_dict['az'], useCache=False)
    stacker = stackers.ZenithDistStacker(altCol=colmap_dict['alt'])
    sql = ''
    metric = metrics.CountMetric(colmap_dict['mjd'], metricName='Nvisits as function of Alt/Az')
    plotFuncs = [plots.LambertSkyMap()]
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs, stackerList=[stacker])
    bundleList.append(bundle)

    # and per filter
    for sql in sql_per_filt:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, plotFuncs=plotFuncs, stackerList=[stacker])
        bundleList.append(bundle)

    # Things to check per night
    # Open Shutter per night

    # XXX--hit this with mean/median summary stats.
    slicer = slicers.OneDSlicer(sliceColName=colmap_dict['night'], binsize=1)
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap_dict['slewtime'],
                                               expTimeCol=colmap_dict['exptime'],
                                               visitTimeCol=colmap_dict['visittime'])
    sql = ''
    bundle = metricBundles.MetricBundle(metric, slicer, sql)
    bundleList.append(bundle)

    # Number of filter changes per night
    metric = metrics.NChangesMetric(col=colmap_dict['filter'], matricName='Filter Changes')
    bundle = metricBundles.MetricBundle(metric, slicer, sql)
    bundleList.append(bundle)

    # Slewtime distribution
    slicer = slicers.OneDSlicer(sliceColName=colmap_dict['slewtime'], binsize=2)
    metric = metrics.CountMetric(col=colmap_dict['slewtime'], metricName='Slew Time Histogram')
    bundle = metricBundles.MetricBundle(metric, slicer, sql, plotDict={'logScale': True, 'ylabel': 'Count'})
    bundleList.append(bundle)


    # A few basic maps
    # Number of observations, coadded depths


    # Checking a few basic science things
    # Maybe parallax and proper motion, fraction of visits in a good pair for SS, and SN detection & LC sampling? 


    bd = metricBundles.makeBundlesDictFromList(bundleList)
    return bd

