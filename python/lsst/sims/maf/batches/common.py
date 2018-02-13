from __future__ import print_function

import lsst.sims.maf.metrics as metrics

__all__ = ['filterList', 'standardSummary', 'extendedSummary', 'standardMetrics', 'extendedMetrics']


def filterList(all=True, extraSql=None, extraMetadata=None):
    """Return a list of filters, plot colors and orders.

    Parameters
    ----------
    all : boolean, opt
        Include 'all' in the list of filters and as part of the colors/orders dictionaries.
        Default True.
    extraSql : str, opt
        Additional sql constraint to add to sqlconstraints returned per filter.
        Default None.
    extraMetadata : str, opt
        Substitute metadata to add to metadata strings composed per band.
        Default None.

    Returns
    -------
    list, dict, dict
        List of filter names, dictionary of colors (for plots), dictionary of orders (for display)
    """
    if all:
        filterlist = ('all', 'u', 'g', 'r', 'i', 'z', 'y')
    else:
        filterlist = ('u', 'g', 'r', 'i', 'z', 'y')
    colors = {'u': 'cyan', 'g': 'g', 'r': 'orange', 'i': 'r', 'z': 'm', 'y': 'b'}
    orders = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6}
    if all:
        colors['all'] = 'k'
        orders['all'] = 0
    sqls = {}
    metadata = {}
    if extraMetadata is None:
        if extraSql is None or len(extraSql) == 0:
            md = ''
        else:
            md = '%s '  % extraSql
    else:
        md = '%s ' % extraMetadata
    for f in filterlist:
        if f == 'all':
            sqls[f] = ''
            metadata[f] = md + 'all bands'
        else:
            sqls[f] = 'filter = "%s"' % f
            metadata[f] = md + '%s band' % f
    if extraSql is not None and len(extraSql) > 0:
        for s in sqls:
            if s == 'all':
                sqls[s] = extraSql
            else:
                sqls[s] = '(%s) and (%s)' % (extraSql, sqls[s])
    return filterlist, colors, orders, sqls, metadata


def standardSummary():
    """A set of standard summary metrics, to calculate Mean, RMS, Median, #, Max/Min, and # 3-sigma outliers.
    """
    standardSummary = [metrics.MeanMetric(),
                       metrics.RmsMetric(),
                       metrics.MedianMetric(),
                       metrics.CountMetric(),
                       metrics.MaxMetric(),
                       metrics.MinMetric(),
                       metrics.NoutliersNsigmaMetric(metricName='N(+3Sigma)', nSigma=3),
                       metrics.NoutliersNsigmaMetric(metricName='N(-3Sigma)', nSigma=-3.)]
    return standardSummary


def extendedSummary():
    """An extended set of summary metrics, to calculate all that is in the standard summary stats,
    plus 25/75 percentiles."""

    extendedStats = standardSummary()
    extendedStats += [metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                      metrics.PercentileMetric(metricName='75th%ile', percentile=75)]
    return extendedStats


def standardMetrics(colname, replace_colname=None):
    """A set of standard simple metrics for some quanitity. Typically would be applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    replace_colname: str or None, opt
        Value to replace colname with in the metricName.
        i.e. if replace_colname='' then metric name is Mean, instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is Mean seeingGeom instead of Mean seeingFwhmGeom.
        Default is None, which does not alter the metric name.

    Returns
    -------
    List of configured metrics.
    """
    standardMetrics = [metrics.MeanMetric(colname),
                       metrics.MedianMetric(colname),
                       metrics.MinMetric(colname),
                       metrics.MaxMetric(colname)]
    if replace_colname is not None:
        for m in standardMetrics:
            if len(replace_colname) > 0:
                m.name = m.name.replace('%s' % colname, '%s' % replace_colname)
            else:
                m.name = m.name.rstrip(' %s' % colname)
    return standardMetrics


def extendedMetrics(colname, replace_colname=None):
    """An extended set of simple metrics for some quantity. Typically applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    replace_colname: str or None, opt
        Value to replace colname with in the metricName.
        i.e. if replace_colname='' then metric name is Mean, instead of Mean Airmass, or
        if replace_colname='seeingGeom', then metric name is Mean seeingGeom instead of Mean seeingFwhmGeom.
        Default is None, which does not alter the metric name.

    Returns
    -------
    List of configured metrics.
    """
    extendedMetrics = standardMetrics(colname, replace_colname=None)
    extendedMetrics += [metrics.RmsMetric(colname),
                        metrics.NoutliersNsigmaMetric(colname, metricName='N(+3Sigma) ' + colname, nSigma=3),
                        metrics.NoutliersNsigmaMetric(colname, metricName='N(-3Sigma) ' + colname, nSigma=-3),
                        metrics.PercentileMetric(colname, percentile=25),
                        metrics.PercentileMetric(colname, percentile=75),
                        metrics.CountMetric(colname)]
    if replace_colname is not None:
        for m in extendedMetrics:
            if len(replace_colname) > 0:
                m.name = m.name.replace('%s' % colname, '%s' % replace_colname)
            else:
                m.name = m.name.rstrip(' %s' % colname)
    return extendedMetrics
