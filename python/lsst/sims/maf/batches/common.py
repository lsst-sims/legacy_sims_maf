from __future__ import print_function

import lsst.sims.maf.metrics as metrics

__all__ = ['sqlWheres', 'standardSummary', 'extendedSummary', 'standardMetrics', 'extendedMetrics']


def sqlWheres(dbobj):
    """Return the WFD and DD sql statements.

    Parameters
    -----------
    dbobj : ~lsst.sims.maf.db.OpsimDatabase
        A MAF OpsimDatabase object.

    Returns
    -------
    dict
        Dictionary keyed "WFD" and "DD" with values of the sqlconstraints for WFD and DD proposals.
    """
    sqlWhere = {}
    propids, proptags = dbobj.fetchPropInf()
    sqlWhere['WFD'] = utils.createSQLWhere('WFD', proptags)
    print('# WFD "where" clause: %s' % (sqlWhere['WFD']))
    sqlWhere['DD'] = utils.createSQLWhere('DD', proptags)
    print('# DD "where" clause: %s' % (sqlWhere['DD']))
    return sqlWhere

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

    extendedStats = standardSummaryMetrics()
    extendedStats += [metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                      metrics.PercentileMetric(metricName='75th%ile', percentile=75)]
    return extendedStats


def standardMetrics(colname, strip_colname=False):
    """A set of standard simple metrics for some quanitity. Typically would be applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    strip_colname: bool, opt
        Flag to strip colname from the metricName. (i.e. "Mean" instead of "Mean Airmass").

    Returns
    -------
    List of configured metrics.
    """
    standardMetrics = [metrics.MeanMetric(colname),
                       metrics.MedianMetric(colname),
                       metrics.MinMetric(colname),
                       metrics.MaxMetric(colname)]
    if strip_colname:
        for m in standardMetrics:
            m.name = m.name.rstrip(' %s' % colname)
    return standardMetrics

def extendedMetrics(colname, strip_colname=False):
    """An extended set of simple metrics for some quantity. Typically applied with unislicer.

    Parameters
    ----------
    colname : str
        The column name to apply the metrics to.
    strip_colname: bool, opt
        Flag to strip colname from the metricName. (i.e. "Mean" instead of "Mean Airmass").

    Returns
    -------
    List of configured metrics.
    """
    extendedMetrics = standardMetrics(colname)
    extendedMetrics += [metrics.RmsMetric(colname),
                        metrics.NoutliersNsigmaMetric(metricName='N(+3Sigma)' + colname, nSigma=3),
                        metrics.NoutliersNsigmaMetric(metricName='N(-3Sigma)' + colname, nSigma=-3),
                        metrics.PercentileMetric(metricName='25th%ile' + colname, percentile=25),
                        metrics.PercentileMetric(metricName='75th%ile' + colname, percentile=75)]
    if strip_colname:
        for m in extendedMetrics:
            m.name = m.name.rstrip(' %s' % colname)
    return extendedMetrics
