from __future__ import print_function

import lsst.sims.maf.metrics as metrics

__all__ = ['sqlWheres', 'standardSummaryMetrics', 'extendedSummaryMetrics']


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

def standardSummaryMetrics():
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


def extendedSummaryMetrics():
    """An extended set of summary metrics, to calculate all that is in the standard summary stats, plus 
     25/75 percentiles."""

    extendedStats = standardSummaryMetrics()
    extendedStats += [metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                      metrics.PercentileMetric(metricName='75th%ile', percentile=75)]
    return extendedStats

