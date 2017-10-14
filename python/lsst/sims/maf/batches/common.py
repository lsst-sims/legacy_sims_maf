from __future__ import print_function

import lsst.sims.maf.metrics as metrics

__all__ = ['standardSummary', 'extendedSummary', 'standardMetrics', 'extendedMetrics',
           'summaryCompletenessAtTime','summaryCompletenessOverH']


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

def summaryCompletenessAtTime(times, Hval, Hindex=0.33):
    """A simple list of summary metrics to be applied to the Discovery_Time or PreviouslyKnown metrics.
    (can be used with any moving object metric which returns the time of discovery).

    Parameters
    ----------
    times : np.ndarray or list
        The times at which to evaluate the completeness @ Hval.
    Hval : float
        The H value at which to evaluate the completeness (cumulative and differential).
    Hindex : float, opt
        The index of the power law to integrate H over (for cumulative completeness).
        Default is 0.33.

    Returns
    -------
    List of moving object MoCompletenessAtTime metrics (cumulative and differential)
    """
    summaryMetrics = [metrics.MoCompletenessAtTimeMetric(times=times, Hval=Hval, Hindex=Hindex,
                                                         cumulative=False),
                      metrics.MoCompletenessAtTimeMetric(times=times, Hval=Hval, Hindex=Hindex,
                                                         cumulative=True)]
    return summaryMetrics

def summaryCompletenessOverH(requiredChances=1, Hindex=0.33):
    """A simple list of summary metrics to be applied to the Discovery_N_Chances metric.

    Parameters
    ----------
    requiredChances : int, opt
        Number of discovery opportunities required to consider an object 'discovered'.
    Hindex : float, opt
        The index of the power law to integrate H over (for cumulative completeness).
        Default is 0.33.

    Returns
    -------
    List of moving object MoCompleteness metrics (cumulative and differential)
    """
    summaryMetrics = [metrics.MoCompletenessMetric(requiredChances=requiredChances, cumulative=False, Hindex=0.33),
                      metrics.MoCompletenessMetric(requiredChances=requiredChances, cumulative=True, Hindex=0.33)]
    return summaryMetrics
