"""Evaluate the open shutter fraction.
"""
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as mb
from .colMapDict import ColMapDict
from .common import standardSummary

__all__ = ['openshutterFractions']


def openshutterFractions(colmap=None, runName='opsim', extraSql=None, extraMetadata=None):
    """Evaluate open shutter fraction over whole survey and per night.

    Parameters
    ----------
    colmap : dict, opt
        A dictionary with a mapping of column names. Default will use OpsimV4 column names.
    runName : str, opt
        The name of the simulated survey. Default is "opsim".
    extraSql : str, opt
        Additional constraint to add to any sql constraints (e.g. 'night<365')
        Default None, for no additional constraints.
    extraMetadata : str, opt
        Additional metadata to add before any below (i.e. "WFD").  Default is None.
    """
    if colmap is None:
        colmap = ColMapDict('opsimV4')
    bundleList = []

    group = 'Open Shutter Fraction'

    subgroup = 'All visits'
    if extraMetadata is not None:
        subgroup = extraMetadata

    # Open Shutter fraction over whole survey.
    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}
    displayDict['caption'] = 'Total open shutter fraction over whole survey. ' \
                             'Does not include downtime due to weather.'
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    slicer = slicers.UniSlicer()
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             displayDict=displayDict)
    bundleList.append(bundle)

    # Open shutter fraction per night.
    subgroup = 'Per night'
    if extraMetadata is not None:
        subgroup = extraMetadata + ' ' + subgroup.lower()
    displayDict = {'group': group, 'subgroup': subgroup, 'order': 0}
    displayDict['caption'] = 'Open shutter fraction per night.'
    displayDict['caption'] += ' This compares on-sky image time against on-sky time + slews + filter ' \
                              'changes + readout, but does not include downtime due to weather.'
    metric = metrics.OpenShutterFractionMetric(slewTimeCol=colmap['slewtime'],
                                               expTimeCol=colmap['exptime'],
                                               visitTimeCol=colmap['visittime'])
    slicer = slicers.OneDSlicer(sliceColName=colmap['night'], binsize=1)
    bundle = mb.MetricBundle(metric, slicer, extraSql, metadata=subgroup,
                             summaryMetrics=standardSummary(), displayDict=displayDict)
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    return mb.makeBundlesDictFromList(bundleList)