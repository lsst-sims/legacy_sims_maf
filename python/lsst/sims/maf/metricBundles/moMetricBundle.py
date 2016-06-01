import os
from copy import deepcopy
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from lsst.sims.maf.slicers import MoObjSlicer
from lsst.sims.maf.metrics import BaseMoMetric
from lsst.sims.maf.slicers import MoObjSlicer
from lsst.sims.maf.stackers import AllMoStackers
import lsst.sims.maf.utils as utils
from lsst.sims.maf.plots import PlotHandler
from lsst.sims.maf.plots import BasePlotter
from lsst.sims.maf.plots import MetricVsH

from .metricBundle import MetricBundle

__all__ = ['MoMetricBundle', 'MoMetricBundleGroup', 'createEmptyMoMetricBundle']


def createEmptyMoMetricBundle():
    """Create an empty metric bundle.

    Returns
    -------
    MoMetricBundle
        An empty metric bundle, configured with just the :class:`BaseMetric` and :class:`BaseSlicer`.
    """
    return MoMetricBundle(BaseMoMetric(), MoObjSlicer(), None)


class MoMetricBundle(MetricBundle):
    def __init__(self, metric, slicer, constraint=None,
                 runName='opsim', metadata=None,
                 fileRoot=None,
                 plotDict=None, plotFuncs=None,
                 displayDict=None,
                 childMetrics=None,
                 summaryMetrics=None):
        """
        Instantiate moving object metric bundle, save metric/slicer/constraint, etc.
        """
        self.metric = metric
        self.slicer = slicer
        if constraint == '':
            constraint = None
        self.constraint = constraint
        # Add the summary stats, if applicable.
        self.setSummaryMetrics(summaryMetrics)
        # Set the provenance/metadata.
        self.runName = runName
        self._buildMetadata(metadata)
        # Build the output filename root if not provided.
        if fileRoot is not None:
            self.fileRoot = fileRoot
        else:
            self._buildFileRoot()
        # Set the plotting classes/functions.
        self.setPlotFuncs(plotFuncs)
        # Set the plotDict and displayDicts.
        self.plotDict = {'units': '@H'}
        self.setPlotDict(plotDict)
        # Update/set displayDict.
        self.displayDict = {}
        self.setDisplayDict(displayDict)
        # Set the list of child metrics.
        self.setChildBundles(childMetrics)
        # This is where we store the metric values and summary stats.
        self.metricValues = None
        self.summaryValues = None

    def _resetMetricBundle(self):
        """Reset all properties of MetricBundle.
        """
        self.metric = None
        self.slicer = None
        self.constraint = None
        self.summaryMetrics = []
        self.plotFuncs = []
        self.runName = 'opsim'
        self.metadata = ''
        self.dbCols = None
        self.fileRoot = None
        self.plotDict = {}
        self.displayDict = {}
        self.childMetrics = None
        self.metricValues = None
        self.summaryValues = None

    def _buildMetadata(self, metadata):
        """If no metadata is provided, auto-generate it from the obsFile + constraint.
        """
        if metadata is None:
            try:
                self.metadata = self.slicer.obsfile.replace('.txt', '').replace('.dat', '')
                self.metadata = self.metadata.replace('_obs', '').replace('_allObs', '')
            except AttributeError:
                self.metadata = 'noObs'
            # And modify by constraint.
            if self.constraint is not None:
                self.metadata += ' ' + self.constraint
        else:
            self.metadata = metadata

    def _findReqCols(self):
        # Doesn't quite work the same way yet. No stacker list, for example.
        raise NotImplementedError

    def setChildBundles(self, childMetrics=None):
        """
        Identify any child metrics to be run on this (parent) bundle.
        and create the new metric bundles that will hold the child values, linking to this bundle.
        Remove the summaryMetrics from self afterwards.
        """
        self.childBundles = {}
        if childMetrics is None:
            childMetrics = self.metric.childMetrics
        for cName, cMetric in childMetrics.iteritems():
            cBundle = MoMetricBundle(metric=cMetric, slicer=self.slicer,
                                     constraint=self.constraint,
                                     runName=self.runName, metadata=self.metadata,
                                     plotDict=self.plotDict, plotFuncs=self.plotFuncs,
                                     summaryMetrics=self.summaryMetrics)
            self.childBundles[cName] = cBundle
        if len(childMetrics) > 0:
            self.summaryMetrics = []

    def computeSummaryStats(self, resultsDb=None):
        """
        Compute summary statistics on metricValues, using summaryMetrics, for self and child bundles.
        """
        if self.summaryValues is None:
            self.summaryValues = {}
        if self.summaryMetrics is not None:
            # Build array of metric values, to use for (most) summary statistics.
            for m in self.summaryMetrics:
                summaryName = m.name
                summaryVal = m.run(self.metricValues, self.slicer.slicePoints['H'])
                self.summaryValues[summaryName] = summaryVal
                # Add summary metric info to results database, if applicable.
                if resultsDb:
                    metricId = resultsDb.updateMetric(self.metric.name, self.slicer.slicerName,
                                                      self.runName, self.constraint, self.metadata, None)
                    resultsDb.updateSummaryStat(metricId, summaryName=summaryName, summaryValue=summaryVal)

    def reduceMetric(self, reduceFunc, reducePlotDict=None, reduceDisplayDict=None):
        raise NotImplementedError

    def read(self, filename):
        "Read metric data back into a metricBundle, as best as possible."
        if not os.path.isfile(filename):
            raise IOError('%s not found' % filename)
        self._resetMetricBundle()
        # Must read the data using a moving object slicer.
        slicer = MoObjSlicer()
        self.metricValues, self.slicer = slicer.readData(filename)
        # It's difficult to reinstantiate the metric object, as we don't
        # know what it is necessarily -- the metricName can be changed.
        self.metric = BaseMoMetric()
        # But, for plot label building, we do need to try to recreate the
        #  metric name and units. We can't really do that yet - need more infrastructure.
        self.metric.name = ''
        self.constraint = None
        self.metadata = None
        path, head = os.path.split(filename)
        self.fileRoot = head.replace('.h5', '')
        self.setPlotFuncs([MetricVsH()])


class MoMetricBundleGroup(object):
    def __init__(self, bundleDict, outDir='.', resultsDb=None, verbose=True,
                 saveEarly=False):
        # Not really handling resultsDb yet.
        self.verbose = verbose
        self.bundleDict = bundleDict
        self.outDir = outDir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        self.resultsDb = resultsDb

        self.slicer = self.bundleDict.itervalues().next().slicer
        for b in self.bundleDict.itervalues():
            if b.slicer != self.slicer:
                raise ValueError('Currently, the slicers for the MoMetricBundleGroup must be equal - using the same observations and Hvals.')
        self.constraints = list(set([b.constraint for b in bundleDict.values()]))

        self.saveEarly = saveEarly

    def _setCurrent(self, constraint):
        """Private utility to set the currentBundleDict (i.e. set of metricBundles with the same constraint).
        """
        self.currentBundleDict = {}
        for k, b in self.bundleDict.iteritems():
            if b.constraint == constraint:
                self.currentBundleDict[k] = b

    def runCurrent(self, constraint):
        """Calculate the metric values for set of (parent and child) bundles,
        using the same constraint and slicer.
        """
        # Identify the observations which are relevant for this constraint.
        self.slicer.subsetObs(constraint)
        # Set up all the stackers (this assumes we run all of the stackers all of the time).
        allStackers = AllMoStackers()
        # Set up all of the metric values, including for the child bundles.
        for b in self.currentBundleDict.itervalues():
            b._setupMetricValues()
            for cb in b.childBundles.itervalues():
                cb._setupMetricValues()
        # Calculate the metric values.
        for i, slicePoint in enumerate(self.slicer):
            ssoObs = slicePoint['obs']
            for j, Hval in enumerate(slicePoint['Hvals']):
                # Run stackers to add extra columns (that depend on H)
                ssoObs = allStackers.run(ssoObs, slicePoint['orbit']['H'], Hval)
                # Run all the parent metrics.
                for b in self.currentBundleDict.itervalues():
                    # Mask the parent metric (and then child metrics) if there was no data.
                    if len(ssoObs) == 0:
                        b.metricValues.mask[i][j] = True
                        for cb in b.childBundles.itervalues():
                            cb.metricValues.mask[i][j] = True
                    # Otherwise, calculate the metric value for the parent, and then child.
                    else:
                        # Calculate for the parent.
                        mVal = b.metric.run(ssoObs, slicePoint['orbit'], Hval)
                        # Mask if the parent metric returned a bad value.
                        if mVal == b.metric.badval:
                            b.metricValues.mask[i][j] = True
                            for cb in b.childBundles.itervalues():
                                cb.metricValues.mask[i][j] = True
                        # Otherwise, set the parent value and calculate the child metric values as well.
                        else:
                            b.metricValues.data[i][j] = mVal
                            for cb in b.childBundles.itervalues():
                                childVal = cb.metric.runChild(ssoObs, slicePoint['orbit'], Hval, mVal)
                                if childVal == cb.metric.badval:
                                    cb.metricValues.mask[i][j] = True
                                else:
                                    cb.metricValues.data[i][j] = childVal

    def runAll(self):
        """
        Run all constraints and metrics for these moMetricBundles.
        """
        for constraint in self.constraints:
            self._setCurrent(constraint)
            self.runCurrent(constraint)
        if self.verbose:
            print 'Calculated all metrics.'

    def plotCurrent(self, savefig=True, outfileSuffix=None, figformat='pdf', dpi=600, thumbnail=True,
                closefigs=True):
        plotHandler = PlotHandler(outDir=self.outDir, resultsDb=self.resultsDb,
                                  savefig=savefig, figformat=figformat, dpi=dpi, thumbnail=thumbnail)
        for b in self.currentBundleDict.itervalues():
            b.plot(plotHandler=plotHandler, outfileSuffix=outfileSuffix, savefig=savefig)
            for cb in b.childBundles.itervalues():
                cb.plot(plotHandler=plotHandler, outfileSuffix=outfileSuffix, savefig=savefig)
            if closefigs:
                plt.close('all')
        if self.verbose:
            print 'Plotting complete.'

    def plotAll(self, savefig=True, outfileSuffix=None, figformat='pdf', dpi=600, thumbnail=True,
                closefigs=True):
        """
        Make a few generically desired plots. This needs more flexibility in the future.
        """
        for constraint in self.constraints:
            self._setCurrent(constraint)
            self.plotCurrent(savefig=savefig, outfileSuffix=outfileSuffix, figformat=figformat, dpi=dpi,
                             thumbnail=thumbnail, closefigs=closefigs)
        if self.verbose:
            print 'Plotted all metrics.'

    def summaryCurrent(self):
        """
        Run summary statistics on all the metricBundles in currentBundleDict.
        """
        for b in self.currentBundleDict.itervalues():
            b.computeSummaryStats(self.resultsDb)
            for cB in b.childBundles.itervalues():
                cB.computeSummaryStats(self.resultsDb)

    def summaryAll(self):
        """
        Run the summary statistics for all metrics in bundleDict.
        This assumes that 'clearMemory' was false.
        """
        for constraint in self.constraints:
            self._setCurrent(constraint)
            self.summaryCurrent()

    def writeAll(self):
        """Save all the MetricBundles to disk.

        Saving all MetricBundles to disk at this point assumes that clearMemory was False.
        """
        for constraint in self.constraints:
            self._setCurrent(constraint)
            self.writeCurrent()

    def writeCurrent(self):
        """Save all the MetricBundles in the currently active set to disk.
        """
        if self.verbose:
            if self.saveEarly:
                print 'Re-saving metric bundles.'
            else:
                print 'Saving metric bundles.'
        for b in self.currentBundleDict.itervalues():
            b.write(outDir=self.outDir, resultsDb=self.resultsDb)
            for cB in b.childBundles.itervalues():
                cB.write(outDir=self.outDir, resultsDb=self.resultsDb)