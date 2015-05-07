import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from collections import OrderedDict

import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
from lsst.sims.maf.plots import PlotHandler
from .metricBundle import MetricBundle

__all__ = ['makeBundleDict', 'MetricBundleGroup']

def makeBundleDict(bundleList):
    """
    Utility to convert a list of MetricBundles into a dictionary, keyed by the fileRoot names.
    """
    bDict = {b.fileRoot:b for b in bundleList}
    return bDict

class MetricBundleGroup(object):
    """
    Handles group of MetricBundle objects with the same SQL constraint.
    Primary job is to query data from database, and find and run "compatible" subgroups of MetricBundles to
    populate them with data.
    A compatible subgroup of metricbundles has the same SQL constraint, as well as the same slicer, mapsList, and stackerList.
    Thus, they modify the data returned from the query in the same way and iterate over the same slicer to generate metric values.

    Each MetricBundleSet of metric bundles should be a dictionary.
    Each group must query on the same database and have the same SQL constraint.
    The data returned from the db query is stored in the MetricBundleGroup object.
    MetricBundleGroup also provides convenience methods to generate all plots, run all summary statistics,
    run all reduce functions, and write all metricbundles to disk.
    Thus, it also tracks the 'outDir' and 'resultsDb'.
    """
    def __init__(self, bundleDict, dbObj, outDir='.', resultsDb=None, verbose=True, saveEarly=True):
        """
        Set up the MetricBundleGroup, check that all MetricBundles have the same sql constraint.
        """
        # Print occasional messages to screen.
        self.verbose = verbose
        # Save metric results as soon as possible (in case of crash).
        self.saveEarly = saveEarly
        # Check for output directory, create it if needed.
        self.outDir = outDir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        # Do some type checking on the MetricBundle dictionary.
        if not isinstance(bundleDict, dict):
            raise ValueError('BundleDict should be a dictionary containing MetricBundle objects.')
        for b in bundleDict.itervalues():
            if not isinstance(b, MetricBundle):
                raise ValueError('bundleDict should contain only MetricBundle objects.')
        # Check that all metricBundles have the same sql constraint.
        self.sqlconstraint = bundleDict.itervalues().next().sqlconstraint
        for k, b in bundleDict.iteritems():
            if b.sqlconstraint != self.sqlconstraint:
                raise ValueError('MetricBundleGroup must have the same sqlconstraint:',
                                 '%s (in MetricBundle %s) != %s (first constraint)' % (b.sqlconstraint, k, self.sqlconstraint))
        self.bundleDict = bundleDict
        # Check the dbObj.
        if not isinstance(dbObj, db.Database):
            raise ValueError('dbObj should be an instantiated lsst.sims.maf.db.Database (or child) object.')
        self.dbObj = dbObj
        # Check the resultsDb (optional).
        if resultsDb is not None:
            if not isinstance(resultsDb, db.ResultsDb):
                raise ValueError('resultsDb should be an lsst.sims.maf.db.ResultsDb object')
        self.resultsDb = resultsDb

        # Build list of all the columns needed from the database.
        self.dbCols = []
        for b in self.bundleDict.itervalues():
            self.dbCols.extend(b.dbCols)
        self.dbCols = list(set(self.dbCols))
        self.simData = None

        # Find compatible subsets of the MetricBundle dictionary, which can be run/metrics calculated/ together.
        self._findCompatibleLists()

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bk in bundleDict:
            self.hasRun[bk] = False

    def getData(self):
        """
        Query the data from the database.
        """
        # This could be done automatically on init, but it seems that it's nice to let the user
        #  be prepared for this step (as it could be a bit long if much data is needed). This way
        #  they could theoretically also verify which columns could be queries, what the sqlconstraint was, etc.
        # Query the data from the dbObj.
        if self.verbose:
            print "Querying database with constraint %s" % self.sqlconstraint
        # Note that we do NOT run the stackers at this point (this must be done in each 'compatible' group).
        self.simData = utils.getSimData(self.dbObj, self.sqlconstraint, self.dbCols)
        if self.verbose:
            print "Found %i visits" % self.simData.size

        # Query for the fieldData if we need it for the opsimFieldSlicer.
        # Determine if we have a opsimFieldSlicer:
        needFields = False
        for b in self.bundleDict.itervalues():
            if b.slicer.slicerName == 'OpsimFieldSlicer':
                needFields = True
        if needFields:
            self.fieldData = utils.getFieldData(self.dbObj, self.sqlconstraint)
        else:
            self.fieldData = None

    def _getDictSubset(self, origdict, subsetkeys):
        """
        Utility to return a dictionary with a subset of an original dictionary, identified by subsetkeys.
        """
        newdict = {key:origdict.get(key) for key in subsetkeys}
        return newdict

    def _checkCompatible(self, metricBundle1, metricBundle2):
        """
        Check if two MetricBundles are "compatible".
        Compatible indicates that the sql constraints, the slicers, and the maps are the same, and
        that the stackers do not interfere with each other (i.e. are not trying to set the same column in different ways).
        Returns True if the MetricBundles are compatible, False if not.
        """
        if metricBundle1.sqlconstraint != metricBundle2.sqlconstraint:
            return False
        if metricBundle1.slicer != metricBundle2.slicer:
            return False
        if metricBundle1.mapsList.sort() != metricBundle2.mapsList.sort():
            return False
        for stacker in metricBundle1.stackerList:
            for stacker2 in metricBundle2.stackerList:
                # If the stackers have different names, that's OK, and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (stacker != stacker2):
                    return False
        # But if we got this far, everything matches.
        return True

    def _findCompatibleLists(self):
        """
        Find sets of compatible metricBundles from the bundleDict.
        """
        # Making this explicit lets the user see each set of compatible metricBundles --
        # This ought to make it easier to pick up and re-run compatible subsets if there are failures.
        # CompatibleLists stores a list of lists;
        #   each (nested) list contains the bundleDict keys of a compatible set of metricBundles.
        #
        #  .. nevermind (previous comments) - I had a loop control problem that I think is fixed now.
        compatibleLists = []
        for k, b in self.bundleDict.iteritems():
            foundCompatible = False
            for compatibleList in compatibleLists:
                comparisonMetricBundleKey = compatibleList[0]
                compatible = self._checkCompatible(self.bundleDict[comparisonMetricBundleKey], b)
                if compatible:
                    # Must compare all metricBundles in each subset (if they are a potential match),
                    #  as the stackers could be different (and one could be incompatible, not necessarily the first)
                    for comparisonMetricBundleKey in compatibleList[1:]:
                        compatible = self._checkCompatible(self.bundleDict[comparisonMetricBundleKey], b)
                        if not compatible:
                            # If we find one which is not compatible, stop and go on to the next subset list.
                            break
                    # Otherwise, we reached the end of the subset and they were all compatible.
                    foundCompatible=True
                    compatibleList.append(k)
            if not foundCompatible:
                # Didn't find a pre-existing compatible set; make a new one.
                compatibleLists.append([k,])
        self.compatibleLists = compatibleLists

    def runAll(self):
        """
        Run all the metricBundles in the entire metricBundle group.
        Also runs 'reduceAll' and then 'summaryAll'.
        """
        if self.simData is None:
            self.getData()
        for compatibleList in self.compatibleLists:
            if self.verbose:
                print 'Running: ', compatibleList
            self.runCompatible(compatibleList)
            if self.verbose:
                print 'Completed metric generation.'
            for key in compatibleList:
                self.hasRun[key] = True
        if self.verbose:
            print 'Running reduce methods.'
        self.reduceAll()
        if self.verbose:
            print 'Running summary statistics.'
        self.summaryAll()
        if self.verbose:
            print 'Completed.'

    def runCompatible(self, compatibleList):
        """
        Runs a set of 'compatible' metricbundles in the MetricBundleGroup dictionary, identified by 'compatibleList' keys.
        """
        # Grab a dictionary representation of this subset of the dictionary, for easier iteration.
        bDict = self._getDictSubset(self.bundleDict, compatibleList)

        compatMaps = []
        compatStackers = []
        for b in bDict.itervalues():
            for mapsList in b.mapsList:
                compatMaps.extend(mapsList)
            for stacker in b.stackerList:
                if stacker not in compatStackers:
                    compatStackers.append(stacker)

        # May need to do a more rigorous purge of duplicate stackers and maps
        compatMaps = list(set(compatMaps))

        for stacker in compatStackers:
            # Check that stackers can clobber cols that are already there
            self.simData = stacker.run(self.simData)

        # Pull out one of the slicers to use as our 'slicer'.
        # This will be forced back into all of the metricBundles at the end (so that they track
        #  the same metadata such as the slicePoints, in case the same actual object wasn't used).
        slicer = bDict.itervalues().next().slicer
        if slicer.slicerName == 'OpsimFieldSlicer':
            slicer.setupSlicer(self.simData, self.fieldData, maps=compatMaps)
        else:
            slicer.setupSlicer(self.simData, maps=compatMaps)
        # Copy the slicer (after setup) back into the individual metricBundles.
        if slicer.slicerName != 'HealpixSlicer' or slicer.slicerName != 'UniSlicer':
            for b in bDict.itervalues():
                b.slicer = slicer

        # Set up (masked) arrays to store metric data in each metricBundle.
        for b in bDict.itervalues():
            b._setupMetricValues()

        # Set up an ordered dictionary to be the cache if needed:
        # (Currently using OrderedDict, it might be faster to use 2 regular Dicts instead)
        if slicer.cacheSize > 0:
            cacheDict = OrderedDict()
            cache = True
        else:
            cache = False
        # Run through all slicepoints and calculate metrics.
        for i, slice_i in enumerate(slicer):
            slicedata = self.simData[slice_i['idxs']]
            if len(slicedata)==0:
                # No data at this slicepoint. Mask data values.
                for b in bDict.itervalues():
                    b.metricValues.mask[i] = True
            else:
                # There is data! Should we use our data cache?
                if cache:
                    # Make the data idxs hashable.
                    cacheKey = str(sorted(slice_i['idxs']))[1:-1].replace(' ','')
                    # If key exists, set flag to use it, otherwise add it
                    if cacheKey in cacheDict:
                        useCache = True
                    else:
                        cacheDict[cacheKey] = i
                        useCache = False
                    # If we are above the cache size, drop the oldest element from the cache dict
                    if i > slicer.cacheSize:
                        cacheDict.popitem(last=False) #remove 1st item
                    for b in bDict.itervalues():
                        if useCache:
                            b.metricValues.data[i] = b.metricValues.data[cacheDict[cacheKey]]
                        else:
                            b.metricValues.data[i] = b.metric.run(slicedata, slicePoint=slice_i['slicePoint'])
                # Not using memoize, just calculate things normally
                else:
                    for b in bDict.itervalues():
                        b.metricValues.data[i] = b.metric.run(slicedata, slicePoint=slice_i['slicePoint'])

        # Mask data where metrics could not be computed (according to metric bad value).
        for b in bDict.itervalues():
            if b.metricValues.dtype.name == 'object':
                for ind, val in enumerate(b.metricValues.data):
                    if val is b.metric.badval:
                        b.metricValues.mask[ind] = True
            else:
                # For some reason, this doesn't work for dtype=object arrays.
                b.metricValues.mask = np.where(b.metricValues.data==b.metric.badval,
                                               True, b.metricValues.mask)

        # Save data to disk as we go, although this won't keep summary values, etc. (just failsafe).
        if self.saveEarly:
            for b in bDict.itervalues():
                b.write(outDir=self.outDir, resultsDb=self.resultsDb)

    def reduceAll(self, updateSummaries=True):
        """
        Run all reduce functions for the metric in each metricBundle.
        """
        # Create a temporary dictionary to hold the reduced metricbundles.
        reduceBundleDict = {}
        for b in self.bundleDict.itervalues():
            # If there are no reduce functions associated with the metric, skip this metricBundle.
            if len(b.metric.reduceFuncs) > 0:
                # Apply reduce functions, creating a new metricBundle in the process (new metric values).
                for reduceFunc in b.metric.reduceFuncs.itervalues():
                    newmetricbundle = b.reduceMetric(reduceFunc)
                    # Add the new metricBundle to our temporary dictionary.
                    name = newmetricbundle.metric.name
                    if name in self.bundleDict:
                        name = newmetricbundle.fileRoot
                    reduceBundleDict[name] = newmetricbundle
                    if self.saveEarly:
                        newmetricbundle.write(outDir=self.outDir, resultsDb=self.resultsDb)
                # Remove summaryMetrics from top level metricbundle if desired.
                if updateSummaries:
                    b.summaryMetrics = []
        # Add the new metricBundles to the MetricBundleGroup dictionary.
        self.bundleDict.update(reduceBundleDict)

    def summaryAll(self):
        """
        Run summary statistics on all metricbundles.
        """
        for b in self.bundleDict.itervalues():
            b.computeSummaryStats(self.resultsDb)

    def plotAll(self, savefig=True, outfileSuffix=None, figformat='pdf', dpi=600, thumbnail=True,
                closefigs=True):
        """
        Generate the plots for all the metricbundles.
        """
        if self.verbose:
            print 'Plotting.'
        plotHandler = PlotHandler(outDir=self.outDir, resultsDb=self.resultsDb,
                                  savefig=savefig, figformat=figformat, dpi=dpi, thumbnail=thumbnail)
        for b in self.bundleDict.itervalues():
            b.plot(plotHandler=plotHandler, outfileSuffix=outfileSuffix, savefig=savefig)
            if closefigs:
                plt.close('all')
        if self.verbose:
            print 'Plotting complete.'

    def writeAll(self):
        """
        Save all the metricbundles to disk.
        """
        if self.verbose:
            if self.saveEarly:
                print 'Re-saving metric bundles.'
            else:
                print 'Saving metric bundles.'
        for b in self.bundleDict.itervalues():
            b.write(outDir=self.outDir, resultsDb=self.resultsDb)
