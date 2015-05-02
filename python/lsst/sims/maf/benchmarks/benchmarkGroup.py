import os
import numpy as np
import numpy.ma as ma
from collections import OrderedDict

import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
from .benchmark import Benchmark


class BenchmarkGroup(object):
    """
    Handles groups of Benchmark objects with the same SQL constraint.
    Primary job is to query data from database, and find and run "compatible" subgroups of benchmarks to
    populate them with data.
    A compatible subgroup of benchmarks has the same SQL constraint, as well as the same slicer, mapsList, and stackerList.
    Thus, they modify the data returned from the query in the same way and iterate over the same slicer to generate metric values.

    Each benchmarkGroup of benchmarks should be a dictionary.
    Each group must query on the same database and have the same SQL constraint.
    The data returned from the db query is stored in the benchmarkGroup object.
    BenchmarkGroup also provides convenience methods to generate all plots, run all summary statistics,
    run all reduce functions, and write all benchmarks to disk.
    Thus, it also tracks the 'outDir' and 'resultsDb'.
    """
    def __init__(self, benchmarkDict, dbObj, outDir='.', resultsDb=None, verbose=True):
        """
        Set up the benchmark group, check that all benchmarks have the same sql constraint.
        """
        self.verbose = verbose

        # Check for output directory, create it if needed.
        self.outDir = outDir
        if not os.path.isdir(self.outDir):
            os.makedirs(self.outDir)
        # Do some type checking on the benchmarkDict.
        if not isinstance(benchmarkDict, dict):
            raise ValueError('benchmarkDict should be a dictionary containing benchmark objects.')
        for b in benchmarkDict.itervalues():
            if not isinstance(b, Benchmark):
                raise ValueError('benchmarkDict should contain only benchmark objects.')
        # Check that all benchmarks have the same sql constraint.
        self.sqlconstraint = benchmarkDict.itervalues().next().sqlconstraint
        for k, b in benchmarkDict.iteritems():
            if b.sqlconstraint != self.sqlconstraint:
                raise ValueError('BenchmarkGroup must have the same sqlconstraint: %s (in Benchmark %s) != %s (first constraint)'\
                                 % (b.sqlconstraint, k, self.sqlconstraint))
        self.benchmarkDict = benchmarkDict
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
        for b in self.benchmarkDict.itervalues():
            self.dbCols.extend(b.dbCols)
        self.dbCols = list(set(self.dbCols))
        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bk in benchmarkDict:
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
        self.simdata = utils.getSimData(self.dbObj, self.sqlconstraint, self.dbCols)
        if self.verbose:
            print "Found %i visits" % self.simdata.size

        # Query for the fieldData if we need it for the opsimFieldSlicer.
        # Determine if we have a opsimFieldSlicer:
        needFields = False
        for b in self.benchmarkDict.itervalues():
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

    def _checkCompatible(self, benchmark1, benchmark2):
        """
        Check if two benchmarks are "compatible".
        Compatible indicates that the sql constraints, the slicers, and the maps are the same, and
        that the stackers do not interfere with each other (i.e. are not trying to set the same column in different ways).
        Returns True if the benchmarks are compatible, False if not.
        """
        if benchmark1.sqlconstraint != benchmark2.sqlconstraint:
            return False
        if benchmark1.slicer != benchmark2.slicer:
            return False
        if benchmark1.mapsList.sort() != benchmark2.mapsList.sort():
            return False
        for stacker in benchmark1.stackerList:
            for stacker2 in benchmark2.stackerList:
                # If the stackers have different names, that's OK, and if they are identical, that's ok.
                if (stacker.__class__.__name__ == stacker2.__class__.__name__) & (stacker != stacker2):
                    return False
        # But if we got this far, everything matches.
        return True

    def _findCompatibleLists(self):
        """
        Find sets of compatible benchmarks from the benchmarkDict.
        """
        # Making this explicit lets the user see each set of compatible benchmarks --
        # This ought to make it easier to pick up and re-run compatible subsets if there are failures.
        # CompatibleLists stores a list of lists;
        #   each (nested) list contains the benchmarkDict keys of a compatible set of benchmarks.
        #
        #  .. nevermind (previous comments) - I had a loop control problem that I think is fixed now.
        compatibleLists = []
        for k, b in self.benchmarkDict.iteritems():
            foundCompatible = False
            for compatibleList in compatibleLists:
                comparisonBenchmarkKey = compatibleList[0]
                compatible = self._checkCompatible(self.benchmarkDict[comparisonBenchmarkKey], b)
                if compatible:
                    # Must compare all benchmarks in each subset (if they are a potential match),
                    #  as the stackers could be different (and one could be incompatible, not necessarily the first)
                    for comparisonBenchmarkKey in compatibleList[1:]:
                        compatible = self._checkCompatible(self.benchmarkDict[comparisonBenchmarkKey], b)
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
        Run all the benchmarks in the entire benchmark group.
        """
        self._findCompatibleLists()
        for compatibleList in self.compatibleLists:
            if self.verbose:
                print 'Running: ', compatibleList
            self.runCompatible(compatibleList)
            if self.verbose:
                print 'Completed'
            for key in compatibleList:
                self.hasRun[key] = True

    def runCompatible(self, compatibleList):
        """
        Runs a set of 'compatible' benchmarks in the benchmarkGroup dictionary, identified by 'compatibleList' keys.
        """
        # Grab a dictionary representation of this subset of the dictionary, for easier iteration.
        bDict = self._getDictSubset(self.benchmarkDict, compatibleList)

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
            self.simdata = stacker.run(self.simdata)

        # Pull out one of the slicers to use as our 'slicer'.
        # This will be forced back into all of the benchmarks at the end (so that they track
        #  the same metadata such as the slicePoints, in case the same actual object wasn't used).
        #  ?? (or maybe we just copy the metadata into the other slicers, if they aren't the same object?)
        slicer = bDict.itervalues().next().slicer
        if slicer.slicerName == 'OpsimFieldSlicer':
            slicer.setupSlicer(self.simdata, self.fieldData, maps=compatMaps)
        else:
            slicer.setupSlicer(self.simdata, maps=compatMaps)

        # Set up (masked) arrays to store metric data in each benchmark.
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
            slicedata = self.simdata[slice_i['idxs']]
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

    def reduceAll(self):
        """
        Run all reduce functions for the metric in each benchmark.
        """
        pass

    def plotAll(self):
        """
        Generate the plots for all the benchmarks.
        """
        for b in self.benchmarkDict.itervalues():
            b.plot(outDir=self.outDir, resultsDb=self.resultsDb)

    def writeAll(self):
        """
        Save all the benchmarks to disk.
        """
        for b in self.benchmarkDict.itervalues():
            b.writeBenchmark(outDir=self.outDir, resultsDb=self.resultsDb)
