import numpy as np
import numpy.ma as ma
from collections import OrderedDict
import os
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
from .benchmark import Benchmark

class BenchmarkGroup(object):
    """
    Handles groups of Benchmark objects with the same SQL constraint.
    Primary job is to query data from database, and find and run "compatible" subgroups of benchmarks to
    populate them with data.
    A "compatible" subgroup of benchmarks has the same SQL constraint, as well as the same slicer, mapsList, and stackerList.
    Thus, they modify the data returned from the query in the same way and iterate over the same slicer to generate metric values.

    Each 'group' of benchmarks should be a dictionary.
    Each group must query on the same database and have the same SQL constraint. The data returned from the db query is stored in
    the object.
    This class also provides convenience methods to generate all plots, run all summary statistics, run all reduce functions,
    and write all benchmarks to disk.
    Thus, it also tracks the 'outDir' and 'resultsDb'.
    """

    def __init__(self, benchmarkDict, dbObj, outDir='.', useResultsDb=False, resultsDbAddress=None, verbose=True):
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
                raise ValueError('BenchmarkGroup must have the same sqlconstraint: %s (in Benchmark %s) != %s (first constraint)'
                                 % (b.sqlconstraint, k, self.sqlconstraint))
        self.benchmarkDict = benchmarkDict
        # Check the dbObj.
        if not isinstance(dbObj, db.Database):
            raise ValueError('dbObj should be an instantiated lsst.sims.maf.db.Database object.')
        self.dbObj = dbObj
        # Set up resultsDb. (optional for use).
        if useResultsDb:
           self.resultsDb = ResultsDb(outDir=self.outDir,
                                      resultsDbAddress=resultsDbAddress)
        else:
            self.resultsDb = False

        # Build list of all the columns needed from the database.
        self.dbCols = []
        for b in self.benchmarkDict.itervalues():
            self.dbCols.extend(b.dbCols)
        self.dbCols = list(set(self.dbCols))

    def getData(self):
        """
        Query the data from the database.
        """
        # This could be done automatically on init, but it seems that it's nice to let the user
        #  be prepared for this step (as it could be a bit long if much data is needed). This way
        #  they could theoretically also verify which columns could be queries, what the sqlconstraint was, etc.
        # Query the data from the dbObj.
        if verbose:
            print "Querying database with constraint %s" % self.sqlconstraint
        # Note that we do NOT run the stackers at this point (this must be done in each 'compatible' group).
        self.simdata = utils.getSimData(dbObj, self.sqlconstraint, self.dbCols)
        if verbose:
            print "Found %i visits" % self.simdata.size

        # Query for the fieldData if we need it for the opsimFieldSlicer.
        # Determine if we have a opsimFieldSlicer:
        needFields = False
        for b in self.benchmarkDict.itervalues():
            if b.slicer.slicerName == 'OpsimFieldSlicer':
                needFields = True
        if needFields:
            self.fieldData = utils.getFieldData(dbObj, sqlconstraint)
        else:
            self.fieldData = None

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bk in benchmarkDict:
            self.hasRun[bk] = False

    def _getDictSubset(self, origdict, subsetkeys):
        newdict = {key:origdict.get(key) for key in somekeys}
        return newdict

    def _checkCompatible(self, benchmark1, benchmark2):
        """
        Check if two benchmarks are "compatible".
        Compatible indicates that the sql constraints, the slicers, and the maps are the same, and
        that the stackers do not interfere with each other (i.e. are not trying to set the same column in different ways).
        Returns True if the benchmarks are compatible, False if not.
        """
        result = False
        if (benchmark1.sqlconstraint == benchmark2.sqlconstraint) and (benchmark1.slicer == benchmark2.slicer):
            if benchmark1.mapsList.sort() == benchmark2.mapsList.sort():
                if len(benchmark1.stackerList) == 0 or len(benchmark2.stackerList) == 0:
                    result = True
                else:
                    for stacker in benchmark1.stackerList:
                        for stacker2 in benchmark2.stackerList:
                            # If the stackers have different names, that's OK, and if they are identical, that's ok.
                            if (stacker.__class__.__name__ != stacker2.__class__.__name__) | (stacker == stacker2):
                                result= True
                            else:
                                result = False
        return result

    def _sortCompatible(self):
        """
        Find sets of compatible benchmarks from the benchmarkDict.
        """
        # Making this explicit lets the user see each set of compatible benchmarks --
        # This ought to make it easier to pick up and re-run compatible subsets if there are failures.
        # CompatibleLists stores a list of lists;
        #   each sublist contains the benchmarkDict keys of a compatible set of benchmarks.
        compatibleLists = []
        for k, b in self.benchmarkDict.iteritems():
            foundCompatible = False
            for compatibleSubSet in compatibleLists:
                compareB = self.benchmarkDict[compatibleSubSet[0]]
                if self._checkCompatible(compareB, b):
                    # Then compareB and b are compatible; add it to this sublist.
                    compatibleSubSet.append(k)
                    foundCompatible = True
            if not foundCompatible:
                # Didn't find a pre-existing compatible set; make a new one.
                compatibleLists.append([k,])
        self.compatibleLists = compatibleLists


    def runAll(self):
        """
        Run all the benchmarks in the entire benchmark group.
        """
        self._sortCompatible()
        for compatibleList in compatibleLists:
            if self.verbose:
                print 'Running: ', compatibleList
            self.runCompatible(compatibleList)
            if self.verbose:
                print 'Completed'
            for key in compatibleList:
                self.hasRun[key] = True

    def runCompatible(self, compatibleList):
        """
        Runs a set of 'compatible' benchmarks in the benchmark group, identified by 'keys'.
        """
        # think about passing in a subset of the dictionary here? (might not be practical, but might make iteration clearer)
        # Or at least changing name from keys?
        # Maybe add a check that they are indeed compatible

        maps = []
        stackers = []
        for bkey in keys:
            for mapsList in self.benchmarkDict[bkey].mapsList:
                maps.extend(mapsList)
            for stacker in self.benchmarkDict[bkey].stackerList:
                if stacker not in stackers:
                    stackers.append(stacker)

        # May need to do a more rigorous purge of duplicate stackers and maps
        maps = list(set(maps))

        for stacker in stackers:
            # Check that stackers can clobber cols that are already there
            self.simdata = stacker.run(self.simdata)

        slicer = self.benchmarkDict[keys[0]].slicer
        if slicer.slicerName == 'OpsimFieldSlicer':
            slicer.setupSlicer(self.simdata, self.fieldData, maps=maps)
        else:
            slicer.setupSlicer(self.simdata, maps=maps)

        # Set up (masked) arrays to store metric data.
        for k in keys:
           self.benchmarkDict[k].metricValues = \
           ma.MaskedArray(data = np.empty(len(slicer), self.benchmarkDict[k].metric.metricDtype),
                          mask = np.zeros(len(slicer), 'bool'),
                          fill_value= slicer.badval)

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
               for k in keys:
                  self.benchmarkDict[k].metricValues.mask[i] = True
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
                  for k in keys:
                     if useCache:
                        self.benchmarkDict[k].metricValues.data[i] = \
                                            self.benchmarkDict[k].metricValues.data[cacheDict[cacheKey]]
                     else:
                        self.benchmarkDict[k].metricValues.data[i] = \
                            self.benchmarkDict[k].metric.run(slicedata, slicePoint=slice_i['slicePoint'])
               # Not using memoize, just calculate things normally
               else:
                  for k in keys:
                     self.benchmarkDict[k].metricValues.data[i] = \
                                self.benchmarkDict[k].metric.run(slicedata, slicePoint=slice_i['slicePoint'])

        # Mask data where metrics could not be computed (according to metric bad value).
        for k in keys:
           if self.benchmarkDict[k].metricValues.dtype.name == 'object':
              for ind, val in enumerate(self.benchmarkDict[k].metricValues.data):
                 if val is self.benchmarkDict[k].metricObjs.badval:
                    self.benchmarkDict[k].metricValues.mask[ind] = True
           else:
              # For some reason, this doesn't work for dtype=object arrays.
              self.benchmarkDict[k].metricValues.mask = \
                np.where(self.benchmarkDict[k].metricValues.data==self.benchmarkDict[k].metric.badval,
                         True, self.benchmarkDict[k].metricValues.mask)

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
