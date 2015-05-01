import lsst.sims.maf.utils as utils
import numpy as np
import numpy.ma as ma
from collections import OrderedDict

class BenchmarkGroup(object):
    """
    take a dictionary of Benchmark objects, make sure they have the same SQL constraint, then pull the data
    """

    def __init__(self,benchmarkDict, dbObj, verbose=True):
        self.verbose = verbose
        # Check that all benchmarks have the same sql
        sql1 = benchmarkDict[benchmarkDict.keys()[0]].sqlconstraint
        for bm in benchmarkDict:
            if benchmarkDict[bm].sqlconstraint != sql1:
                raise ValueError('Benchmarks must have same sqlconstraint %s != %s' % (sql1, benchmarkDict[bm].sqlconstraint))

        self.benchmarkDict = benchmarkDict
        self.dbObj = dbObj
        # Build list of all the columns needed:
        dbCols = []
        for bm in self.benchmarkDict:
            dbCols.extend(self.benchmarkDict[bm].dbCols)
        dbCols = list(set(dbCols))

        # Pull the data
        if verbose:
            print "Calling DB with constraint %s"%sql1
        self.simdata = utils.getSimData(dbObj, sql1, dbCols)
        if verbose:
            print "Found %i visits"%self.simdata.size

        # If we need fieldData, grab it here:
        if benchmarkDict[bm].slicer.slicerName == 'OpsimFieldSlicer':
            self.getFieldData(benchmarkDict[bm].slicer, sql1)
        else:
            self.fieldData = None

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bm in benchmarkDict:
            self.hasRun[bm] = False
        self.bmKeys = benchmarkDict.keys()

    def _checkCompatible(self,bm1,bm2):
        """
        figure out which benchmarks are compatable.
        If the sql constraints the same, slicers the same, and stackers have different names, or are equal.
        returns True if the benchmarks could be run together, False if not.
        """
        result = False
        if (bm1.sqlconstraint == bm2.sqlconstraint) & (bm1.slicer == bm2.slicer):
            if bm1.mapsList.sort() == bm2.mapsList.sort():
                for stacker in bm1.stackerList:
                    for stacker2 in bm2.stackerList:
                        # If the stackers have different names, that's OK, and if they are identical, that's ok.
                        if (stacker.__class__.__name__ != stacker2.__class__.__name__) | (stacker == stacker2):
                            result= True
            return result


    def runAll(self):
        """
        run all the benchmarks
        """
        while False in self.hasRun.values():
            toRun = []

            for bm in self.benchmarkDict:
                if self.hasRun[bm] is False:
                    if len(toRun) == 0:
                        toRun.append(bm)
                    else:
                        for key in toRun:
                            if key != bm:
                                if self._checkCompatible(self.benchmarkDict[bm], self.benchmarkDict[key]):
                                    toRun.append(bm)

            if self.verbose:
                print 'Running:'
                for key in toRun:
                    print key
            self._runCompatable(toRun)
            if self.verbose:
                print 'Completed'
            for key in toRun:
                self.hasRun[key] = True



    def _runCompatable(self,keys):
        """
        given a batch of compatable benchmarks, run them.
        These are the keys to the
        """
        # Maybe add a check that they are indeede compatible

        maps = []
        stackers = []
        for bm in keys:
            for mapsList in self.benchmarkDict[bm].mapsList:
                maps.extend(mapsList)
            for stacker in self.benchmarkDict[bm].stackerList:
                if stacker not in stackers:
                    stackers.append(stacker)

        # May need to do a more rigerous purge of duplicate stackers and maps
        maps = list(set(maps))

        for stacker in stackers:
            # Check that stackers can clobber cols that are already there
            self.simdata = stacker.run(self.simdata)

        slicer = self.benchmarkDict(keys[0]).slicer
        if slicer.slicerName == 'OpsimFieldSlicer':
            slicer.setupSlicer(self.simdata, self.fieldData, maps=maps)
        else:
            slicer.setupSlicer(self.simdata, maps=maps)

        # Set up (masked) arrays to store metric data.
        for key in keys:
           self.benchmarckDict[key].metricValues = \
           ma.MaskedArray(data = np.empty(len(self.slicer),self.benchmarkDict[key].metric.metricDtype),
                          mask = np.zeros(len(self.slicer), 'bool'),
                          fill_value=self.slicer.badval)

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
               for key in keys:
                  self.metricDict[key].metricValues.mask[i] = True
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
                     if i > self.slicer.cacheSize:
                        cacheDict.popitem(last=False) #remove 1st item
                  for key in keys:
                     if useCache:
                        self.benchmarkDict[key].metricValues.data[i] = \
                                            self.benchmarkDict[key].metricValues.data[cacheDict[cacheKey]]
                     else:
                        self.benchmarkDict[key].metricValues.data[i] = \
                            self.benchmarkDict[key].metric.run(slicedata, slicePoint=slice_i['slicePoint'])
               # Not using memoize, just calculate things normally
               else:
                  for key in keys:
                     self.benchmarkDict[key].metricValues.data[i] = \
                                self.benchmarkDict[key].metric.run(slicedata,slicePoint=slice_i['slicePoint'])

        # Mask data where metrics could not be computed (according to metric bad value).
        for key in keys:
           if self.benchmarkDict[key].metricValues.dtype.name == 'object':
              for ind,val in enumerate(self.benchmarkDict[key].metricValues.data):
                 if val is self.benchmarkDict[key].metricObjs.badval:
                    self.benchmarkDict[key].metricValues.mask[ind] = True
           else:
              # For some reason, this doesn't work for dtype=object arrays.
              self.benchmarkDict[key].metricValues.mask = \
                np.where(self.benchmarkDict[key].metricValues.data==self.benchmarkDict[key].metric.badval,
                         True, self.benchmarkDict[key].metricValues.mask)

    def plotAll(self):
        """
        Generate the plots for all the benchmarks.
        """
        for bm in self.benchmarkDict:
            self.benchmarkDict[bm].plot()

    def writeAll(self):
        """
        Save all the benchmarks
        """
        for bm in self.benchmarkDict:
            self.benchmarkDict[bm].writeBenchmark()

    def getFieldData(self, slicer, sqlconstraint):
        """Given an opsim slicer, generate the FieldData """
        # Do a bunch of parsing to get the propids out of the sqlconstraint.
        if 'propID' not in sqlconstraint:
            propids = self.propids.keys()
        else:
            # example sqlconstraint: filter = r and (propid = 219 or propid = 155) and propid!= 90
            sqlconstraint = sqlconstraint.replace('=', ' = ').replace('(', '').replace(')', '')
            sqlconstraint = sqlconstraint.replace("'", '').replace('"', '')
            # Allow for choosing all but a particular proposal.
            sqlconstraint = sqlconstraint.replace('! =' , ' !=')
            sqlconstraint = sqlconstraint.replace('  ', ' ')
            sqllist = sqlconstraint.split(' ')
            propids = []
            nonpropids = []
            i = 0
            while i < len(sqllist):
                if sqllist[i].lower() == 'propid':
                    i += 1
                    if sqllist[i] == "=":
                        i += 1
                        propids.append(int(sqllist[i]))
                    elif sqllist[i] == '!=':
                        i += 1
                        nonpropids.append(int(sqllist[i]))
                i += 1
            if len(propids) == 0:
                propids = self.propids.keys()
            if len(nonpropids) > 0:
                for nonpropid in nonpropids:
                    if nonpropid in propids:
                        propids.remove(nonpropid)
        # And query the field Table.
        if 'Field' in self.dbObj.tables:
            self.fieldData = self.dbObj.fetchFieldsFromFieldTable(propids)
        else:
            fieldID, idx = np.unique(self.simdata[slicer.simDataFieldIDColName], return_index=True)
            ra = self.data[slicer.fieldRaColName][idx]
            dec = self.data[slicer.fieldDecColName][idx]
            self.fieldData = np.core.records.fromarrays([fieldID, ra, dec],
                                               names=['fieldID', 'fieldRA', 'fieldDec'])
