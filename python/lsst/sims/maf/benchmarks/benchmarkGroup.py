import lsst.sims.maf.utils as utils

class BenchmarkGroup(object):
    """
    take a dictionary of Benchmark objects, make sure they have the same SQL constaint, then pull the data
    """

    def __init__(benchmarkDict, dbObj, verbose=True):

        # Check that all benchmarks have the same sql
        sql1 = benchmarkDict[benchmarkDict.keys()[0]].sqlconstriant
        for bm in benchmarkDict:
            if benchmarkDict[bm].sqlconstaint != sql1:
                raise ValueError('Benchmarks must have same sqlconstraint %s != %s' % (sql1, benchmarkDict[bm].sqlconstaint))

        self.benchmarkDict = benchmarkDict
        self.dbObj = dbObj
        # Build list of all the columns needed:
        dbCols = []
        for bm in self.benchmarkDict:
            dbCols.extend(bm.dbCols)
        dbCols = list(set(dbCols))

        # Pull the data
        if verbose:
            print "Calling DB with constriant %s"%sql1
        self.simdata = utils.getSimData(dbObj, sql1, dbcols)
        if verbose:
            print "Found %i visits"%self.simdata.size

        # If we need fieldData, grab it here:
        if benchmarkDict[bm].slicer.slicerName == 'OpsimFieldSlicer':
            self.getFieldData(benchmarkDict[bm].slicer, sql1)
        else self.fieldData = None

        # Dict to keep track of what's been run:
        self.hasRun = {}
        for bm in benchmarkDict:
            self.hasRun[bm] = False
        self.bmKeys = benchmarkDict.keys()

    def _checkCompatable(bm1,bm2):
        """
        figure out which benchmarks are compatable.
        If the sql constaints the same, slicers the same, and stackers have different names, or are equal.
        returns True if the benchmarks could be run together, False if not.
        """
        result = False
        if (bm1.sqlconstraint == bm2.sqlconstraint) & (bm1.slicer == bm2.slicer:):
            for stacker in bm1.stackers:
                for stacker2 in bm2.stackers:
                    if (stacker.__class__.__name__ != staker2.__class__.__name__) | (stacker == stacker2):
                        result= True
        return result


    def runAll():
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
                            if self.checkCompatible(self.benchmarkDict[bm], self.benchmarkDict[key]):
                                toRun.append(bm)

            self._runCompatable(toRun)



    def _runCompatable(keys):
        """
        given a batch of compatable benchmarks, run them.
        These are the keys to the
        """
        # Maybe add a check that they are indeede compatible

        for bm in keys:
            for stacker in self.benchmarkDict[bm].stackerList:
                # Check that stackers can clobber cols that are already there
                self.simData = stacker.run(simData)

    def plotAll():
        """
        Generate the plots for all the benchmarks.
        """
        for bm in self.benchmarkDict:
            self.benchmarkDict[bm].plot()

    def writeAll():
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
            fieldID, idx = np.unique(self.simData[slicer.simDataFieldIDColName], return_index=True)
            ra = self.data[slicer.fieldRaColName][idx]
            dec = self.data[slicer.fieldDecColName][idx]
            self.fieldData = np.core.records.fromarrays([fieldID, ra, dec],
                                               names=['fieldID', 'fieldRA', 'fieldDec'])
