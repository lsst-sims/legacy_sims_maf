import os
from collections import OrderedDict
import numpy as np
import lsst.sims.maf.db as db
from .mafRunResults import MafRunResults

__all__ = ['MafTracking']

class MafTracking(object):
    """
    Class to read MAF's tracking database (tracking all MAF runs) and handle the output for web display.

    Deals with a single MAF run (one output directory, one resultsDb) only. """
    def __init__(self, database=None):
        """
        Instantiate the (multi-run) layout visualization class.
        """
        if database is None:
            database = os.path.join(os.getcwd(), 'trackingDb_sqlite.db')

        # Read in the results database.
        database = db.Database(database=database, longstrings=True,
                               dbTables={'runs':['runs', 'mafRunId']})
        self.runs = database.queryDatabase('runs', 'select * from runs')
        self.runs = self.sortRuns(self.runs)
        self.runsPage = {}

    def runInfo(self, run):
        """
        Return ordered dict of run information, for a given run.
        """
        runInfo = OrderedDict()
        runInfo['OpsimRun'] = run['opsimRun']
        runInfo['MafComment'] = run['mafComment']
        runInfo['OpsimComment'] = run['opsimComment']
        runInfo['MafDir'] = run['mafDir']
        runInfo['OpsimDate'] = run['opsimDate']
        runInfo['MafDate'] = run['mafDate']
        return runInfo

    def sortRuns(self, runs, order=['opsimRun', 'mafComment', 'mafRunId']):
        return np.sort(runs, order=order)

    def getRun(self, mafRunId):
        """
        For a chosen runID, instantiate a mafRunResults object to read and handle the
        individual run results.
        Store this information internally.
        """
        if not isinstance(mafRunId, int):
            if isinstance(mafRunId, dict):
                mafRunId = int(mafRunId['runId'][0][0])
            if isinstance(mafRunId, list):
                mafRunId = int(mafRunId[0])
        if mafRunId in self.runsPage:
            return self.runsPage[mafRunId]
        match = (self.runs['mafRunId'] == mafRunId)
        mafDir = self.runs[match]['mafDir'][0]
        runName = self.runs[match]['opsimRun'][0]
        if runName == 'NULL':
            runName = None
        self.runsPage[mafRunId] = MafRunResults(mafDir, runName)
        return self.runsPage[mafRunId]


