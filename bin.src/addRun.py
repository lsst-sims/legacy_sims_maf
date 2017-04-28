#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import lsst.sims.maf.db as db


def addToDatabase(mafDir, trackingDbFile, opsimGroup=None,
                  opsimRun=None, opsimComment=None,
                  mafComment=None,  dbFile=None):
    """Adds information about a MAF analysis run to a MAF tracking database.
    
    Parameters
    ----------
    mafDir : str
        Relative path to the directory where the MAF results are located.
    trackingDb : str
        Full filename (+path) to the tracking database storing the MAF run information.
    opsimGroup: str, opt
        Name to use to group this run with other opsim runs. Default None.
    opsimRun : str, opt
        Name of the opsim run. If not provided, will attempt to use runName from confSummary.txt.
    opsimComment : str, opt
        Comment about the opsim run. If not provided, will attempt to use runComment from confSummary.txt.
    mafComment : str, opt
        Comment about the MAF analysis. If not provided, no comment will be recorded.
    dbFile : str, opt
        Relative path + name of the opsim database file. If not provided, no location will be recorded.
    """
    mafDir = os.path.relpath(mafDir)
    if not os.path.isdir(mafDir):
        raise ValueError('There is no directory containing MAF outputs at %s.' % (mafDir))

    trackingDb = db.TrackingDb(database=trackingDbFile)
    autoOpsimRun = None
    autoOpsimComment = None
    opsimVersion = None
    opsimDate = None
    mafVersion = None
    mafDate = None
    if os.path.isfile(os.path.join(mafDir, 'configSummary.txt')):
        file = open(os.path.join(mafDir, 'configSummary.txt'))
        for line in file:
            tmp = line.split()
            if tmp[0].startswith('RunName'):
                autoOpsimRun = ' '.join(tmp[1:])
            if tmp[0].startswith('RunComment'):
                autoOpsimComment = ' '.join(tmp[1:])
            # MAF Date may be in a line with "MafDate" (new configs)
            #  or at the end of "MAFVersion" (old configs).
            if tmp[0].startswith('MAFDate'):
                mafDate = tmp[-1]
            if tmp[0].startswith('MAFVersion'):
                mafVersion = tmp[1]
                if len(tmp) > 2:
                    mafDate = tmp[-1]
            if tmp[0].startswith('OpsimDate'):
                opsimDate = tmp[-2]
            if tmp[0].startswith('OpsimVersion'):
                opsimVersion = tmp[1]
                if len(tmp) > 2:
                    opsimDate = tmp[-2]
    # And convert formats to '-' (again, multiple versions of configs).
    if len(mafDate.split('/')) > 1:
        t = mafDate.split('/')
        if len(t[2]) == 2:
            t[2] = '20' + t[2]
        mafDate = '-'.join([t[2], t[1], t[0]])
    if len(opsimDate.split('/')) > 1:
        t = opsimDate.split('/')
        if len(t[2]) == 2:
            t[2] = '20' + t[2]
        opsimDate = '-'.join([t[2], t[1], t[0]])

    if opsimRun is None:
        opsimRun = autoOpsimRun
    if opsimComment is None:
        opsimComment = autoOpsimComment

    print('Adding to tracking database at %s:' % (trackingDbFile))
    print(' MafDir = %s' % (mafDir))
    print(' MafComment = %s' % (mafComment))
    print(' OpsimGroup = %s' % (opsimGroup))
    print(' OpsimRun = %s' % (opsimRun))
    print(' OpsimComment = %s' % (opsimComment))
    print(' OpsimVersion = %s' % (opsimVersion))
    print(' OpsimDate = %s' % (opsimDate))
    print(' MafVersion = %s' % (mafVersion))
    print(' MafDate = %s' % (mafDate))
    print(' Opsim dbFile = %s' % (dbFile))
    runId = trackingDb.addRun(opsimGroup=opsimGroup, opsimRun=opsimRun, opsimComment=opsimComment,
                              opsimVersion=opsimVersion, opsimDate=opsimDate,
                              mafComment=mafComment, mafVersion=mafVersion, mafDate=mafDate,
                              mafDir=mafDir, dbFile=dbFile)
    print('Used MAF RunID %d' % (runId))
    trackingDb.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add a MAF run to the tracking database.")
    parser.add_argument("mafDir", type=str, help="Directory containing MAF outputs.")
    parser.add_argument("-c", "--mafComment", type=str, default=None, help="Comment on MAF analysis.")
    parser.add_argument("--group", type=str, default=None, help="Opsim Group name.")
    parser.add_argument("--opsimRun", type=str, default=None, help="Opsim Run Name.")
    parser.add_argument("--opsimComment", type=str, default=None, help="Comment on OpSim run.")
    parser.add_argument("--dbFile", type=str, default='None', help="Opsim Sqlite filename")
    defaultdb = 'trackingDb_sqlite.db'
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb,
                        help="Tracking database filename. Default is %s, in the current directory."
                             % (defaultdb))
    args = parser.parse_args()

    addToDatabase(args.mafDir, args.trackingDb, args.group, args.opsimRun,
                  args.opsimComment, args.mafComment, args.dbFile)
