#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import lsst.sims.maf.db as db


def addToDatabase(mafDir, trackingDbFile,
                  opsimRun=None, opsimComment=None, opsimDate=None,
                  mafComment=None, mafDate=None, dbFile=None):
    """Adds information about a MAF analysis run to a MAF tracking database.
    
    Parameters
    ----------
    mafDir : str
        Name of the directory where the MAF results are located.
    trackingDb : str
        Full filename (+path) to the tracking database storing the MAF run information.
    opsimRun : str, opt
        Name of the opsim run. If not provided, will attempt to use runName from confSummary.txt.
    opsimComment : str, opt
        Comment about the opsim run. If not provided, will attempt to use runComment from confSummary.txt.
    opsimDate : str, opt
        Date of the opsim run. If not provided, will attempt to use date from confSummary.txt.
    mafComment : str, opt
        Comment about the MAF analysis. If not provided, no comment will be recorded.
    mafDate : str, opt
        Date of the MAF analysis. If not provided, will attempt to use date from confSummary.txt.
    dbFile : str, opt
        Relative path to the opsim database file. If not provided, no location will be recorded.
    """
    mafDir = os.path.abspath(args.mafDir)
    if not os.path.isdir(mafDir):
        raise ValueError('There is no directory containing MAF outputs at %s.' % (mafDir))

    trackingDb = db.TrackingDb(database=trackingDbFile)
    if (opsimRun is None) or (opsimComment is None) or (opsimDate is None) or (mafDate is None):
        if os.path.isfile(os.path.join(mafDir, 'configSummary.txt')):
            file = open(os.path.join(mafDir, 'configSummary.txt'))
            for line in file:
                tmp = line.split()
                if tmp[0].startswith('RunName'):
                    if opsimRun is None:
                        opsimRun = ' '.join(tmp[1:])
                if tmp[0].startswith('RunComment'):
                    if opsimComment is None:
                        opsimComment = ' '.join(tmp[1:])
                # MAF Date may be in a line with "MafDate" (new configs)
                #  or at the end of "MAFVersion" (old configs).
                if tmp[0].startswith('MAFDate') or tmp[0].startswith('MAFVersion'):
                    print(tmp, tmp[-1])
                    if mafDate is None:
                        mafDate = tmp[-1]
                        # And convert formats to '-' (again, multiple versions of configs).
                    if len(mafDate.split('/')) > 1:
                        t = mafDate.split('/')
                        mafDate = '-'.join(['20' + t[2], t[1], t[0]])
                if tmp[0].startswith('OpsimDate') or tmp[0].startswith('OpsimVersion'):
                    if opsimDate is None:
                        opsimDate = tmp[-2]
                    if len(opsimDate.split('/')) > 1:
                        t = opsimDate.split('/')
                        opsimDate = '-'.join(['20' + t[2], t[1], t[0]])

    print('Adding to tracking database at %s:' % (trackingDbFile))
    print(' MafDir = %s' % (mafDir))
    print(' MafComment = %s' % (mafComment))
    print(' OpsimRun = %s' % (opsimRun))
    print(' OpsimComment = %s' % (opsimComment))
    print(' OpsimDate = %s' % (opsimDate))
    print(' MafDate = %s' % (mafDate))
    print(' DbFile = %s' % (dbFile))
    runId = trackingDb.addRun(opsimRun, opsimComment, mafComment, mafDir, opsimDate, mafDate, dbFile)
    print('Used MAF RunID %d' % (runId))
    trackingDb.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add a MAF run to the tracking database.")
    parser.add_argument("mafDir", type=str, help="Directory containing MAF outputs.")
    parser.add_argument("-c", "--mafComment", type=str, default=None, help="Comment on MAF run.")
    parser.add_argument("--opsimRun", type=str, default=None, help="Opsim Run Name.")
    parser.add_argument("--opsimComment", type=str, default=None, help="Comment on OpSim run.")
    parser.add_argument("--opsimDate", type=str, default=None, help="Date Opsim was run")
    parser.add_argument("--mafDate", type=str, default=None, help="Date MAF was run")
    parser.add_argument("--dbFile", type=str, default='None', help="Opsim Sqlite filename")
    defaultdb = 'trackingDb_sqlite.db'
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb,
                        help="Tracking database filename. Default is %s, in the current directory."
                             % (defaultdb))
    args = parser.parse_args()

    addToDatabase(args.mafDir, args.trackingDb, args.opsimRun, args.opsimComment, args.opsimDate,
                  args.mafComment, args.mafDate, args.dbFile)
