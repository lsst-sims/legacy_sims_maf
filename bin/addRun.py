#! /usr/bin/env python
import os,argparse
import lsst.sims.maf.db as db


if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description="Add a MAF run to the tracking database.")
    parser.add_argument("mafDir", type=str, help="Directory containing MAF outputs.")
    parser.add_argument("-c", "--mafComment", type=str, default=None, help="Comment on MAF run.")
    parser.add_argument("--opsimRun", type=str, default=None, help="Opsim Run Name.")
    parser.add_argument("--opsimComment", type=str, default=None, help="Comment on OpSim run.")
    defaultdb = 'trackingDb_sqlite.db'
    defaultdb = 'sqlite:///' + defaultdb
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb, help="Tracking database dbAddress.")
    args = parser.parse_args()

    mafDir = os.path.realpath(args.mafDir)
    if not os.path.isdir(mafDir):
        print 'There is no directory containing MAF outputs at %s.' %(mafDir)
        print 'Exiting.'
        exit(-1)

    trackingDbAddress = args.trackingDb
    if not trackingDbAddress.startswith('sqlite:///'):
        trackingDbAddress = 'sqlite:///' + trackingDbAddress
    trackingDb = db.TrackingDb(trackingDbAddress=trackingDbAddress)    
    
    # If opsim run name or comment not set, try to set it from maf outputs.
    opsimRun = args.opsimRun
    opsimComment = args.opsimComment
    if (opsimRun is None) or (opsimComment is None):
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

    print 'Adding to tracking database at %s:' %(args.trackingDb)
    print ' MafDir = %s' %(mafDir)
    print ' MafComment = %s' %(args.mafComment)
    print ' OpsimRun = %s' %(opsimRun)
    print ' OpsimComment = %s' %(opsimComment)
                    
    trackingDb.addRun(opsimRun, opsimComment, args.mafComment, mafDir)
    
