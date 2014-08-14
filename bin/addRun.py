#! /usr/bin/env python
import os,argparse
import lsst.sims.maf.db as db


if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description="Add a MAF run to the tracking database.")
    parser.add_argument("mafDir", type=str, help="Directory containing MAF outputs.")
    parser.add_argument("--mafComment", type=str, default='NULL', help="Comment on MAF run.")
    parser.add_argument("--opsimRun", type=str, default='NULL', help="Opsim Run Name.")
    parser.add_argument("--opsimComment", type=str, default='NULL', help="Comment on OpSim run.")
    defaultdb = os.path.join(os.getenv('SIMS_MAF_DIR'), 'bin', 'trackingDb_sqlite.db')
    defaultdb = 'sqlite:///' + defaultdb
    parser.add_argument("--trackingDb", type=str, default=defaultdb, help="Tracking database dbAddress.")
    args = parser.parse_args()

    if not os.path.isdir(args.mafDir):
        print 'There is no directory containing MAF outputs at %s.' %(args.mafDir)
        print 'Exiting.'
        exit(-1)

    trackingDb = db.TrackingDb(trackingDbAddress=args.trackingDb)    
    
    # If opsim run name or comment not set, try to set it from maf outputs.
    opsimRun = args.opsimRun
    opsimComment = args.opsimComment
    if (opsimRun == 'NULL') or (opsimComment == 'NULL'):
        if os.path.isfile(os.path.join(args.mafDir, 'configSummary.txt')):
            file = open(os.path.join(args.mafDir, 'configSummary.txt'))
            for line in file:
                tmp = line.split()
                if tmp[0].startswith('RunName'):
                    tmp_opsimRun = ' '.join(tmp[1:])
                if tmp[0].startswith('RunComment'):
                    tmp_opsimComment = ' '.join(tmp[1:])
    if opsimRun == 'NULL':
        opsimRun = tmp_opsimRun
    if opsimComment == 'NULL':
        opsimComment = tmp_opsimComment

    print 'Adding to tracking database at %s:' %(args.trackingDb)
    print ' MafDir = %s' %(args.mafDir)
    print ' MafComment = %s' %(args.mafComment)
    print ' OpsimRun = %s' %(opsimRun)
    print ' OpsimComment = %s' %(opsimComment)
                    
    trackingDb.addRun(opsimRun, opsimComment, args.mafComment, args.mafDir)
    
    exit()
