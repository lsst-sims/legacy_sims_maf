#! /usr/bin/env python
import os, argparse
import lsst.sims.maf.db as db


if __name__ == "__main__":
      
    parser = argparse.ArgumentParser(description="Remove a MAF run from the tracking database.")
    parser.add_argument("mafRunId", type=int, help="MAF Run ID from the tracking database.")
    defaultdb = os.path.join(os.getenv('SIMS_MAF_DIR'), 'bin', 'trackingDb_sqlite.db')
    defaultdb = 'sqlite:///' + defaultdb
    parser.add_argument("--trackingDb", type=str, default=defaultdb, help="Tracking database dbAddress.")
    args = parser.parse_args()

    trackingDb = db.TrackingDb(trackingDbAddress=args.trackingDb)    
                
    trackingDb.delRun(int(args.mafRunId))
