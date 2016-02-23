#!/usr/bin/env python

import argparse
import lsst.sims.maf.db as db

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Remove a MAF run from the tracking database.")
    parser.add_argument("mafRunId", type=int, help="MAF Run ID from the tracking database.")
    defaultdb = 'trackingDb_sqlite.db'
    parser.add_argument("-t", "--trackingDb", type=str, default=defaultdb, help="Tracking database filename.")
    args = parser.parse_args()

    # If db is just a filename, assume sqlite and prepend address string

    trackingDb = db.TrackingDb(database=args.trackingDb)
    trackingDb.delRun(int(args.mafRunId))
