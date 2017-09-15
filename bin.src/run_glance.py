#!/usr/bin/env python

from __future__ import print_function

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.batches import glanceBatch
from lsst.sims.maf.batches import ColMapDict
import argparse

def connectDb(dbfile):
    version = db.testOpsimVersion(dbfile)
    if version == "Unknown":
        opsdb = db.Database(dbfile)
        colmap = ColMapDict('barebones')
    elif version == "V3":
        opsdb = db.OpsimDatabaseV3(dbfile)
        colmap = ColMapDict('OpsimV3')
    elif version == "V4":
        opsdb = db.OpsimDatabaseV4(dbfile)
        colmap = ColMapDict('OpsimV4')
    return opsdb, colmap


def runGlance(opsdb, colmap,  outDir='Glance', runName='opsim'):

    gb = glanceBatch(colmap=colmap, runName=runName)
    resultsDb = db.ResultsDb(outDir=outDir)

    group = metricBundles.MetricBundleGroup(gb, opsdb, outDir=outDir, resultsDb=resultsDb)

    group.runAll()
    group.plotAll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the survey at a glance bundle.")
    parser.add_argument("dbfile", type=str, help="Sqlite file of observations (full path).")
    parser.add_argument("--runName", type=str, default=None, help="Run name."
                                                                  "Default is based on dbfile name.")
    parser.add_argument("--outDir", type=str, default=None, help="Output directory."
                                                                 "Default is runName/glance.")
    args = parser.parse_args()

    opsdb, colmap = connectDb(args.dbfile)

    if args.runName is None:
        if runName is None:
            runName = os.path.basename(dbFile).replace('_sqlite.db', '')
            runName = runName.replace('.db', '')


    if args.outDir is None:
        outDir = os.path.join(args.runName, "glance")
    else:
        outDir = args.outDir

    runGlance(opsdb, colmap, outDir=outDir, runName=args.runName)
