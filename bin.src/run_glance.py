#!/usr/bin/env python

from __future__ import print_function

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.bundles import glanceBundle
from lsst.sims.maf.bundles import ColMapDict
import argparse

def connectDb(dbfile):
    version = db.testOpsimVersion(dbfile)
    if version == "Unknown":
        conn = db.Database(dbfile)
        colmap = ColMapDict('barebones')
    elif version == "V3":
        conn = db.OpsimDatabaseV3(dbfile)
        colmap = ColMapDict('OpsimV3')
    elif version == "V4":
        conn = db.OpsimDatabaseV4(dbfile)
        colmap = ColMapDict('OpsimV4')
    return conn, colmap


def runGlance(conn, colmap,  outDir='Glance', runName='runname'):

    gb = glanceBundle(colmap=colmap, runName=runName)
    resultsDb = db.ResultsDb(outDir=outDir)

    group = metricBundles.MetricBundleGroup(gb, conn, outDir=outDir, resultsDb=resultsDb)

    group.runAll()
    group.plotAll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the survey at a glance bundle.")
    parser.add_argument("dbfile", type=str, help="Sqlite file of observations (full path).")
    parser.add_argument("--runName", type=str, default="opsim", help="Run name.")
    parser.add_argument("--outDir", type=str, default="Output", help="Output directory.")
    args = parser.parse_args()

    conn, colmap = connectDb(args.dbfile)
    runGlance(conn, colmap, outDir=args.outDir, runName=args.runName)
